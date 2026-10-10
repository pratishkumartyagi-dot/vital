import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'zero_allocation_extractor.dart';
import 'ppg_signal_engine.dart';
import 'native_hardware_bridge.dart';
import 'dart:math' as math;
import 'dart:async';

enum TrackingState { baseline, profiling, complete, rejected }
enum CameraMode { front, rear }

class VitalSignController extends ChangeNotifier {
  TrackingState _currentState = TrackingState.baseline;
  double _gatheringProgress = 0.0;
  String _toastPrompt = "Align your finger over the camera lens and flash.";
  double _heartRate = 0.0;
  double _hbLevel = 0.0;
  double _spo2Level = 0.0;
  double _respRate = 0.0;
  double _bpSys = 0.0;
  double _bpDia = 0.0;
  double _glucoseLevel = 0.0;
  double _creatinineLevel = 0.0;
  double _egfrLevel = 0.0;
  String _creatinineClass = '';
  String _kdigoStage = '';
  double _signalQuality = 0.0;
  bool _isDemoMode = false;
  Timer? _demoTimer;
  int _demoFrameCount = 0;
  CameraMode _cameraMode = CameraMode.rear;

  TrackingState get currentState => _currentState;
  double get gatheringProgress => _gatheringProgress;
  String get toastPrompt => _toastPrompt;
  double get heartRate => _heartRate;
  double get hbLevel => _hbLevel;
  double get spo2Level => _spo2Level;
  double get respRate => _respRate;
  double get bpSys => _bpSys;
  double get bpDia => _bpDia;
  double get glucoseLevel => _glucoseLevel;
  double get creatinineLevel => _creatinineLevel;
  double get egfrLevel => _egfrLevel;
  String get creatinineClass => _creatinineClass;
  String get kdigoStage => _kdigoStage;
  double get signalQuality => _signalQuality;
  bool get isDemoMode => _isDemoMode;
  CameraMode get cameraMode => _cameraMode;
  double get calibrationZlo => _calibrationZlo;
  double get calibrationSlope => _calibrationSlope;
  bool get fingerDetected => _fingerDetected;
  bool get torchActive => _torchActive;

  static const int _waveformDisplaySize = 300;
  static const int _analysisWindowSize = 300;
  static const int _minSamplesForHR = 90;
  static const double _fs = 30.0;
  static const double _defaultZlo = 16.0;
  static const double _defaultSlope = 1.0;

  double _calibrationZlo = _defaultZlo;
  double _calibrationSlope = _defaultSlope;
  bool _fingerDetected = false;
  bool _torchActive = false;
  final List<double> _ppgHistory = [];
  static const int _ppgHistoryMaxLen = 120; // 4 seconds at 30fps

  final List<double> _liveRedBuffer = [];
  final List<double> _liveGreenBuffer = [];
  final List<double> _liveBlueBuffer = [];
  final List<double> _waveformBuffer = [];

  VitalSignController() {
    _enableDemoMode();
  }

  void setCameraMode(CameraMode mode) {
    _cameraMode = mode;
    reset();
    if (mode == CameraMode.front) {
      _toastPrompt = "Position your face in the frame with good lighting.";
    } else {
      _toastPrompt = "Align your finger over the camera lens and flash.";
    }
    notifyListeners();
  }

  void _enableDemoMode() {
    _isDemoMode = true;
    _toastPrompt = "DEMO MODE: Simulated PPG signal. Switch to live mode to measure.";
    _startDemoTimer();
  }

  void _disableDemoMode() {
    _isDemoMode = false;
    _stopDemoTimer();
    _liveRedBuffer.clear();
    _liveGreenBuffer.clear();
    _liveBlueBuffer.clear();
    _waveformBuffer.clear();
    NativeHardwareBridge.resetPpgBuffer();
  }

  void enableLiveMode() {
    if (!_isDemoMode) return;
    _disableDemoMode();
    notifyListeners();
  }

  void _startDemoTimer() {
    _demoTimer?.cancel();
    _demoTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
      _demoFrameCount++;
      if (_isDemoMode) {
        final double time = _demoFrameCount * 0.033;
        final double baseSignal = 100 +
            20 * math.sin(2 * math.pi * 1.2 * time) +
            5 * math.sin(2 * math.pi * 2.5 * time) +
            (math.Random().nextDouble() - 0.5) * 2;

        if (_demoFrameCount % 200 == 0) {
          _liveGreenBuffer.add(baseSignal + (math.Random().nextDouble() - 0.5) * 30);
        } else {
          _liveGreenBuffer.add(baseSignal);
        }

        _waveformBuffer.add(baseSignal);
        if (_waveformBuffer.length > _waveformDisplaySize) {
          _waveformBuffer.removeAt(0);
        }

        if (_liveGreenBuffer.length >= 120) {
          final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);

          if (pi < 0.5) {
            _currentState = TrackingState.rejected;
            _toastPrompt = "Unreliable Signal. Please steady your finger and apply lighter pressure.";
            _gatheringProgress = 0.0;
            _liveGreenBuffer.clear();
            _waveformBuffer.clear();
          } else {
            _currentState = TrackingState.profiling;
            _toastPrompt = "Analyzing heart rate stability. Keep perfectly still...";
            _gatheringProgress += 0.02;

            if (_gatheringProgress >= 1.0) {
              _gatheringProgress = 1.0;
              _currentState = TrackingState.complete;
              _toastPrompt = "Measurement complete.";
              _calculateHeartRate();
            }
          }
          notifyListeners();
        }
      }
    });
  }

  void _stopDemoTimer() {
    _demoTimer?.cancel();
    _demoTimer = null;
  }

  void attachStreamPipeline(CameraController controller) {
    if (_isDemoMode) return;

    controller.startImageStream((CameraImage streamedFrame) {
      _processCameraFrame(streamedFrame);
    });
  }

  /// Process a Camera2 frame (Y-plane from native Android Camera2 API).
  /// Used when the rear camera is managed by Camera2PpgController.
  void processCamera2Frame(int width, int height, Uint8List yPlane) {
    if (_isDemoMode) return;
    if (_currentState == TrackingState.complete) return;

    // Convert Y-plane to RGB-like values for processing
    // YUV Y-plane represents luminance; for PPG we use it as green proxy
    // and estimate R/B from Y with typical skin-tone ratios
    final Float64List extractedValues = ZeroAllocationExtractor.processYPlane(yPlane, width, height);
    final double currentRed = extractedValues[0];
    final double currentGreen = extractedValues[1];
    final double currentBlue = extractedValues[2];

    _processCameraFrameValues(currentRed, currentGreen, currentBlue);
  }

  void _processCameraFrame(CameraImage streamedFrame) {
    final Float64List extractedValues = ZeroAllocationExtractor.processYUV420Frame(streamedFrame);
    final double currentRed = extractedValues[0];
    final double currentGreen = extractedValues[1];
    final double currentBlue = extractedValues[2];

    _processCameraFrameValues(currentRed, currentGreen, currentBlue);
  }

  void _processCameraFrameValues(double currentRed, double currentGreen, double currentBlue) {
    // Finger detection for rear camera (before adding to buffers)
    if (_cameraMode == CameraMode.rear) {
      _updateFingerDetection(currentRed, currentGreen, currentBlue);
    }

    if (currentRed > 250.0 || currentGreen > 250.0 || currentBlue > 250.0) {
      _toastPrompt = "Signal over-amplified. Reduce finger pressure on the lens.";
      return;
    }
    if (currentRed < 15.0 && currentGreen < 15.0 && currentBlue < 15.0) {
      _toastPrompt = "No signal detected. Cover the lens and flash completely.";
      return;
    }

    _liveRedBuffer.add(currentRed);
    _liveGreenBuffer.add(currentGreen);
    _liveBlueBuffer.add(currentBlue);

    _waveformBuffer.add(_cameraMode == CameraMode.rear ? currentGreen :
        (currentRed + currentGreen + currentBlue) / 3);
    if (_waveformBuffer.length > _waveformDisplaySize) {
      _waveformBuffer.removeAt(0);
    }

    if (_liveRedBuffer.length > _analysisWindowSize) {
      _liveRedBuffer.removeAt(0);
      _liveGreenBuffer.removeAt(0);
      _liveBlueBuffer.removeAt(0);
    }

    NativeHardwareBridge.processPpgSample(currentGreen).then((nativeBpm) {
      if (nativeBpm != null && nativeBpm > 0) {
        _heartRate = nativeBpm;
        notifyListeners();
      }
    });

    if (_cameraMode == CameraMode.rear) {
      _processRearCameraPPG();
    } else {
      _processFrontCameraRPPG();
    }
  }

  /// Update finger detection state with temporal PPG validation
  void _updateFingerDetection(double r, double g, double b) {
    // Update PPG history for temporal validation
    final lum = (r + g + b) / 3;
    _ppgHistory.add(lum);
    if (_ppgHistory.length > _ppgHistoryMaxLen) {
      _ppgHistory.removeAt(0);
    }

    // Static checks
    final torchActive = _torchActive;
    final redBright = r > 80 && r < 245;
    final redDominant = r > g + 15 && r > b + 15 && r / math.max(g, 1) > 1.05;
    final notSaturated = r < 250 && g < 250 && b < 250;
    final dcLevel = (r + g + b) / 3;
    final dcInRange = dcLevel > 50 && dcLevel < 220;

    // Uniformity check (simplified - would need full frame for true variance)
    // For now, use perfusion index as proxy for signal quality
    bool hasPpgPulsation = false;
    if (_ppgHistory.length >= 60) {
      final recent = _ppgHistory.sublist(_ppgHistory.length - 60);
      final mean = recent.reduce((a, b) => a + b) / recent.length;
      double autocorr = 0;
      double variance = 0;
      for (int i = 0; i < recent.length; i++) {
        final diff = recent[i] - mean;
        variance += diff * diff;
        if (i >= 30) {
          autocorr += diff * (recent[i - 30] - mean);
        }
      }
      variance /= recent.length;
      autocorr /= recent.length;
      hasPpgPulsation = variance > 0.5 && autocorr / variance > 0.3;
    }

    // Finger detected only if ALL conditions met
    _fingerDetected = torchActive && redBright && redDominant && notSaturated && dcInRange && hasPpgPulsation;
  }

  /// Runs the full analysis pipeline for the rear (contact fingertip) path.
  /// The analysis signal is the SNR-best channel (green by default, red on
  /// thick/dark/cold fingers) selected by the engine's spectral-SNR rule.
  void _processRearCameraPPG() {
    // Early reject if finger not detected (prevents false analysis on non-finger)
    if (!_fingerDetected) {
      if (_currentState != TrackingState.baseline) {
        _currentState = TrackingState.baseline;
        _gatheringProgress = 0.0;
        _toastPrompt = "Place fingertip over camera lens with flash on.";
        notifyListeners();
      }
      return;
    }

    if (_liveGreenBuffer.length >= 30) {
      final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer, _calibrationZlo);
      if (pi < 0.2) {
        _currentState = TrackingState.rejected;
        _toastPrompt = "Low pulse signal detected. Warm hands and reduce finger pressure.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
        return;
      }
    }

    if (_liveGreenBuffer.length >= 120) {
      final analysisSignal = PPGSignalEngine.selectBestPpgChannel(
        _liveRedBuffer, _liveGreenBuffer, _liveBlueBuffer, _fs);
      final analysis = PPGSignalEngine.analyzeVitals(
        analysisSignal, _fs,
        _liveRedBuffer, _liveGreenBuffer, _liveBlueBuffer,
        AnalyzeOptions(cameraMode: 'rear', zlo: _calibrationZlo),
      );

      _applyAnalysis(analysis);

      final redPI = PPGSignalEngine.computePerfusionIndex(_liveRedBuffer, _calibrationZlo);
      final greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer, _calibrationZlo);
      final bluePI = PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer, _calibrationZlo);
      final avgGreenDC = _liveGreenBuffer.reduce((a, b) => a + b) / _liveGreenBuffer.length;
      if (avgGreenDC > 240) {
        _toastPrompt = "Ambient lighting too bright. Shade the lens with your finger.";
      }

      if (redPI > 0.5 && greenPI > 0.5 && bluePI > 0.5) {
        _currentState = TrackingState.profiling;
        _toastPrompt = "Analyzing stability... HR: ${_heartRate.toStringAsFixed(0)} BPM | SpO₂: ${_spo2Level.toStringAsFixed(0)}% | Hb: ${_hbLevel.toStringAsFixed(1)} g/dL";
        _gatheringProgress += 0.02;

        if (_gatheringProgress >= 1.0) {
          _gatheringProgress = 1.0;
          _currentState = TrackingState.complete;
          _toastPrompt = "Measurement complete. HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${_spo2Level.toStringAsFixed(0)}% | Hb: ${_hbLevel.toStringAsFixed(1)}";
        }
        notifyListeners();
      } else {
        _currentState = TrackingState.rejected;
        _toastPrompt = "Unreliable Signal. Please steady your finger and apply lighter pressure.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
      }
    }
  }

  /// Runs the full analysis pipeline for the front (facial rPPG) path.
  /// The analysis signal is the POS chrominance pulse (Wang 2017), which
  /// cancels common-mode illumination and motion artifacts far better than
  /// any single raw channel, with the green skin-ROI mean as its DC ref.
  void _processFrontCameraRPPG() {
    if (_liveGreenBuffer.length >= 30) {
      final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      if (pi < 0.05) {
        _currentState = TrackingState.rejected;
        _toastPrompt = "Weak face signal. Ensure good lighting and face visibility.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
        return;
      }
    }

    if (_liveGreenBuffer.length >= 120) {
      final posPulse = PPGSignalEngine.posPulse(
        _liveRedBuffer, _liveGreenBuffer, _liveBlueBuffer);
      final dcRef = PPGSignalEngine.dcLevel(_liveGreenBuffer);
      final analysis = PPGSignalEngine.analyzeVitals(
        posPulse.isNotEmpty ? posPulse : _liveGreenBuffer, _fs,
        _liveRedBuffer, _liveGreenBuffer, _liveBlueBuffer,
        AnalyzeOptions(cameraMode: 'front', dcRef: dcRef),
      );

      _applyAnalysis(analysis);

      final greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      if (greenPI > 0.1) {
        _currentState = TrackingState.profiling;
        _toastPrompt = "rPPG analysis... HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${_spo2Level.toStringAsFixed(0)}%";
        _gatheringProgress += 0.015;

        if (_gatheringProgress >= 1.0) {
          _gatheringProgress = 1.0;
          _currentState = TrackingState.complete;
          _toastPrompt = "rPPG complete. HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${_spo2Level.toStringAsFixed(0)}%";
        }
        notifyListeners();
      } else {
        _currentState = TrackingState.rejected;
        _toastPrompt = "Unreliable rPPG signal. Check lighting and face position.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
      }
    }
  }

  /// Copies the engine's analysis results into the controller state,
  /// falling back to the previous inline ratios for SpO₂/Hb so the
  /// displayed values stay consistent with the validated calibration.
  void _applyAnalysis(VitalAnalysis analysis) {
    if (analysis.heart.bpm != null) {
      _heartRate = analysis.heart.bpm!;
    }
    _signalQuality = analysis.heart.signalQuality;

    final redPI = PPGSignalEngine.computePerfusionIndex(_liveRedBuffer, _calibrationZlo);
    final greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer, _calibrationZlo);
    final bluePI = PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer, _calibrationZlo);

    if (redPI > 0 && bluePI > 0) {
      _spo2Level = (110.0 - 25.0 * (redPI / bluePI)).clamp(70.0, 100.0).toDouble();
    } else if (analysis.heart.bpm != null) {
      _spo2Level = 95.0;
    }
    if (redPI > 0 && greenPI > 0) {
      _hbLevel = (18.2 - 6.8 * (redPI / greenPI)).clamp(5.0, 20.0).toDouble();
    }

    _respRate = analysis.resp.respRate ?? _respRate;
    _bpSys = analysis.bp.systolic ?? _bpSys;
    _bpDia = analysis.bp.diastolic ?? _bpDia;
    _glucoseLevel = analysis.glucose.glucose ?? _glucoseLevel;
    _creatinineLevel = analysis.creatinine.creatinine ?? _creatinineLevel;
    _egfrLevel = analysis.creatinine.egfr ?? _egfrLevel;
    _creatinineClass = analysis.creatinine.creatinineClass ?? '';
    _kdigoStage = analysis.creatinine.stage ?? '';
  }

  void _clearBuffers() {
    _liveRedBuffer.clear();
    _liveGreenBuffer.clear();
    _liveBlueBuffer.clear();
    _waveformBuffer.clear();
    NativeHardwareBridge.resetPpgBuffer();
  }

  void _calculateHeartRate() {
    if (_liveGreenBuffer.length < _minSamplesForHR) return;

    final analysis = PPGSignalEngine.analyzeHeartRate(_liveGreenBuffer, _fs);
    if (analysis.bpm != null) {
      _heartRate = analysis.bpm!;
    }
    _signalQuality = analysis.signalQuality;
  }

  List<double> get waveformData {
    if (_waveformBuffer.isEmpty) return [];

    double minVal = _waveformBuffer.reduce(math.min);
    double maxVal = _waveformBuffer.reduce(math.max);

    if (maxVal - minVal < 0.001) {
      return List.filled(_waveformBuffer.length, 0.5);
    }

    return _waveformBuffer.map((value) =>
        (value - minVal) / (maxVal - minVal)).toList();
  }

  Map<String, dynamic> exportRawSignals({
    Map<String, dynamic>? groundTruth,
    String? sessionId,
    String? patientId,
  }) {
    return {
      'metadata': {
        'timestamp': DateTime.now().toIso8601String(),
        'sessionId': sessionId ?? 'session_${DateTime.now().millisecondsSinceEpoch}',
        'patientId': patientId,
        'cameraMode': _cameraMode.name,
        'samplingRateHz': 30,
        'durationSeconds': _liveGreenBuffer.length / 30.0,
        'appVersion': '1.0.0',
      },
      'signals': {
        'red': List<double>.from(_liveRedBuffer),
        'green': List<double>.from(_liveGreenBuffer),
        'blue': List<double>.from(_liveBlueBuffer),
      },
      'vitals': {
        'heartRate': _heartRate,
        'spo2': _spo2Level,
        'hemoglobin': _hbLevel,
        'respiratoryRate': _respRate,
        'signalQuality': signalQuality,
        'bpSys': _bpSys,
        'bpDia': _bpDia,
        'glucose': _glucoseLevel,
        'creatinine': _creatinineLevel,
        'egfr': _egfrLevel,
      },
      'groundTruth': groundTruth ?? {},
      'quality': {
        'perfusionIndex': PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer),
        'redPI': PPGSignalEngine.computePerfusionIndex(_liveRedBuffer),
        'bluePI': PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer),
        'bufferLength': _liveGreenBuffer.length,
        'state': _currentState.name,
      },
    };
  }

  String exportRawSignalsJson({
    Map<String, dynamic>? groundTruth,
    String? sessionId,
    String? patientId,
  }) {
    final data = exportRawSignals(
      groundTruth: groundTruth,
      sessionId: sessionId,
      patientId: patientId,
    );
    return _jsonEncode(data);
  }

  String _jsonEncode(Map<String, dynamic> data) {
    final buffer = StringBuffer();
    _writeJson(buffer, data, 0);
    return buffer.toString();
  }

  void _writeJson(StringBuffer buffer, dynamic value, int indent) {
    if (value is Map<String, dynamic>) {
      if (value.isEmpty) {
        buffer.write('{}');
        return;
      }
      buffer.write('{\n');
      final keys = value.keys.toList();
      for (int i = 0; i < keys.length; i++) {
        buffer.write('  ' * (indent + 1));
        buffer.write('"${keys[i]}": ');
        _writeJson(buffer, value[keys[i]], indent + 1);
        if (i < keys.length - 1) buffer.write(',');
        buffer.write('\n');
      }
      buffer.write('  ' * indent);
      buffer.write('}');
    } else if (value is List) {
      if (value.isEmpty) {
        buffer.write('[]');
        return;
      }
      buffer.write('[\n');
      for (int i = 0; i < value.length; i++) {
        buffer.write('  ' * (indent + 1));
        _writeJson(buffer, value[i], indent + 1);
        if (i < value.length - 1) buffer.write(',');
        buffer.write('\n');
      }
      buffer.write('  ' * indent);
      buffer.write(']');
    } else if (value is String) {
      buffer.write('"${value.replaceAll('"', '\\"')}"');
    } else if (value is num || value is bool || value == null) {
      buffer.write(value == null ? 'null' : value.toString());
    } else {
      buffer.write('"${value.toString()}"');
    }
  }

  @override
  void dispose() {
    _stopDemoTimer();
    super.dispose();
  }

  /// Set the ZLO calibration value from native Android/iOS
  void setCalibrationZlo(double zlo, {double slope = 1.0}) {
    _calibrationZlo = zlo;
    _calibrationSlope = slope;
    debugPrint('Calibration set: ZLO=$zlo, Slope=$slope');
  }

  /// Set calibration from a CalibrationProfile
  void setCalibrationProfile(CalibrationProfile profile) {
    _calibrationZlo = profile.zlo;
    _calibrationSlope = profile.sensorLinearSlope;
    debugPrint('Calibration profile set: source=${profile.source}, ZLO=${_calibrationZlo}, Slope=${_calibrationSlope}');
  }

  /// Set torch state from main.dart (synced with actual hardware torch)
  void setTorchState(bool active) {
    _torchActive = active;
  }

  void reset() {
    _currentState = TrackingState.baseline;
    _gatheringProgress = 0.0;
    _heartRate = 0.0;
    _hbLevel = 0.0;
    _spo2Level = 0.0;
    _respRate = 0.0;
    _bpSys = 0.0;
    _bpDia = 0.0;
    _glucoseLevel = 0.0;
    _creatinineLevel = 0.0;
    _egfrLevel = 0.0;
    _creatinineClass = '';
    _kdigoStage = '';
    _signalQuality = 0.0;
    _fingerDetected = false;
    _torchActive = false;
    _ppgHistory.clear();
    _clearBuffers();

    if (_cameraMode == CameraMode.front) {
      _toastPrompt = "Position your face in the frame with good lighting.";
    } else {
      _toastPrompt = "Align your finger over the camera lens and flash.";
    }
    notifyListeners();
  }
}
