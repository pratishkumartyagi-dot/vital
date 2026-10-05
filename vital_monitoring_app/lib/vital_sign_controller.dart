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
  bool get isDemoMode => _isDemoMode;
  CameraMode get cameraMode => _cameraMode;

  /// Signal quality (0-100%) computed from perfusion index of the live
  /// green buffer using the corrected zero-phase bandpass filter.
  ///
  /// Uses the same square-root perfusion law as `computeSignalQuality` in
  /// `src/app/vital-monitoring/ppg-analysis.ts` (full credit at 0.5% AC/DC).
  /// The previous linear `pi / 1.5` mapping badly under-reported usable
  /// signals: a healthy fingertip PPG at ~0.5% AC/DC scored 33% and a facial
  /// rPPG trace below 1% scored proportionally worse, which read as "very low
  /// signal strength" even while the heart rate was being read correctly.
  double get signalQuality {
    if (_liveGreenBuffer.length < _minSamplesForHR) return 0.0;
    final pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
    // sqrt law: 0.05% -> 32%, 0.2% -> 63%, 0.5% and above -> 100%.
    return (math.sqrt((pi / 0.5).clamp(0.0, 1.0))) * 100.0;
  }

  // Separate buffers for multi-wavelength analysis
  final List<double> _liveRedBuffer = [];
  final List<double> _liveGreenBuffer = [];
  final List<double> _liveBlueBuffer = [];
  
  // Buffer for waveform display
  final List<double> _waveformBuffer = [];
  static const int _waveformDisplaySize = 300;
  static const int _analysisWindowSize = 300; // 10 seconds at 30fps
  static const int _minSamplesForHR = 90; // 3 seconds

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
    _demoTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) { // ~30fps
      _demoFrameCount++;
      if (_isDemoMode) {
        final double time = _demoFrameCount * 0.033;
        final double baseSignal = 100 + 
            20 * math.sin(2 * math.pi * 1.2 * time) + // ~72 BPM (1.2 Hz)
            5 * math.sin(2 * math.pi * 2.5 * time) +   // Harmonics
            (math.Random().nextDouble() - 0.5) * 2; // Noise
        
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
      if (_currentState == TrackingState.complete) return;

      // Extract raw RGB metrics
      final Float64List extractedValues = ZeroAllocationExtractor.processYUV420Frame(streamedFrame);
      final double currentRed = extractedValues[0];
      final double currentGreen = extractedValues[1];
      final double currentBlue = extractedValues[2];

      // Software guard: reject saturated or dark frames
      if (currentRed > 250.0 || currentGreen > 250.0 || currentBlue > 250.0) {
        _toastPrompt = "Signal over-amplified. Reduce finger pressure on the lens.";
        return;
      }
      if (currentRed < 15.0 && currentGreen < 15.0 && currentBlue < 15.0) {
        _toastPrompt = "No signal detected. Cover the lens and flash completely.";
        return;
      }
      
      // Append to multi-wavelength buffers
      _liveRedBuffer.add(currentRed);
      _liveGreenBuffer.add(currentGreen);
      _liveBlueBuffer.add(currentBlue);
      
      // Add to waveform display (use green for rear, combined for front)
      _waveformBuffer.add(_cameraMode == CameraMode.rear ? currentGreen : 
          (currentRed + currentGreen + currentBlue) / 3);
      if (_waveformBuffer.length > _waveformDisplaySize) {
        _waveformBuffer.removeAt(0);
      }

      // Maintain analysis window size
      if (_liveRedBuffer.length > _analysisWindowSize) {
        _liveRedBuffer.removeAt(0);
        _liveGreenBuffer.removeAt(0);
        _liveBlueBuffer.removeAt(0);
      }

      // Send green channel to native PPG DSP for heart rate
      NativeHardwareBridge.processPpgSample(currentGreen).then((nativeBpm) {
        if (nativeBpm != null && nativeBpm > 0) {
          _heartRate = nativeBpm;
          notifyListeners();
        }
      });

      // Process based on camera mode
      if (_cameraMode == CameraMode.rear) {
        _processRearCameraPPG();
      } else {
        _processFrontCameraRPPG();
      }
    });
  }

  void _processRearCameraPPG() {
    // Safety gate: Perfusion Index check
    if (_liveGreenBuffer.length >= 30) {
      final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      if (pi < 0.2) {
        _currentState = TrackingState.rejected;
        _toastPrompt = "Low pulse signal detected. Warm hands and reduce finger pressure.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
        return;
      }
    }

    // Multi-wavelength evaluation in 4-second windows (120 frames at 30fps)
    if (_liveGreenBuffer.length >= 120) {
      // SpO2 via Red/Blue ratio (using blue as IR proxy - not clinical grade)
      final double redPI = PPGSignalEngine.computePerfusionIndex(_liveRedBuffer);
      final double bluePI = PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer);
      final double greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      
      double calculatedSpO2 = 110.0 - (25.0 * (redPI / bluePI));
      calculatedSpO2 = calculatedSpO2.clamp(70.0, 100.0);
      _spo2Level = calculatedSpO2;

      // Haemoglobin via Red/Green ratio
      double calculatedHb = 18.2 - (6.8 * (redPI / greenPI));
      calculatedHb = calculatedHb.clamp(5.0, 20.0);
      _hbLevel = calculatedHb;

      // Ambient light check
      final double avgGreenDC = _liveGreenBuffer.reduce((a,b) => a+b) / _liveGreenBuffer.length;
      if (avgGreenDC > 240) {
        _toastPrompt = "Ambient lighting too bright. Shade the lens with your finger.";
      }
      
      // Respiratory rate estimation from baseline wander
      _respRate = _estimateRespiratoryRate(_liveGreenBuffer);

      // Update state and progress
      if (redPI > 0.5 && greenPI > 0.5 && bluePI > 0.5) {
        _currentState = TrackingState.profiling;
        _toastPrompt = "Analyzing stability... HR: ${_heartRate.toStringAsFixed(0)} BPM | SpO₂: ${calculatedSpO2.toStringAsFixed(0)}% | Hb: ${calculatedHb.toStringAsFixed(1)} g/dL";
        _gatheringProgress += 0.02;
        
        if (_gatheringProgress >= 1.0) {
          _gatheringProgress = 1.0;
          _currentState = TrackingState.complete;
          _toastPrompt = "Measurement complete. HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${calculatedSpO2.toStringAsFixed(0)}% | Hb: ${calculatedHb.toStringAsFixed(1)}";
          _calculateHeartRate();
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

  void _processFrontCameraRPPG() {
    // Front camera rPPG uses different thresholds and processing
    // Requires face detection - simplified here
    
    if (_liveGreenBuffer.length >= 30) {
      // For rPPG, signal amplitude is much smaller
      final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      if (pi < 0.05) { // Lower threshold for rPPG
        _currentState = TrackingState.rejected;
        _toastPrompt = "Weak face signal. Ensure good lighting and face visibility.";
        _gatheringProgress = 0.0;
        _clearBuffers();
        notifyListeners();
        return;
      }
    }

    if (_liveGreenBuffer.length >= 120) {
      // For rPPG, use green channel primarily (best SNR for face)
      // Can also use POS algorithm (Plane-Orthogonal-to-Skin) for better results
      final double greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
      final double redPI = PPGSignalEngine.computePerfusionIndex(_liveRedBuffer);
      final double bluePI = PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer);
      
      // rPPG SpO2 estimation (very approximate without IR)
      double calculatedSpO2 = 110.0 - (25.0 * (redPI / bluePI));
      calculatedSpO2 = calculatedSpO2.clamp(70.0, 100.0);
      _spo2Level = calculatedSpO2;

      // Hb estimation
      double calculatedHb = 18.2 - (6.8 * (redPI / greenPI));
      calculatedHb = calculatedHb.clamp(5.0, 20.0);
      _hbLevel = calculatedHb;

      // Respiratory rate from green channel baseline
      _respRate = _estimateRespiratoryRate(_liveGreenBuffer);

      if (greenPI > 0.1) { // Lower threshold for rPPG
        _currentState = TrackingState.profiling;
        _toastPrompt = "rPPG analysis... HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${calculatedSpO2.toStringAsFixed(0)}%";
        _gatheringProgress += 0.015; // Slower progress for rPPG
        
        if (_gatheringProgress >= 1.0) {
          _gatheringProgress = 1.0;
          _currentState = TrackingState.complete;
          _toastPrompt = "rPPG complete. HR: ${_heartRate.toStringAsFixed(0)} | SpO₂: ${calculatedSpO2.toStringAsFixed(0)}%";
          _calculateHeartRate();
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

  double _estimateRespiratoryRate(List<double> signal) {
    // Two-stage respiratory rate estimation:
    // Stage 1: Cardiac bandpass (0.5-3.5 Hz) to isolate the pulse, then
    //   extract the amplitude modulation (AM envelope) via peak-to-trough
    //   tracking. Cardiac information is removed from the respiratory path.
    // Stage 2: Respiratory bandpass (0.06-0.7 Hz) on the AM envelope and
    //   raw baseline wander to isolate breathing, then peak/track fusion.
    // The previous single-stage exponential smoothing (alpha=0.15, ~0.72 Hz
    // cutoff) did not remove the cardiac component, causing aliasing to
    // ~24 BPM on a 72 BPM heart signal.
    if (signal.length < 180) return 15.0; // Need at least 6s at 30fps

    const double fs = 30.0;

    // Stage 1: Extract cardiac pulse via bandpass, then derive AM envelope
    final cardiac = PPGSignalEngine.applyCardiacFilter(signal);
    final envelope = _amplitudeEnvelope(cardiac);

    // Stage 2: Respiratory bandpass on envelope and raw baseline wander
    final envelopeResp = PPGSignalEngine.applyRespiratoryFilter(envelope, fs);
    final (double envFreq, double envPower) = _findDominantFreq(envelopeResp, fs, 0.1, 0.5);
    final envPeaks = _findRespPeaks(envelopeResp);
    final envRate = envPeaks.isNotEmpty ? _peaksToRate(envPeaks, fs) : 0.0;

    // Also extract from baseline wander (detrended raw signal)
    final mean = signal.reduce((a, b) => a + b) / signal.length;
    final detrended = signal.map((v) => v - mean).toList();
    final baseline = PPGSignalEngine.applyRespiratoryFilter(detrended, fs);
    final (double baseFreq, double basePower) = _findDominantFreq(baseline, fs, 0.1, 0.5);
    final basePeaks = _findRespPeaks(baseline);
    final baseRate = basePeaks.isNotEmpty ? _peaksToRate(basePeaks, fs) : 0.0;

    // Peak/track fusion
    final candidates = <double>[];
    if (envRate > 0) candidates.add(envRate);
    if (baseRate > 0) candidates.add(baseRate);
    if (envFreq > 0) candidates.add(envFreq * 60);
    if (baseFreq > 0) candidates.add(baseFreq * 60);

    double respBpm = 0;
    if (candidates.isNotEmpty) {
      // Use median of valid candidates to reject outliers
      final sorted = List<double>.from(candidates)..sort();
      final medianRate = sorted[sorted.length ~/ 2];
      final valid = sorted.where((r) => (r - medianRate).abs() < 6).toList();
      respBpm = valid.isNotEmpty
          ? valid.reduce((a, b) => a + b) / valid.length
          : medianRate;
    }

    if (respBpm < 5 || respBpm > 50) return 15.0;
    return respBpm.clamp(8.0, 35.0);
  }

  /// Extract amplitude envelope from a filtered cardiac signal.
  /// Computes the instantaneous amplitude as half the peak-to-trough range
  /// in overlapping windows.
  List<double> _amplitudeEnvelope(List<double> signal) {
    if (signal.isEmpty) return [];
    const int winSize = 15; // ~0.5s at 30fps
    final result = List<double>.filled(signal.length, 0.0);
    for (int i = 0; i < signal.length; i++) {
      final start = (i - winSize).clamp(0, signal.length - 1);
      final end = (i + winSize + 1).clamp(0, signal.length);
      double minVal = signal[start];
      double maxVal = signal[start];
      for (int j = start; j < end; j++) {
        if (signal[j] < minVal) minVal = signal[j];
        if (signal[j] > maxVal) maxVal = signal[j];
      }
      result[i] = (maxVal - minVal) / 2.0;
    }
    return result;
  }

  /// Find peaks in a respiratory signal (baseline wander).
  List<int> _findRespPeaks(List<double> signal) {
    return _findPeaks(signal, minDistance: 45, threshold: 0.1);
  }

  /// Convert peak indices to a rate in BPM.
  double _peaksToRate(List<int> peaks, double fs) {
    if (peaks.length < 2) return 0.0;
    final intervals = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      final interval = (peaks[i] - peaks[i - 1]) / fs; // seconds
      if (interval >= 1.2 && interval <= 6.0) intervals.add(interval); // 10-50 BPM
    }
    if (intervals.isEmpty) return 0.0;
    final avgInterval = intervals.reduce((a, b) => a + b) / intervals.length;
    if (avgInterval <= 0) return 0.0;
    return 60.0 / avgInterval;
  }

  /// Find dominant frequency and its power in a given band using DFT.
  (double, double) _findDominantFreq(
      List<double> signal, double fs, double fLow, double fHigh) {
    if (signal.length < 4) return (0.0, 0.0);

    final n = signal.length;
    final mean = signal.reduce((a, b) => a + b) / n;
    final centered = signal.map((v) => v - mean).toList();

    double bestFreq = 0;
    double bestPower = 0;
    final freqRes = fs / n;
    final kMin = (fLow / freqRes).ceil();
    final kMax = (fHigh / freqRes).floor();

    for (int k = kMin; k <= kMax && k < n ~/ 2; k++) {
      double real = 0;
      double imag = 0;
      final angle = 2 * math.pi * k / n;
      for (int i = 0; i < n; i++) {
        final phi = angle * i;
        real += centered[i] * math.cos(phi);
        imag -= centered[i] * math.sin(phi);
      }
      final power = real * real + imag * imag;
      if (power > bestPower) {
        bestPower = power;
        bestFreq = k * freqRes;
      }
    }
    return (bestFreq, bestPower / (n * n));
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
    
    // Apply bandpass filter to get cleaner signal
    final filtered = PPGSignalEngine.applyBandpassFilter(_liveGreenBuffer);
    
    // Find peaks in the filtered signal
    final List<int> peaks = _findPeaks(filtered, minDistance: 8, threshold: 0.3);
    
    if (peaks.length >= 3) {
      final List<double> intervals = [];
      for (int i = 1; i < peaks.length; i++) {
        intervals.add((peaks[i] - peaks[i-1]) / 30.0);
      }
      
      // Remove outliers (intervals outside 0.33-1.5s = 40-180 BPM)
      final validIntervals = intervals.where((i) => i >= 0.33 && i <= 1.5).toList();
      if (validIntervals.length >= 2) {
        final double avgInterval = validIntervals.reduce((a, b) => a + b) / validIntervals.length;
        if (avgInterval > 0) {
          _heartRate = (60 / avgInterval).clamp(40, 180);
        }
      }
    }
  }
  
  List<int> _findPeaks(List<double> signal, {int minDistance = 1, double threshold = 0.0}) {
    final List<int> peaks = <int>[];
    if (signal.length < 3) return peaks;
    
    // Dynamic threshold based on signal statistics
    final mean = signal.reduce((a, b) => a + b) / signal.length;
    final stdDev = math.sqrt(signal.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / signal.length);
    final dynamicThreshold = threshold * stdDev;
    
    for (int i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && 
          signal[i] > signal[i + 1] && 
          signal[i] > dynamicThreshold) {
        if (peaks.isEmpty || i - peaks.last >= minDistance) {
          peaks.add(i);
        }
      }
    }
    return peaks;
  }
  
  // Get normalized waveform data for display (0-1 range)
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

  @override
  void dispose() {
    _stopDemoTimer();
    super.dispose();
  }

  void reset() {
    _currentState = TrackingState.baseline;
    _gatheringProgress = 0.0;
    _heartRate = 0.0;
    _hbLevel = 0.0;
    _spo2Level = 0.0;
    _respRate = 0.0;
    _clearBuffers();
    
    if (_cameraMode == CameraMode.front) {
      _toastPrompt = "Position your face in the frame with good lighting.";
    } else {
      _toastPrompt = "Align your finger over the camera lens and flash.";
    }
    notifyListeners();
  }
}