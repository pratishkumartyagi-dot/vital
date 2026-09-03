import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'zero_allocation_extractor.dart';
import 'ppg_signal_engine.dart';
import 'native_hardware_bridge.dart';
import 'dart:math' as math;
import 'dart:async';

enum TrackingState { baseline, profiling, complete, rejected }

class VitalSignController extends ChangeNotifier {
  TrackingState _currentState = TrackingState.baseline;
  double _gatheringProgress = 0.0;
  String _toastPrompt = "Align your finger over the camera lens and flash.";
  double _heartRate = 0.0;
  double _hbLevel = 0.0;
  bool _isDemoMode = false;
  Timer? _demoTimer;
  int _demoFrameCount = 0;

  TrackingState get currentState => _currentState;
  double get gatheringProgress => _gatheringProgress;
  String get toastPrompt => _toastPrompt;
  double get heartRate => _heartRate;
  double get hbLevel => _hbLevel;
  bool get isDemoMode => _isDemoMode;

  // Separate buffers for multi-wavelength analysis
  final List<double> _liveRedBuffer = [];
  final List<double> _liveGreenBuffer = [];
  final List<double> _liveBlueBuffer = [];
  
  // Buffer for waveform display
  final List<double> _waveformBuffer = [];
  static const int _waveformDisplaySize = 300;

  VitalSignController() {
    // Start in demo mode by default for research/testing
    _enableDemoMode();
  }

  void _enableDemoMode() {
    _isDemoMode = true;
    _toastPrompt = "DEMO MODE: Simulated PPG signal. Place finger over camera to switch to live mode.";
    _startDemoTimer();
  }

  void _disableDemoMode() {
    _isDemoMode = false;
    _toastPrompt = "Align your finger over the camera lens and flash.";
    _stopDemoTimer();
    _liveGreenBuffer.clear();
    _waveformBuffer.clear();
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
        // Generate simulated PPG signal with heartbeat-like pattern
        final double time = _demoFrameCount * 0.033; // 33ms per frame
        // Base signal with noise
        final double baseSignal = 100 + 
            20 * math.sin(2 * math.pi * 1.2 * time) + // ~72 BPM (1.2 Hz)
            5 * math.sin(2 * math.pi * 2.5 * time) +   // Harmonics
            (math.Random().nextDouble() - 0.5) * 2; // Noise
        
        // Add occasional motion artifacts
        if (_demoFrameCount % 200 == 0) {
          _liveGreenBuffer.add(baseSignal + (math.Random().nextDouble() - 0.5) * 30);
        } else {
          _liveGreenBuffer.add(baseSignal);
        }
        
        // Add to waveform display buffer
        _waveformBuffer.add(baseSignal);
        // Keep display buffer at fixed size
        if (_waveformBuffer.length > _waveformDisplaySize) {
          _waveformBuffer.removeAt(0);
        }

        // Process for state changes (same as live mode)
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
            _gatheringProgress += 0.02; // Increment progress tracking
            
            if (_gatheringProgress >= 1.0) {
              _gatheringProgress = 1.0;
              _currentState = TrackingState.complete;
              _toastPrompt = "Measurement complete.";
              // Calculate heart rate when measurement completes
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
    if (_isDemoMode) return; // Don't attach live stream if in demo mode
    
    controller.startImageStream((CameraImage streamedFrame) {
      if (_currentState == TrackingState.complete) return;

      // Extract raw RGB metrics
      final Float64List extractedValues = ZeroAllocationExtractor.processYUV420Frame(streamedFrame);
      final double currentRed = extractedValues[0];
      final double currentGreen = extractedValues[1];
      final double currentBlue = extractedValues[2];

      // Software guard: reject saturated or dark frames before they reach PPG math
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
      
      // Add green to waveform display
      _waveformBuffer.add(currentGreen);
      if (_waveformBuffer.length > _waveformDisplaySize) {
        _waveformBuffer.removeAt(0);
      }

      // Send green channel to native PPG DSP for more accurate heart rate estimation
      NativeHardwareBridge.processPpgSample(currentGreen).then((nativeBpm) {
        if (nativeBpm != null && nativeBpm > 0) {
          _heartRate = nativeBpm;
          notifyListeners();
        }
      });

      // Safety gate: Perfusion Index check (low = flatline / too much pressure / cold)
      if (_liveGreenBuffer.length >= 30) {
        final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
        if (pi < 0.2) {
          _currentState = TrackingState.rejected;
          _toastPrompt = "Low pulse signal detected. Warm hands and reduce finger pressure.";
          _gatheringProgress = 0.0;
          _liveRedBuffer.clear();
          _liveGreenBuffer.clear();
          _liveBlueBuffer.clear();
          _waveformBuffer.clear();
          NativeHardwareBridge.resetPpgBuffer();
          notifyListeners();
          return;
        }
      }

      // Multi-wavelength evaluation in 4-second windows
      if (_liveGreenBuffer.length >= 120) {
        // SpO2 via Red/Blue ratio
        final double redPI = PPGSignalEngine.computePerfusionIndex(_liveRedBuffer);
        final double bluePI = PPGSignalEngine.computePerfusionIndex(_liveBlueBuffer);
        double calculatedSpO2 = 110.0 - (25.0 * (redPI / bluePI));
        calculatedSpO2 = calculatedSpO2.clamp(70.0, 100.0);

        // Haemoglobin via Red/Green ratio (18.2 - 6.8 * rHb)
        final double greenPI = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
        double calculatedHb = 18.2 - (6.8 * (redPI / greenPI));
        calculatedHb = calculatedHb.clamp(5.0, 20.0);
        _hbLevel = calculatedHb;
        
        // Ambient light check: if DC base values too high, warn
        final double avgGreenDC = _liveGreenBuffer.reduce((a,b) => a+b) / _liveGreenBuffer.length;
        if (avgGreenDC > 240) {
           _toastPrompt = "Ambient lighting too bright. Shade the lens with your finger.";
        }
        
        // Update state and progress
        if (redPI > 0.5 && greenPI > 0.5 && bluePI > 0.5) {
          _currentState = TrackingState.profiling;
          _toastPrompt = "Analyzing stability. Keep finger steady... SpO₂: ${calculatedSpO2.toStringAsFixed(0)}% | Hb: ${calculatedHb.toStringAsFixed(1)} g/dL";
          _gatheringProgress += 0.02;
          
          if (_gatheringProgress >= 1.0) {
            _gatheringProgress = 1.0;
            _currentState = TrackingState.complete;
            _toastPrompt = "Measurement complete. SpO₂: ${calculatedSpO2.toStringAsFixed(0)}% | Hb: ${calculatedHb.toStringAsFixed(1)} g/dL";
            _calculateHeartRate();
          }
          notifyListeners();
        } else {
          _currentState = TrackingState.rejected;
          _toastPrompt = "Unreliable Signal. Please steady your finger and apply lighter pressure.";
          _gatheringProgress = 0.0;
          _liveRedBuffer.clear();
          _liveGreenBuffer.clear();
          _liveBlueBuffer.clear();
          _waveformBuffer.clear();
          NativeHardwareBridge.resetPpgBuffer();
          notifyListeners();
        }
      }
    });
  }
  
  void _calculateHeartRate() {
    if (_liveGreenBuffer.length < 120) return;
    
    // Apply bandpass filter to get cleaner signal for heart rate calculation
    final filtered = PPGSignalEngine.applyBandpassFilter(_liveGreenBuffer);
    
    // Find peaks in the filtered signal
    final List<int> peaks = _findPeaks(filtered, minDistance: 10, threshold: 0.5);
    
    if (peaks.length >= 2) {
      // Calculate average interval between peaks
      final List<double> intervals = [];
      for (int i = 1; i < peaks.length; i++) {
        intervals.add((peaks[i] - peaks[i-1]) / 30.0); // Convert frames to seconds (assuming 30fps)
      }
      
      final double avgInterval = intervals.reduce((a, b) => a + b) / intervals.length;
      if (avgInterval > 0) {
        _heartRate = 60 / avgInterval; // Convert to BPM
        // Constrain to reasonable heart rate range
        _heartRate = _heartRate.clamp(40, 180);
      }
    }
  }
  
  List<int> _findPeaks(List<double> signal, {int minDistance = 1, double threshold = 0.0}) {
    final List<int> peaks = <int>[];
    if (signal.length < 3) return peaks;
    
    for (int i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && 
          signal[i] > signal[i + 1] && 
          signal[i] > threshold) {
        // Check if this peak is far enough from the last one
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
    
    // Find min and max for normalization
    double minVal = _waveformBuffer.reduce(math.min);
    double maxVal = _waveformBuffer.reduce(math.max);
    
    // Avoid division by zero
    if (maxVal - minVal < 0.001) {
      return List.filled(_waveformBuffer.length, 0.5);
    }
    
    // Normalize to 0-1 range
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
    _toastPrompt = "Align your finger over the camera lens and flash.";
    _heartRate = 0.0;
    _hbLevel = 0.0;
    _liveRedBuffer.clear();
    _liveGreenBuffer.clear();
    _liveBlueBuffer.clear();
    _waveformBuffer.clear();
    NativeHardwareBridge.resetPpgBuffer();
    _enableDemoMode();
    notifyListeners();
  }
}