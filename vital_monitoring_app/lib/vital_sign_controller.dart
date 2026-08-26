import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'zero_allocation_extractor.dart';
import 'ppg_signal_engine.dart';
import 'dart:math';

enum TrackingState { baseline, profiling, complete, rejected }

class VitalSignController extends ChangeNotifier {
  TrackingState _currentState = TrackingState.baseline;
  double _gatheringProgress = 0.0;
  String _toastPrompt = "Align your finger over the camera lens and flash.";
  double _heartRate = 0.0;

  TrackingState get currentState => _currentState;
  double get gatheringProgress => _gatheringProgress;
  String get toastPrompt => _toastPrompt;
  double get heartRate => _heartRate;

  // Buffer for processing (used for perfusion index calculation)
  final List<double> _liveGreenBuffer = [];
  
  // Buffer for waveform display (shows more recent points)
  final List<double> _waveformBuffer = [];
  static const int _waveformDisplaySize = 300; // Show ~10 seconds at 30fps

  void attachStreamPipeline(CameraController controller) {
    controller.startImageStream((CameraImage streamedFrame) {
      if (_currentState == TrackingState.complete) return;

      // Extract raw metrics using the high-performance processor
      final Float64List extractedValues = ZeroAllocationExtractor.processYUV420Frame(streamedFrame);
      final double currentGreenIntensity = extractedValues[1];
      
      // Add to processing buffer
      _liveGreenBuffer.add(currentGreenIntensity);
      
      // Add to waveform display buffer
      _waveformBuffer.add(currentGreenIntensity);
      // Keep display buffer at fixed size
      if (_waveformBuffer.length > _waveformDisplaySize) {
        _waveformBuffer.removeAt(0);
      }

      // Evaluate signal safety profiles in 4-second sliding data windows
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
}