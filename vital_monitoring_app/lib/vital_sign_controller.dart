import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'zero_allocation_extractor.dart';
import 'ppg_signal_engine.dart';

enum TrackingState { baseline, profiling, complete, rejected }

class VitalSignController extends ChangeNotifier {
  TrackingState _currentState = TrackingState.baseline;
  double _gatheringProgress = 0.0;
  String _toastPrompt = "Align your finger over the camera lens and flash.";

  TrackingState get currentState => _currentState;
  double get gatheringProgress => _gatheringProgress;
  String get toastPrompt => _toastPrompt;

  final List<double> _liveGreenBuffer = [];

  void attachStreamPipeline(CameraController controller) {
    controller.startImageStream((CameraImage streamedFrame) {
      if (_currentState == TrackingState.complete) return;

      // Extract raw metrics using the high-performance processor
      final Float64List extractedValues = ZeroAllocationExtractor.processYUV420Frame(streamedFrame);
      final double currentGreenIntensity = extractedValues[1];
      
      _liveGreenBuffer.add(currentGreenIntensity);

      // Evaluate signal safety profiles in 4-second sliding data windows
      if (_liveGreenBuffer.length >= 120) {
        final double pi = PPGSignalEngine.computePerfusionIndex(_liveGreenBuffer);
        
        if (pi < 0.5) {
          _currentState = TrackingState.rejected;
          _toastPrompt = "Unreliable Signal. Please steady your finger and apply lighter pressure.";
          _gatheringProgress = 0.0;
          _liveGreenBuffer.clear();
        } else {
          _currentState = TrackingState.profiling;
          _toastPrompt = "Analyzing heart rate stability. Keep perfectly still...";
          _gatheringProgress += 0.02; // Increment progress tracking
          
          if (_gatheringProgress >= 1.0) {
            _gatheringProgress = 1.0;
            _currentState = TrackingState.complete;
            _toastPrompt = "Measurement complete.";
          }
        }
        notifyListeners();
      }
    });
  }
}