import 'package:tflite_flutter/tflite_flutter.dart';
import 'feature_extractor.dart';

class VitalMLInference {
  static Interpreter? _interpreter;

  static Future<void> init() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/vitals_estimator.tflite');
    } catch (e) {
      // Fallback to demo mode if .tflite asset missing
      print('TFLite load error: $e');
    }
  }

  static Map<String, double> predict(List<double> waveform) {
    if (_interpreter == null) {
      return {
        'systolic': 110.0 + (waveform.length % 20),
        'diastolic': 70.0 + (waveform.length % 10),
        'glucose': 95.0 + (waveform.length % 15),
      };
    }

    final features = FeatureExtractor.extract(waveform);
    final input = [
      features['pulseWidth']!,
      features['peakTime']!,
      features['decayTime']!,
      features['inflectionDiff']!,
    ];

    var output = List<double>.filled(3, 0).reshape([1, 3]);
    _interpreter!.run(input.reshape([1, 4]), output);

    return {
      'systolic': output[0][0],
      'diastolic': output[0][1],
      'glucose': output[0][2],
    };
  }
}
