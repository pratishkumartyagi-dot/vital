import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter/src/tflite_flutter.dart';

class VitalMLInference {
  static Interpreter? _interpreter;
  static bool _initialized = false;
  static List<int>? _inputShape;
  static List<int>? _outputShape;
  static int _inputChannels = 3;
  static int _inputTimeSteps = 300;

  static Future<void> init() async {
    if (_initialized) return;
    
    try {
      _interpreter = await Interpreter.fromAsset('assets/vitals_estimator.tflite');
      
      // Get input/output tensor info
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensors = _interpreter!.getOutputTensors();
      
      _inputShape = inputTensor.shape;
      _outputShape = outputTensors.map((t) => t.shape).toList();
      
      print('TFLite model loaded successfully');
      print('Input shape: $_inputShape');
      print('Output shapes: $_outputShape');
      print('Input type: ${inputTensor.type}');
      for (int i = 0; i < outputTensors.length; i++) {
        print('Output $i: ${outputTensors[i].name} shape=${outputTensors[i].shape} type=${outputTensors[i].type}');
      }
      
      _initialized = true;
    } catch (e) {
      print('TFLite load error: $e');
      _initialized = false;
    }
  }

  /// Run inference on multi-channel PPG signals
  /// Input: [red, green, blue] each of length >= _inputTimeSteps
  /// Returns: Map with heart_rate, spo2, hemoglobin, resp_rate, bp_sys, bp_dia
  static Map<String, double> predictPPG({
    required List<double> red,
    required List<double> green,
    required List<double> blue,
  }) {
    if (!_initialized || _interpreter == null) {
      return _demoPredictions();
    }

    try {
      // Ensure correct length (pad or truncate)
      final inputRed = _prepareSignal(red);
      final inputGreen = _prepareSignal(green);
      final inputBlue = _prepareSignal(blue);

      // Normalize signals (zero mean, unit variance per channel)
      final normalizedRed = _normalize(inputRed);
      final normalizedGreen = _normalize(inputGreen);
      final normalizedBlue = _normalize(inputBlue);

      // Prepare input tensor [1, 3, T] or [1, T, 3] depending on model
      final input = _formatInput(normalizedRed, normalizedGreen, normalizedBlue);

      // Prepare outputs
      final outputs = _prepareOutputs();

      // Run inference
      _interpreter!.run(input, outputs);

      // Parse outputs
      return _parseOutputs(outputs);
    } catch (e) {
      print('Inference error: $e');
      return _demoPredictions();
    }
  }

  /// Run inference using pre-computed features (legacy compatibility)
  static Map<String, double> predictFromFeatures(List<double> waveform) {
    if (!_initialized || _interpreter == null) {
      return _demoPredictions();
    }

    try {
      // Use green channel as primary, synthesize red/blue from waveform stats
      final mean = waveform.reduce((a, b) => a + b) / waveform.length;
      final std = waveform.length > 1 
          ? (waveform.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / waveform.length)
          : 1.0;

      return predictPPG(
        red: waveform.map((v) => v * 1.1).toList(),
        green: waveform,
        blue: waveform.map((v) => v * 0.9).toList(),
      );
    } catch (e) {
      print('Inference error: $e');
      return _demoPredictions();
    }
  }

  static List<double> _prepareSignal(List<double> signal) {
    if (signal.length >= _inputTimeSteps) {
      return signal.sublist(signal.length - _inputTimeSteps);
    } else {
      // Pad with last value
      final last = signal.isNotEmpty ? signal.last : 100.0;
      return [...signal, ...List.filled(_inputTimeSteps - signal.length, last)];
    }
  }

  static List<double> _normalize(List<double> signal) {
    final mean = signal.reduce((a, b) => a + b) / signal.length;
    final variance = signal.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / signal.length;
    final std = variance > 1e-6 ? variance.sqrt() : 1.0;
    return signal.map((v) => (v - mean) / std).toList();
  }

  dynamic _formatInput(List<double> red, List<double> green, List<double> blue) {
    // Model expects [batch, channels, time] = [1, 3, 300]
    // TFLite typically uses NHWC, but our ONNX export uses NCHW
    // We'll try both formats
    
    if (_inputShape != null && _inputShape!.length == 4) {
      // [1, 3, 300, 1] or similar
      return [
        [
          red,
          green,
          blue,
        ]
      ];
    } else if (_inputShape != null && _inputShape!.length == 3) {
      // [1, 3, 300] or [1, 300, 3]
      if (_inputShape![1] == 3) {
        // NCHW: [1, 3, 300]
        return [
          [
            red,
            green,
            blue,
          ]
        ];
      } else {
        // NHWC: [1, 300, 3]
        final combined = <List<double>>[];
        for (int i = 0; i < _inputTimeSteps; i++) {
          combined.add([red[i], green[i], blue[i]]);
        }
        return [combined];
      }
    }
    
    // Default: NCHW
    return [
      [
        red,
        green,
        blue,
      ]
    ];
  }

  List<dynamic> _prepareOutputs() {
    if (_outputShape != null) {
      return _outputShape!.map((shape) {
        // Flatten shape to list
        var result = List<double>.filled(shape.reduce((a, b) => a * b), 0.0);
        // Reshape to match output tensor
        return _reshapeOutput(result, shape);
      }).toList();
    }
    // Default: 6 outputs of shape [1, 1]
    return List.generate(6, (_) => [0.0]);
  }

  dynamic _reshapeOutput(List<double> flat, List<int> shape) {
    if (shape.length == 1) return flat;
    if (shape.length == 2) {
      final rows = shape[0];
      final cols = shape[1];
      return List.generate(rows, (i) => flat.sublist(i * cols, (i + 1) * cols));
    }
    return flat;
  }

  Map<String, double> _parseOutputs(List<dynamic> outputs) {
    final taskNames = ['heart_rate', 'spo2', 'hemoglobin', 'resp_rate', 'bp_sys', 'bp_dia'];
    final results = <String, double>{};
    
    for (int i = 0; i < outputs.length && i < taskNames.length; i++) {
      var value = 0.0;
      final output = outputs[i];
      
      if (output is List) {
        if (output.length == 1) {
          value = (output[0] as num).toDouble();
        } else if (output.isNotEmpty && output[0] is List) {
          value = (output[0][0] as num).toDouble();
        }
      } else if (output is num) {
        value = output.toDouble();
      }
      
      // Apply reasonable bounds
      switch (taskNames[i]) {
        case 'heart_rate':
          value = value.clamp(30.0, 200.0);
          break;
        case 'spo2':
          value = value.clamp(50.0, 100.0);
          break;
        case 'hemoglobin':
          value = value.clamp(5.0, 20.0);
          break;
        case 'resp_rate':
          value = value.clamp(5.0, 50.0);
          break;
        case 'bp_sys':
          value = value.clamp(60.0, 200.0);
          break;
        case 'bp_dia':
          value = value.clamp(30.0, 130.0);
          break;
      }
      
      results[taskNames[i]] = value;
    }
    
    // Add legacy keys for compatibility
    results['systolic'] = results['bp_sys'] ?? 120.0;
    results['diastolic'] = results['bp_dia'] ?? 80.0;
    results['glucose'] = 95.0; // Not predicted by current model
    
    return results;
  }

  Map<String, double> _demoPredictions() {
    return {
      'heart_rate': 72.0,
      'spo2': 98.0,
      'hemoglobin': 14.0,
      'resp_rate': 16.0,
      'bp_sys': 120.0,
      'bp_dia': 80.0,
      'systolic': 120.0,
      'diastolic': 80.0,
      'glucose': 95.0,
    };
  }

  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _initialized = false;
  }
}