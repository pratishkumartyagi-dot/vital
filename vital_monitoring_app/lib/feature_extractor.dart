import 'dart:math' as math;

/// Extracts morphological PPG features for ML inference.
/// Pulse cycle: [PeakTime, DecayTime, Width, InflectionDiff]
class FeatureExtractor {
  static Map<String, double> extract(List<double> waveform) {
    if (waveform.length < 10) {
      return {
        'pulseWidth': 1.0,
        'peakTime': 0.5,
        'decayTime': 0.5,
        'inflectionDiff': 0.0,
      };
    }

    // Find global max/min for normalization
    double minVal = waveform.reduce(math.min);
    double maxVal = waveform.reduce(math.max);
    double range = maxVal - minVal;
    if (range < 0.001) range = 1.0;

    List<double> norm = waveform.map((v) => (v - minVal) / range).toList();

    // Find first peak index
    int peakIdx = 0;
    for (int i = 1; i < norm.length - 1; i++) {
      if (norm[i] > norm[i - 1] && norm[i] > norm[i + 1] && norm[i] > norm[peakIdx]) {
        peakIdx = i;
      }
    }

    double totalLength = norm.length.toDouble();
    double pulseWidth = totalLength / 30.0; // approx seconds at 30fps
    double peakTime = peakIdx / totalLength;
    double decayTime = (norm.length - peakIdx) / totalLength;
    double inflectionDiff = (norm[peakIdx] - norm[0]).abs();

    return {
      'pulseWidth': pulseWidth,
      'peakTime': peakTime,
      'decayTime': decayTime,
      'inflectionDiff': inflectionDiff,
    };
  }
}
