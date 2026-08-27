import 'dart:math' as math;

class PPGSignalEngine {
  /// 4th-Order Butterworth Bandpass Filter (0.5Hz - 3.5Hz at Fs=30Hz)
  static List<double> applyBandpassFilter(List<double> input) {
    final int length = input.length;
    final List<double> filtered = List<double>.filled(length, 0.0);
    
    // Direct Form II Transposed IIR Array Coefficients
    const List<double> b = [0.0039, 0.0, -0.0078, 0.0, 0.0039];
    const List<double> a = [1.0, -3.1806, 3.8612, -2.1122, 0.4351];
    
    List<double> w = List<double>.filled(5, 0.0);

    for (int i = 0; i < length; i++) {
      double out = b[0] * input[i] + w[1];
      w[1] = b[1] * input[i] - a[1] * out + w[2];
      w[2] = b[2] * input[i] - a[2] * out + w[3];
      w[3] = b[3] * input[i] - a[3] * out + w[4];
      w[4] = b[4] * input[i] - a[4] * out;
      filtered[i] = out;
    }
    return filtered;
  }

  /// Calculates haemoglobin (g/dL) using Red/Green AC/DC ratio
  /// Hgb = 18.2 - (6.8 * R_Hb) where R_Hb = (AC_Red/DC_Red) / (AC_Green/DC_Green)
  static double computeHaemoglobin(List<double> redSignal, List<double> greenSignal) {
    final double redPI = computePerfusionIndex(redSignal);
    final double greenPI = computePerfusionIndex(greenSignal);
    if (redPI == 0.0 || greenPI == 0.0) return 0.0;
    final double rHb = redPI / greenPI;
    final double hgb = 18.2 - (6.8 * rHb);
    return hgb.clamp(5.0, 20.0);
  }

  /// Calculates dynamic perfusion indices across raw signals
  static double computePerfusionIndex(List<double> rawSignal) {
    double maxVal = rawSignal[0];
    double minVal = rawSignal[0];
    double runningSum = 0.0;

    for (double val in rawSignal) {
      if (val > maxVal) maxVal = val;
      if (val < minVal) minVal = val;
      runningSum += val;
    }
    final double ac = maxVal - minVal;
    final double dc = runningSum / rawSignal.length;
    return dc == 0 ? 0.0 : (ac / dc) * 100.0;
  }
}