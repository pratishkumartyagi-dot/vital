import 'dart:math' as math;

class PPGSignalEngine {
  /// 4th-Order Butterworth Bandpass Filter (0.5Hz - 3.5Hz at Fs=30Hz)
  /// Implemented as cascaded Second-Order Sections (SOS) for numerical stability
  static List<double> applyBandpassFilter(List<double> input) {
    final int length = input.length;
    final List<double> filtered = List<double>.filled(length, 0.0);
    
    // SOS coefficients for 4th-order Butterworth bandpass (0.5-3.5 Hz, Fs=30 Hz)
    // Generated via: scipy.signal.butter(4, [0.5/15, 3.5/15], btype='band', output='sos')
    // Each section: [b0, b1, b2, a0, a1, a2] with a0=1.0
    const List<List<double>> sos = [
      [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
      [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
    ];

    // Cascade the two biquad sections
    List<double> stage = List<double>.from(input);
    for (final section in sos) {
      final b0 = section[0], b1 = section[1], b2 = section[2];
      final a0 = section[3], a1 = section[4], a2 = section[5];
      
      final List<double> out = List<double>.filled(length, 0.0);
      double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
      
      for (int i = 0; i < length; i++) {
        final x0 = stage[i];
        final y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        
        out[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
      }
      stage = out;
    }
    
    return stage;
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
  /// PI = (AC / DC) * 100%, where AC is mean pulse amplitude (peak-to-trough)
  /// and DC is the mean signal level
  static double computePerfusionIndex(List<double> rawSignal) {
    if (rawSignal.length < 10) return 0.0;
    
    // Apply bandpass to isolate pulsatile component
    final filtered = applyBandpassFilter(rawSignal);
    
    // Find peaks and troughs in filtered signal
    final List<int> peaks = [];
    final List<int> troughs = [];
    
    for (int i = 1; i < filtered.length - 1; i++) {
      if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1]) {
        peaks.add(i);
      }
      if (filtered[i] < filtered[i - 1] && filtered[i] < filtered[i + 1]) {
        troughs.add(i);
      }
    }
    
    if (peaks.isEmpty || troughs.isEmpty) return 0.0;
    
    // Match peaks to nearest troughs to calculate pulse amplitudes
    double totalAc = 0.0;
    int count = 0;
    
    for (final peakIdx in peaks) {
      // Find closest trough
      int closestTrough = troughs.reduce((a, b) => 
        (a - peakIdx).abs() < (b - peakIdx).abs() ? a : b);
      
      final double pulseAmp = (filtered[peakIdx] - filtered[closestTrough]).abs();
      totalAc += pulseAmp;
      count++;
    }
    
    if (count == 0) return 0.0;
    
    final double meanAc = totalAc / count;
    final double dc = rawSignal.reduce((a, b) => a + b) / rawSignal.length;
    
    return dc == 0 ? 0.0 : (meanAc / dc) * 100.0;
  }
}