import 'dart:math' as math;

class PPGSignalEngine {
  /// 4th-Order Butterworth Bandpass Filter (0.5Hz - 3.5Hz at Fs=30Hz)
  /// Implemented as cascaded Second-Order Sections (SOS) using RBJ audio-EQ cookbook
  /// biquads for numerical stability and correct frequency response.
  ///
  /// This replaces the previous hardcoded scipy-derived SOS coefficients which had
  /// a passband centred near 0.5 Hz and amplified it ~15x, corrupting PI, HR,
  /// and respiratory rate estimates.
  static List<double> applyBandpassFilter(List<double> input) {
    const double fs = 30.0;
    const double low = 0.5;
    const double high = 3.5;

    if (input.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
      return List<double>.from(input);
    }

    final hp = _highpassBiquad(low, fs);
    final lp = _lowpassBiquad(high, fs);
    final dc = input.reduce((a, b) => a + b) / input.length;

    // Forward pass: highpass (seeded with DC steady-state) then lowpass
    var y = _applyBiquad(input, hp, dc, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);

    // Zero-phase (filtfilt): reverse and run forward/backward
    y = y.reversed.toList();
    y = _applyBiquad(y, hp, 0.0, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);
    y = y.reversed.toList();

    return y;
  }

  /// Second-order Butterworth lowpass biquad (Q = 1/sqrt(2)) via RBJ cookbook.
  static _BiquadCoeffs _lowpassBiquad(double fc, double fs, [double q = math.SQRT1_2]) {
    final w0 = (2 * math.pi * fc) / fs;
    final cw = math.cos(w0);
    final sw = math.sin(w0);
    final alpha = sw / (2 * q);
    final a0 = 1 + alpha;
    return _BiquadCoeffs(
      b0: (1 - cw) / 2 / a0,
      b1: (1 - cw) / a0,
      b2: (1 - cw) / 2 / a0,
      a1: (-2 * cw) / a0,
      a2: (1 - alpha) / a0,
    );
  }

  /// Second-order Butterworth highpass biquad (Q = 1/sqrt(2)) via RBJ cookbook.
  static _BiquadCoeffs _highpassBiquad(double fc, double fs, [double q = math.SQRT1_2]) {
    final w0 = (2 * math.pi * fc) / fs;
    final cw = math.cos(w0);
    final sw = math.sin(w0);
    final alpha = sw / (2 * q);
    final a0 = 1 + alpha;
    return _BiquadCoeffs(
      b0: (1 + cw) / 2 / a0,
      b1: -(1 + cw) / a0,
      b2: (1 + cw) / 2 / a0,
      a1: (-2 * cw) / a0,
      a2: (1 - alpha) / a0,
    );
  }

  /// Apply a single biquad section with given initial conditions.
  static List<double> _applyBiquad(List<double> signal, _BiquadCoeffs c, double seedX, double seedY) {
    final out = List<double>.filled(signal.length, 0.0);
    double x1 = seedX, x2 = seedX, y1 = seedY, y2 = seedY;
    for (int i = 0; i < signal.length; i++) {
      final y = c.b0 * signal[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
      out[i] = y;
      x2 = x1;
      x1 = signal[i];
      y2 = y1;
      y1 = y;
    }
    return out;
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
  /// PI = (AC / DC) * 100%, where AC is RMS of bandpass-filtered signal
  /// and DC is the mean of the raw signal. This uses the same zero-phase
  /// bandpass as the web implementation to avoid baseline-wander inflation.
  static double computePerfusionIndex(List<double> rawSignal) {
    if (rawSignal.length < 10) return 0.0;

    // Apply zero-phase bandpass to isolate pulsatile component (0.5-3.5 Hz at 30 Hz)
    final filtered = applyBandpassFilter(rawSignal);
    
    const dc = rawSignal.reduce((a, b) => a + b) / rawSignal.length;
    if (dc <= 1e-9) return 0.0;

    // AC = RMS of filtered signal
    double sq = 0.0;
    for (final v in filtered) {
      sq += v * v;
    }
    final ac = math.sqrt(sq / filtered.length);
    
    return (ac / dc) * 100.0;
  }

  /// Respiratory bandpass: 0.06-0.7 Hz, zero-phase (filtfilt).
  /// Extracts respiratory baseline wander from cardiac PPG.
  static List<double> applyRespiratoryFilter(List<double> input, [double fs = 30.0]) {
    const double low = 0.06;
    const double high = 0.7;

    if (input.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
      return List<double>.from(input);
    }

    final hp = _highpassBiquad(low, fs);
    final lp = _lowpassBiquad(high, fs);
    final dc = input.reduce((a, b) => a + b) / input.length;

    var y = _applyBiquad(input, hp, dc, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);
    y = y.reversed.toList();
    y = _applyBiquad(y, hp, 0.0, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);
    y = y.reversed.toList();

    return y;
  }

  /// Cardiac bandpass: 0.5-3.5 Hz (non zero-phase, for envelope extraction).
  static List<double> applyCardiacFilter(List<double> input, [double fs = 30.0]) {
    return applyBandpassFilter(input);
  }
}

class _BiquadCoeffs {
  final double b0;
  final double b1;
  final double b2;
  final double a1;
  final double a2;
  const _BiquadCoeffs({
    required this.b0,
    required this.b1,
    required this.b2,
    required this.a1,
    required this.a2,
  });
}