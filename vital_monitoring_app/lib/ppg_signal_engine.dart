import 'dart:math' as math;

class CalibrationProfile {
  final String model;
  final String manufacturer;
  final double zlo;
  final bool matched;

  CalibrationProfile({
    required this.model,
    required this.manufacturer,
    required this.zlo,
    required this.matched,
  });

  factory CalibrationProfile.fromMap(Map<String, dynamic> map) {
    return CalibrationProfile(
      model: map['model']?.toString() ?? '',
      manufacturer: map['manufacturer']?.toString() ?? '',
      zlo: (map['zlo'] as num?)?.toDouble() ?? 16.0,
      matched: map['matched'] as bool? ?? false,
    );
  }

  static CalibrationProfile defaultProfile() {
    return CalibrationProfile(
      model: '',
      manufacturer: '',
      zlo: 16.0,
      matched: false,
    );
  }
}

class PPGSignalEngine {
  static const double cardiacLo = 0.5;
  static const double cardiacHi = 3.5;
  static const double noiseBandWidth = 6.0;
  static const double defaultFs = 30.0;

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

    var y = _applyBiquad(input, hp, dc, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);

    y = y.reversed.toList();
    y = _applyBiquad(y, hp, 0.0, 0.0);
    y = _applyBiquad(y, lp, 0.0, 0.0);
    y = y.reversed.toList();

    return y;
  }

  static List<double> applyBandpassFilterCustom(
      List<double> input, double low, double high, double fs) {
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

  // ZLO (Zero Light Offset) Correction
  // Based on: Xuan et al., "A calibration method for smartphone camera photoplethysmography"
  // Frontiers in Digital Health, 2023
  static double applyZloCorrection(double rawPixelValue, double zlo) {
    if (zlo <= 0.0) return rawPixelValue;
    final corrected = rawPixelValue - zlo;
    return corrected > 0.0 ? corrected : 0.0;
  }

  static List<double> applyZloCorrectionToSignal(List<double> signal, double zlo) {
    if (zlo <= 0.0) return List<double>.from(signal);
    return signal.map((v) => applyZloCorrection(v, zlo)).toList();
  }

  static double computeZloCorrectedDc(List<double> rawSignal, double zlo) {
    final correctedSignal = applyZloCorrectionToSignal(rawSignal, zlo);
    if (correctedSignal.isEmpty) return 0.0;
    return correctedSignal.reduce((a, b) => a + b) / correctedSignal.length;
  }

  static double computePerfusionIndex(List<double> rawSignal, [double zlo = 0.0]) {
    if (rawSignal.length < 10) return 0.0;

    final filtered = applyBandpassFilter(rawSignal);
    
    final dc = computeZloCorrectedDc(rawSignal, zlo);
    if (dc <= 1e-9) return 0.0;

    double sq = 0.0;
    for (final v in filtered) {
      sq += v * v;
    }
    final ac = math.sqrt(sq / filtered.length);
    
    return (ac / dc) * 100.0;
  }

  static double perfusionIndexACDC(List<double> signal, double fs, [double? dcRef, double zlo = 0.0]) {
    if (signal.length < 10) return 0.0;
    final dc = dcRef != null && dcRef > 1e-9 ? dcRef : (zlo > 0 ? computeZloCorrectedDc(signal, zlo) : dcLevel(signal));
    if (dc <= 1e-9) return 0.0;
    final ac = rms(applyBandpassFilterCustom(signal, cardiacLo, cardiacHi, fs));
    return (ac / dc) * 100.0;
  }

  static double filteredPerfusionIndex(List<double> signal, double fs, [double zlo = 0.0]) {
    if (signal.length < fs * 2) return 0.0;
    final filtered = applyBandpassFilterCustom(signal, cardiacLo, cardiacHi, fs);
    final dc = zlo > 0 ? computeZloCorrectedDc(signal, zlo) : (signal.reduce((a, b) => a + b, 0) / signal.length);
    if (dc <= 1e-9) return 0.0;
    return (rms(filtered) / dc) * 100.0;
  }

  static double computeHaemoglobin(List<double> redSignal, List<double> greenSignal, [double zlo = 0.0]) {
    final double redPI = computePerfusionIndex(redSignal, zlo);
    final double greenPI = computePerfusionIndex(greenSignal, zlo);
    if (redPI == 0.0 || greenPI == 0.0) return 0.0;
    final double rHb = redPI / greenPI;
    // Red/green RoR calibration (see ppg-analysis.ts HB_RED_GREEN_*). The old
    // `18.2 - 6.8 * R` was calibrated for the red/IR RoR (R ~ 0.4-0.8); a
    // smartphone camera has no IR channel, so the accepted proxy is red/green,
    // where R is ~0.6-1.8 because green (550 nm) is absorbed far more strongly
    // by haemoglobin than red (660 nm). The old formula returned 5-10 g/dL for
    // every healthy subject with R > 1.2 - the "grossly low" values reported.
    final double hgb = 18.0 - (2.8 * rHb);
    return hgb.clamp(10.0, 20.0).toDouble();
  }

  // Compute Ratio of Ratios (RoR) for SpO2 estimation with ZLO correction
  // RoR = (AC_red / DC_red) / (AC_green / DC_green)
  static double computeRatioOfRatios(
    List<double> redSignal,
    List<double> greenSignal,
    double fs, {
    double zlo = 0.0,
  }) {
    if (redSignal.length < fs * 2 || greenSignal.length < fs * 2) return 1.0;
    
    final redFiltered = applyBandpassFilterCustom(redSignal, cardiacLo, cardiacHi, fs);
    final greenFiltered = applyBandpassFilterCustom(greenSignal, cardiacLo, cardiacHi, fs);
    
    final redDc = zlo > 0 ? computeZloCorrectedDc(redSignal, zlo) : dcLevel(redSignal);
    final greenDc = zlo > 0 ? computeZloCorrectedDc(greenSignal, zlo) : dcLevel(greenSignal);
    
    if (redDc <= 1e-9 || greenDc <= 1e-9) return 1.0;
    
    final redAc = rms(redFiltered);
    final greenAc = rms(greenFiltered);
    
    if (greenAc <= 1e-9) return 1.0;
    
    final redRatio = redAc / redDc;
    final greenRatio = greenAc / greenDc;
    
    return redRatio / greenRatio;
  }

  // Estimate SpO2 from RoR using empirical calibration curve.
  // The old `110 - 25 * R` formula was calibrated for the red/IR ratio-of-ratios
  // (R ~ 0.4-0.8). A smartphone camera has no IR channel, so the accepted
  // proxy is red/green, where R is ~0.6-1.8 because green (550 nm) is absorbed
  // far more strongly by haemoglobin than red (660 nm). The old formula
  // returned SpO2 in the 60-85% range for every healthy subject with R > 1 -
  // the "grossly low" values reported. The new red/green calibration is
  // intercept 106, slope 8, range 88-100 (see ppg-analysis.ts
  // SPO2_RED_GREEN_*). Absolute accuracy is device-specific.
  static double estimateSpO2FromRor(double ror) {
    final spo2 = 106.0 - (8.0 * ror);
    return spo2.clamp(88.0, 100.0).toDouble();
  }

  static double computeHaemoglobinFromRor(
    List<double> redSignal,
    List<double> greenSignal,
    double fs, {
    double zlo = 0.0,
  }) {
    final ror = computeRatioOfRatios(redSignal, greenSignal, fs, zlo: zlo);
    // Red/green RoR calibration (see ppg-analysis.ts HB_RED_GREEN_*). The old
    // `18.2 - 6.8 * R` was calibrated for the red/IR RoR; a smartphone camera
    // has no IR channel, so the accepted proxy is red/green where R is ~0.6-1.8.
    final hgb = 18.0 - (2.8 * ror);
    return hgb.clamp(10.0, 20.0).toDouble();
  }

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

  static List<double> applyCardiacFilter(List<double> input, [double fs = 30.0]) {
    return applyBandpassFilter(input);
  }

  static double rms(List<double> signal) {
    if (signal.isEmpty) return 0.0;
    double sq = 0.0;
    for (final v in signal) sq += v * v;
    return math.sqrt(sq / signal.length);
  }

  static double dcLevel(List<double> signal) {
    if (signal.isEmpty) return 0.0;
    double sum = 0.0;
    for (final v in signal) sum += math.abs(v);
    return sum / signal.length;
  }

  static double channelAcDc(List<double> signal) {
    if (signal.length < 3) return 0.0;
    final n = signal.length;
    final dc = signal.reduce((a, b) => a + b, 0) / n;
    if (dc <= 1e-9) return 0.0;
    double sq = 0.0;
    for (final v in signal) sq += (v - dc) * (v - dc);
    final ac = math.sqrt(sq / n);
    return ac / dc;
  }

  static List<Peak> detectPeaks(List<double> signal, double fs,
      {int minBpm = 40, int maxBpm = 180}) {
    final filtered = applyBandpassFilterCustom(detrend(signal), cardiacLo, cardiacHi, fs, true);
    final peaks = <Peak>[];
    final minDist = math.max(1, (60 / maxBpm) * fs).floor();
    final sorted = List<double>.from(filtered)..sort();
    final n = sorted.length;
    final p10 = sorted[(n * 0.10).floor()];
    final p90 = sorted[(n * 0.90).floor()];
    final p97 = sorted[(n * 0.97).floor()];
    final minHeight = math.max(p10 + 0.5 * (p90 - p10), 0.45 * p97);

    for (int i = 1; i < filtered.length - 1; i++) {
      if (filtered[i] > filtered[i - 1] && filtered[i] >= filtered[i + 1]) {
        if (filtered[i] > minHeight) {
          if (peaks.isEmpty || i - peaks.last.index >= minDist) {
            peaks.add(Peak(index: i, value: filtered[i]));
          } else if (filtered[i] > peaks.last.value) {
            peaks[peaks.length - 1] = Peak(index: i, value: filtered[i]);
          }
        }
      }
    }
    return peaks;
  }

  static HeartMetrics analyzeHeartRate(List<double> ppg, double fs, [double? dcRef]) {
    if (ppg.length < fs * 5) {
      return HeartMetrics(bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0);
    }

    final trimmed = ppg.sublist(math.min(fs.toInt(), ppg.length).toInt());
    final peaks = detectPeaks(trimmed, fs);
    if (peaks.length < 3) {
      return HeartMetrics(bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0);
    }

    final valid = <Peak>[];
    for (final p in peaks) {
      if (valid.isEmpty) {
        valid.add(p);
        continue;
      }
      final dtMs = ((p.index - valid.last.index) / fs) * 1000;
      if (dtMs >= 300) valid.add(p);
    }
    if (valid.length < 3) {
      return HeartMetrics(bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0);
    }

    final rrAll = <double>[];
    for (int i = 1; i < valid.length; i++) {
      rrAll.add(((valid[i].index - valid[i - 1].index) / fs) * 1000);
    }

    final med = median(rrAll);
    final filteredRr = med > 0 ? rrAll.where((d) => (d - med).abs() / med <= 0.3).toList() : List<double>.from(rrAll);
    final rrMs = filteredRr.length >= 2 ? filteredRr : rrAll;
    final meanRR = mean(rrMs);
    final bpm = meanRR > 0 ? 60000 / meanRR : 0;
    final hrv = computeHRV(rrMs);
    final rhythm = classifyRhythm(rrMs);

    final filtered = applyBandpassFilterCustom(detrend(trimmed), cardiacLo, cardiacHi, fs, true);
    final quality = computeSignalQuality(filtered, trimmed, valid, fs, dcRef ?? dcLevel(ppg));

    return HeartMetrics(
      bpm: bpm > 30 && bpm < 220 ? bpm : null,
      peaks: valid.map((p) => p.index).toList(),
      rrIntervals: rrMs,
      hrv: hrv,
      rhythm: rhythm,
      signalQuality: quality,
    );
  }

  static double computeSignalQuality(List<double> filtered, List<double> raw, List<Peak> peaks, double fs, double dcRef) {
    if (peaks.length < 2) return 0.0;

    final dc = dcRef > 1e-9 ? dcRef : dcLevel(raw);
    final ac = rms(filtered);
    final pi = dc > 1e-9 ? (ac / dc) * 100 : 0.0;
    final piScore = clamp01(math.sqrt(pi / 0.5));

    final snrResult = spectralSnr(raw, fs);
    final snrScore = snrResult.snr > 0 ? clamp01((math.log(snrResult.snr) / math.log(10) + 0.3) / 1.2) : 0.0;

    final rrMs = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      rrMs.add(((peaks[i].index - peaks[i - 1].index) / fs) * 1000);
    }
    final med = median(rrMs);
    final filteredRr = med > 0 ? rrMs.where((d) => (d - med).abs() / med <= 0.3).toList() : List<double>.from(rrMs);
    final rrMsClean = filteredRr.length >= 2 ? filteredRr : rrMs;
    final cv = rrMsClean.length > 1 ? (stddev(rrMsClean) / mean(rrMsClean)) * 100 : 0.0;
    final regScore = cv < 10 ? 1.0 : math.max(0.0, 1.0 - (cv - 10) / 30).toDouble();

    final prominence = cardiacSpectralProminence(raw, fs);
    final pulseGate = clamp01((prominence - 4) / 12);

    final base = 0.4 * snrScore + 0.35 * piScore + 0.25 * regScore;
    return (clamp01(base * pulseGate) * 100).roundToDouble();
  }

  static SpectralSnr spectralSnr(List<double> signal, double fs) {
    const none = SpectralSnr(snr: 0, signalPower: 0, noiseDensity: 0, sigBins: 0, noiseBins: 0, hasReference: false);
    final ps = powerSpectrum(signal, fs);
    if (ps.freqs.isEmpty) return none;

    final noiseHi = math.min((fs / 2) * 0.9, cardiacHi + noiseBandWidth).toDouble();
    double signalPower = 0.0;
    int sigBins = 0;
    double noisePower = 0.0;
    int noiseBins = 0;

    for (int i = 0; i < ps.freqs.length; i++) {
      final f = ps.freqs[i];
      if (f >= cardiacLo && f <= cardiacHi) {
        signalPower += ps.psd[i];
        sigBins++;
      } else if (f > cardiacHi && f <= noiseHi) {
        noisePower += ps.psd[i];
        noiseBins++;
      }
    }
    if (sigBins == 0) return none;

    final noiseDensity = noiseBins > 0 ? noisePower / noiseBins : 0.0;
    final inBandNoise = noiseDensity * sigBins;
    if (inBandNoise <= 0) {
      return SpectralSnr(snr: double.infinity, signalPower: signalPower, noiseDensity: noiseDensity, sigBins: sigBins, noiseBins: noiseBins, hasReference: false);
    }
    return SpectralSnr(snr: signalPower / inBandNoise, signalPower: signalPower, noiseDensity: noiseDensity, sigBins: sigBins, noiseBins: noiseBins, hasReference: true);
  }

  static double cardiacSpectralProminence(List<double> signal, double fs) {
    final ps = powerSpectrum(signal, fs);
    final band = <double>[];
    for (int i = 0; i < ps.freqs.length; i++) {
      if (ps.freqs[i] >= cardiacLo && ps.freqs[i] <= cardiacHi) {
        band.add(ps.psd[i]);
      }
    }
    if (band.length < 8) return 0.0;
    final sorted = List<double>.from(band)..sort();
    final median = sorted[sorted.length ~/ 2];
    final peak = sorted.last;
    return median > 0 ? peak / median : 0.0;
  }

  static PowerSpectrumResult powerSpectrum(List<double> signal, double fs) {
    final n = signal.length;
    final freqs = <double>[];
    final psd = <double>[];
    if (n < 8) return PowerSpectrumResult(freqs: freqs, psd: psd);

    final detrended = detrend(signal);
    final win = List<double>.filled(n, 0.0);
    double winSqSum = 0.0;
    for (int i = 0; i < n; i++) {
      final w = 0.5 * (1 - math.cos((2 * math.pi * i) / (n - 1)));
      win[i] = w;
      winSqSum += w * w;
    }
    final wNorm = winSqSum / n;
    if (wNorm == 0) {
      for (int k = 0; k < (n / 2).floor(); k++) {
        freqs.add((k * fs) / n);
        psd.add(0.0);
      }
      return PowerSpectrumResult(freqs: freqs, psd: psd);
    }

    for (int k = 0; k < (n / 2).floor(); k++) {
      double real = 0.0;
      double imag = 0.0;
      final angle = (2 * math.pi * k) / n;
      for (int i = 0; i < n; i++) {
        final phi = angle * i;
        real += detrended[i] * win[i] * math.cos(phi);
        imag -= detrended[i] * win[i] * math.sin(phi);
      }
      freqs.add((k * fs) / n);
      psd.add((real * real + imag * imag) / n / wNorm);
    }
    return PowerSpectrumResult(freqs: freqs, psd: psd);
  }

  static List<double> posPulse(List<double> red, List<double> green, List<double> blue, [int winLen = 32]) {
    final n = math.min(red.length, math.min(green.length, blue.length)).toInt();
    if (n < winLen || winLen < 8) return List<double>.empty();

    final hop = math.max(1, (winLen / 2).floor()).toInt();
    final sqrt6 = math.sqrt(6);
    final acc = List<double>.filled(n, 0.0);
    final wsum = List<double>.filled(n, 0.0);

    for (int start = 0; start + winLen <= n; start += hop) {
      double mr = 0, mg = 0, mb = 0;
      for (int i = 0; i < winLen; i++) {
        mr += red[start + i];
        mg += green[start + i];
        mb += blue[start + i];
      }
      mr /= winLen;
      mg /= winLen;
      mb /= winLen;
      if (mr <= 1 || mg <= 1 || mb <= 1) continue;

      final cnR = List<double>.filled(winLen, 0.0);
      final cnG = List<double>.filled(winLen, 0.0);
      final cnB = List<double>.filled(winLen, 0.0);
      for (int i = 0; i < winLen; i++) {
        cnR[i] = red[start + i] / mr;
        cnG[i] = green[start + i] / mg;
        cnB[i] = blue[start + i] / mb;
      }

      final s1 = List<double>.filled(winLen, 0.0);
      final s2 = List<double>.filled(winLen, 0.0);
      double m1 = 0, m2 = 0;
      for (int i = 0; i < winLen; i++) {
        s1[i] = (cnG[i] - cnB[i]) / sqrt6;
        s2[i] = (-2 * cnR[i] + cnG[i] + cnB[i]) / sqrt6;
        m1 += s1[i];
        m2 += s2[i];
      }
      m1 /= winLen;
      m2 /= winLen;

      double v1 = 0, v2 = 0;
      for (int i = 0; i < winLen; i++) {
        v1 += (s1[i] - m1) * (s1[i] - m1);
        v2 += (s2[i] - m2) * (s2[i] - m2);
      }
      final alpha = v2 > 1e-12 ? math.sqrt(v1 / v2) : 0.0;

      for (int i = 0; i < winLen; i++) {
        final h = s1[i] - m1 + alpha * (s2[i] - m2);
        final w = 0.5 * (1 - math.cos((2 * math.pi * i) / (winLen - 1)));
        acc[start + i] += h * w;
        wsum[start + i] += w;
      }
    }

    final out = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) {
      out[i] = wsum[i] > 1e-9 ? acc[i] / wsum[i] : 0.0;
    }

    double gSum = 0.0;
    for (int i = 0; i < n; i++) gSum += green[i];
    final unitScale = math.sqrt(3) * (gSum / n);
    for (int i = 0; i < n; i++) out[i] *= unitScale;

    return out;
  }

  static List<double> combinePpgChannels(List<double> red, List<double> green, List<double> blue) {
    final n = math.min(red.length, math.min(green.length, blue.length)).toInt();
    if (n == 0) return green.isNotEmpty ? List<double>.from(green) : (red.isNotEmpty ? List<double>.from(red) : List<double>.from(blue));

    final dcR = red.reduce((a, b) => a + b, 0) / n;
    final dcG = green.reduce((a, b) => a + b, 0) / n;
    final dcB = blue.reduce((a, b) => a + b, 0) / n;

    final piR = channelAcDc(red);
    final piG = channelAcDc(green);
    final piB = channelAcDc(blue);
    final totalPi = piR + piG + piB;

    final wR = totalPi > 1e-9 ? piR / totalPi : 0.25;
    final wG = totalPi > 1e-9 ? piG / totalPi : 0.5;
    final wB = totalPi > 1e-9 ? piB / totalPi : 0.25;

    final result = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) {
      final rNorm = (red[i] - dcR) / dcR;
      final gNorm = (green[i] - dcG) / dcG;
      final bNorm = (blue[i] - dcB) / dcB;
      final combined = wR * rNorm + wG * gNorm + wB * bNorm;
      result[i] = dcG + combined * dcG;
    }
    return result;
  }

  static List<double> selectBestPpgChannel(List<double> red, List<double> green, List<double> blue, double fs) {
    final n = math.min(red.length, green.length).toInt();
    if (n < (fs * 2).floor()) return green.sublist(0, math.min(n, green.length).toInt());

    final satThreshold = 250.0;
    final redSatFrac = red.where((v) => v >= satThreshold).length / red.length;
    final greenSatFrac = green.where((v) => v >= satThreshold).length / green.length;

    final redFiltered = applyBandpassFilterCustom(red, cardiacLo, 8.0, fs, true);
    final greenFiltered = applyBandpassFilterCustom(green, cardiacLo, 8.0, fs, true);

    final redSnr = spectralSnr(redFiltered, fs);
    final greenSnr = spectralSnr(greenFiltered, fs);

    final redScore = redSatFrac > 0.1 ? -1.0 : redSnr.snr;
    final greenScore = greenSatFrac > 0.1 ? -1.0 : greenSnr.snr;

    if (greenScore >= redScore || greenScore >= redScore * 0.8) {
      return List<double>.from(greenFiltered);
    }
    return List<double>.from(redFiltered);
  }

  static CreatinineFeatures extractCreatinineFeatures(
      List<double> ppg, List<double> red, List<double> green, List<double> blue,
      List<int> peakIndices, double fs) {
    final empty = CreatinineFeatures.empty();
    if (ppg.length < fs * 5 || peakIndices.length < 3) return empty;

    final filtered = applyBandpassFilterCustom(detrend(ppg), cardiacLo, cardiacHi, fs);
    final min = filtered.reduce(math.min);
    final max = filtered.reduce(math.max);
    final range = max - min;
    if (range <= 1e-9) return empty;
    final norm = filtered.map((v) => (v - min) / range).toList();

    final intervals = <double>[];
    for (int i = 1; i < peakIndices.length; i++) {
      final len = peakIndices[i] - peakIndices[i - 1];
      if (len >= fs * 0.3 && len <= fs * 2) intervals.add(len.toDouble());
    }
    if (intervals.length < 2) return empty;

    final cycleLen = mean(intervals).round();
    if (cycleLen < 4) return empty;

    final cycles = <List<double>>[];
    for (int i = 1; i < peakIndices.length; i++) {
      final start = peakIndices[i - 1];
      if (start + cycleLen <= norm.length) {
        cycles.add(norm.sublist(start, start + cycleLen));
      }
    }
    if (cycles.length < 2) return empty;

    final avgCycle = List<double>.filled(cycleLen, 0.0);
    for (final c in cycles) {
      for (int j = 0; j < cycleLen; j++) avgCycle[j] += c[j];
    }
    for (int j = 0; j < cycleLen; j++) avgCycle[j] /= cycles.length;

    int sysIdx = 0;
    for (int j = 0; j < cycleLen; j++) if (avgCycle[j] > avgCycle[sysIdx]) sysIdx = j;
    final peakAmp = avgCycle[sysIdx];

    int notchIdx = sysIdx;
    final notchSearchEnd = math.min(cycleLen, sysIdx + (fs * 0.5).floor()).toInt();
    for (int j = sysIdx + 1; j < notchSearchEnd; j++) {
      if (avgCycle[j] < avgCycle[notchIdx]) notchIdx = j;
    }
    final dicroticAmp = avgCycle[notchIdx];

    int diasIdx = notchIdx;
    for (int j = notchIdx + 1; j < cycleLen; j++) {
      if (avgCycle[j] > avgCycle[diasIdx]) diasIdx = j;
    }

    final augmentationIndex = peakAmp > 1e-6 ? avgCycle[diasIdx] / peakAmp : 0.0;
    final stiffnessRatio = peakAmp > 1e-6 ? (peakAmp - dicroticAmp) / peakAmp : 0.0;
    final riseSpan = math.max(1, sysIdx).toInt();
    final riseSlope = (avgCycle[sysIdx] - avgCycle[0]) / riseSpan;

    final midLevel = (avgCycle[0] + avgCycle[sysIdx]) / 2;
    int midIdx = 0;
    for (int j = 1; j <= sysIdx; j++) {
      if (avgCycle[j] >= midLevel) {
        midIdx = j;
        break;
      }
    }
    final upstrokeMidpoint = midIdx / cycleLen;
    final diastolicPeakPos = diasIdx / cycleLen;

    final d1 = <double>[];
    for (int j = 1; j < cycleLen; j++) d1.add(avgCycle[j] - avgCycle[j - 1]);
    int firstMax = -1, secondMax = -1;
    for (int j = 1; j < d1.length - 1; j++) {
      if (d1[j] > d1[j - 1] && d1[j] > d1[j + 1]) {
        if (firstMax < 0) firstMax = j;
        else {
          secondMax = j;
          break;
        }
      }
    }
    final secondDerivTime = secondMax > 0 ? secondMax / cycleLen : 0.0;

    int firstMin = -1;
    for (int j = 1; j < d1.length - 1; j++) {
      if (d1[j] < d1[j - 1] && d1[j] < d1[j + 1]) {
        firstMin = j;
        break;
      }
    }
    final firstDerivMinTime = firstMin > 0 ? firstMin / cycleLen : 0.0;

    final diastolicNotchDuration = math.max(0.0, (diasIdx - notchIdx) / cycleLen).toDouble();
    final systolicPeakDuration = sysIdx / cycleLen;

    double totalArea = 0.0;
    final cumulative = <double>[0.0];
    for (int j = 1; j < cycleLen; j++) {
      final area = (avgCycle[j] + avgCycle[j - 1]) / 2;
      totalArea += area;
      cumulative.add(totalArea);
    }
    final inflectionIdx = firstMax > 0 ? firstMax : sysIdx;
    final inflectionAreaRatio = totalArea > 1e-9 ? cumulative[math.min(inflectionIdx, cumulative.length - 1).toInt()] / totalArea : 0.0;

    final pulseInterval = mean(intervals) / fs;
    final rrMs = intervals.map((v) => (v / fs) * 1000).toList();
    final rrCv = mean(rrMs) > 0 ? stddev(rrMs) / mean(rrMs) : 0.0;

    final ppgPI = filteredPerfusionIndex(ppg, fs);
    final greenPI = filteredPerfusionIndex(green.isNotEmpty ? green : ppg, fs);
    final redPI = filteredPerfusionIndex(red.isNotEmpty ? red : ppg, fs);
    final bluePI = filteredPerfusionIndex(blue.isNotEmpty ? blue : ppg, fs);
    final perfusionIndex = ppgPI > 0 ? ppgPI : (greenPI > 0 ? greenPI : (redPI > 0 ? redPI : 0.0));
    final redGreenRatio = greenPI > 0 ? redPI / greenPI : 1.0;
    final redBlueRatio = bluePI > 0 ? redPI / bluePI : 1.0;

    final psd = powerSpectrum(avgCycle, fs).psd;
    final harmonicRatio = psd.length > 3 ? (psd[2] + psd[3]) / psd[1] : 0.0;

    return CreatinineFeatures(
      peakAmp: peakAmp,
      dicroticAmp: dicroticAmp,
      pulseInterval: pulseInterval,
      rrCv: rrCv,
      augmentationIndex: augmentationIndex,
      stiffnessRatio: stiffnessRatio,
      diastolicPeakPos: diastolicPeakPos,
      upstrokeMidpoint: upstrokeMidpoint,
      riseSlope: riseSlope,
      secondDerivTime: secondDerivTime,
      diastolicNotchDuration: diastolicNotchDuration,
      systolicPeakDuration: systolicPeakDuration,
      inflectionAreaRatio: inflectionAreaRatio,
      firstDerivMinTime: firstDerivMinTime,
      perfusionIndex: perfusionIndex,
      redGreenRatio: redGreenRatio,
      redBlueRatio: redBlueRatio,
      harmonicRatio: harmonicRatio,
      beats: cycles.length,
    );
  }

  static CreatinineMetrics estimateCreatinine(
      List<double> ppg, List<double> red, List<double> green, List<double> blue,
      List<int> peakIndices, double? bpm, double signalQuality, double fs,
      String cameraMode, int? ageYears, String? sex) {
    final method = cameraMode == 'front' ? 'rppg-front' : 'ppg-rear-ftir';
    final features = extractCreatinineFeatures(ppg, red, green, blue, peakIndices, fs);
    final invalid = CreatinineMetrics(
      creatinine: null, creatinineUmol: null, creatinineClass: null,
      egfr: null, stage: null, confidence: 0.0, method: method,
    );
    if (ppg.length < fs * 10 || features.beats < 3 || bpm == null || signalQuality < 40) return invalid;

    final isMale = sex == 'male';
    final isFront = cameraMode == 'front';

    final amplitude = normalizeRange(features.peakAmp, 0.05, 0.95);
    final perfusion = isFront
        ? normalizeRange(features.perfusionIndex, 0.02, 0.8)
        : normalizeRange(features.perfusionIndex, 0.15, 3.0);
    final redGreen = normalizeRange(features.redGreenRatio, 0.6, 1.8);
    final stiffness = normalizeRange(features.stiffnessRatio, 0.05, 0.9);
    final augmentation = normalizeRange(features.augmentationIndex, 0.1, 1.2);
    final harmonic = normalizeRange(features.harmonicRatio, 0.05, 1.5);
    final rrCv = normalizeRange(features.rrCv, 0.0, 0.3);
    final notchTime = normalizeRange(features.diastolicNotchDuration, 0.02, 0.5);
    final systolicTime = normalizeRange(features.systolicPeakDuration, 0.05, 0.5);
    final ipa = normalizeRange(features.inflectionAreaRatio, 0.1, 0.7);
    final derivMin = normalizeRange(features.firstDerivMinTime, 0.05, 0.6);

    var creatinine =
        0.85 +
        (isMale ? 0.15 : 0.0) -
        0.55 * (perfusion - 0.5) -
        0.3 * (amplitude - 0.5) -
        0.22 * (redGreen - 0.5) +
        0.28 * (stiffness - 0.5) +
        0.18 * (augmentation - 0.5) +
        0.15 * (harmonic - 0.5) +
        0.12 * (notchTime - 0.5) +
        0.1 * (systolicTime - 0.5) -
        0.14 * (ipa - 0.5) +
        0.08 * (derivMin - 0.5) +
        0.1 * rrCv;

    creatinine = math.min(12.0, math.max(0.4, creatinine)).toDouble();
    creatinine = (creatinine * 100).roundToDouble() / 100;

    final creatinineClass = creatinine > 1.3 ? 'high' : (creatinine < 0.6 ? 'low' : 'normal');
    final sqNorm = clamp01(signalQuality / 100);
    final beatAdequacy = clamp01(features.beats / 15);
    final confidence = math.min(isFront ? 0.55 : 0.8, 0.12 + 0.55 * sqNorm + 0.15 * beatAdequacy).toDouble();

    final age = ageYears != null && ageYears > 0 ? ageYears : 45;
    final egfr = estimateEgfr(creatinine, age, sex);

    return CreatinineMetrics(
      creatinine: creatinine,
      creatinineUmol: (creatinine * 88.4 * 10).roundToDouble() / 10,
      creatinineClass: creatinineClass,
      egfr: egfr?.roundToDouble(),
      stage: kdigoStage(egfr),
      confidence: (confidence * 100).roundToDouble() / 100,
      method: method,
    );
  }

  static double? estimateEgfr(double? creatinineMgDl, int? ageYears, String? sex) {
    if (creatinineMgDl == null || creatinineMgDl <= 0 || !creatinineMgDl.isFinite) return null;
    final age = ageYears != null && ageYears > 0 ? ageYears : 45;
    final female = sex == 'female';
    final k = female ? 0.7 : 0.9;
    final a = female ? -0.241 : -0.302;
    final ratio = creatinineMgDl / k;
    final egfr = (142 *
        math.pow(math.min(ratio, 1), a).toDouble() *
        math.pow(math.max(ratio, 1), -1.2).toDouble() *
        math.pow(0.9938, age).toDouble() *
        (female ? 1.012 : 1.0));
    return egfr.toDouble();
  }

  static String? kdigoStage(double? egfr) {
    if (egfr == null) return null;
    if (egfr >= 90) return 'G1';
    if (egfr >= 60) return 'G2';
    if (egfr >= 45) return 'G3a';
    if (egfr >= 30) return 'G3b';
    if (egfr >= 15) return 'G4';
    return 'G5';
  }

  static List<int> _findRespPeaks(List<double> signal) {
    return _findPeaks(signal, minDistance: 45, threshold: 0.1);
  }

  static double _peaksToRate(List<int> peaks, double fs) {
    if (peaks.length < 2) return 0.0;
    final intervals = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      final interval = (peaks[i] - peaks[i - 1]) / fs;
      if (interval >= 1.2 && interval <= 6.0) intervals.add(interval);
    }
    if (intervals.isEmpty) return 0.0;
    final avgInterval = intervals.reduce((a, b) => a + b) / intervals.length;
    if (avgInterval <= 0) return 0.0;
    return 60.0 / avgInterval;
  }

  static (double, double) _findDominantFreq(List<double> signal, double fs, double fLow, double fHigh) {
    if (signal.length < 4) return (0.0, 0.0);

    final n = signal.length;
    final meanVal = mean(signal);
    final centered = signal.map((v) => v - meanVal).toList();

    double bestFreq = 0.0;
    double bestPower = 0.0;
    final freqRes = fs / n;
    final kMin = (fLow / freqRes).ceil();
    final kMax = (fHigh / freqRes).floor();

    for (int k = kMin; k <= kMax && k < n ~/ 2; k++) {
      double real = 0.0;
      double imag = 0.0;
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

  static List<int> _findPeaks(List<double> signal, {int minDistance = 1, double threshold = 0.0}) {
    final peaks = <int>[];
    if (signal.length < 3) return peaks;

    final meanVal = mean(signal);
    final stdDevVal = stddev(signal);
    final dynamicThreshold = threshold * stdDevVal;

    for (int i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > dynamicThreshold) {
        if (peaks.isEmpty || i - peaks.last >= minDistance) {
          peaks.add(i);
        }
      }
    }
    return peaks;
  }

  static List<double> _amplitudeEnvelope(List<double> signal) {
    if (signal.isEmpty) return List<double>.empty();
    const winSize = 15;
    final result = List<double>.filled(signal.length, 0.0);
    for (int i = 0; i < signal.length; i++) {
      final start = (i - winSize).clamp(0, signal.length - 1).toInt();
      final end = (i + winSize + 1).clamp(0, signal.length).toInt();
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

  /// Public wrapper around the internal two-stage respiratory-rate
  /// estimator (cardiac-band AM envelope + baseline-wander fusion).
  static double estimateRespiratoryRate(List<double> signal) {
    return _estimateRespiratoryRate(signal);
  }

  static double _estimateRespiratoryRate(List<double> signal) {
    if (signal.length < 180) return 15.0;

    const double fs = 30.0;

    final cardiac = applyCardiacFilter(signal);
    final envelope = _amplitudeEnvelope(cardiac);

    final envelopeResp = applyRespiratoryFilter(envelope, fs);
    final (envFreq, envPower) = _findDominantFreq(envelopeResp, fs, 0.1, 0.5);
    final envPeaks = _findRespPeaks(envelopeResp);
    final envRate = envPeaks.isNotEmpty ? _peaksToRate(envPeaks, fs) : 0.0;

    final mean = signal.reduce((a, b) => a + b) / signal.length;
    final detrended = signal.map((v) => v - mean).toList();
    final baseline = applyRespiratoryFilter(detrended, fs);
    final (baseFreq, basePower) = _findDominantFreq(baseline, fs, 0.1, 0.5);
    final basePeaks = _findRespPeaks(baseline);
    final baseRate = basePeaks.isNotEmpty ? _peaksToRate(basePeaks, fs) : 0.0;

    final candidates = <double>[];
    if (envRate > 0) candidates.add(envRate);
    if (baseRate > 0) candidates.add(baseRate);
    if (envFreq > 0) candidates.add(envFreq * 60);
    if (baseFreq > 0) candidates.add(baseFreq * 60);

    double respBpm = 0.0;
    if (candidates.isNotEmpty) {
      final sorted = List<double>.from(candidates)..sort();
      final medianRate = sorted[sorted.length ~/ 2];
      final valid = sorted.where((r) => (r - medianRate).abs() < 6).toList();
      respBpm = valid.isNotEmpty
          ? valid.reduce((a, b) => a + b) / valid.length
          : medianRate;
    }

    if (respBpm < 5 || respBpm > 50) return 15.0;
    return respBpm.clamp(8.0, 35.0).toDouble();
  }

  static double mean(List<double> arr) {
    if (arr.isEmpty) return 0.0;
    return arr.reduce((a, b) => a + b) / arr.length;
  }

  static double stddev(List<double> arr) {
    if (arr.length < 2) return 0.0;
    final m = mean(arr);
    final v = arr.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) / (arr.length - 1);
    return math.sqrt(v);
  }

  static double median(List<double> arr) {
    if (arr.isEmpty) return 0.0;
    final s = List<double>.from(arr)..sort();
    final m = s.length ~/ 2;
    return s.length % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  static List<double> detrend(List<double> signal) {
    final n = signal.length;
    if (n < 3) return List<double>.from(signal);
    final x = List<double>.generate(n, (i) => i.toDouble());
    final xMean = mean(x);
    final yMean = mean(signal);
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
      num += (x[i] - xMean) * (signal[i] - yMean);
      den += (x[i] - xMean) * (x[i] - xMean);
    }
    final slope = den == 0 ? 0.0 : num / den;
    final intercept = yMean - slope * xMean;
    final result = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) {
      result[i] = signal[i] - (slope * i + intercept);
    }
    return result;
  }

  static double clamp01(double v) {
    if (!v.isFinite) return 0.0;
    return math.min(1.0, math.max(0.0, v)).toDouble();
  }

  static double normalizeRange(double v, double lo, double hi) {
    if (hi == lo) return 0.0;
    return clamp01((v - lo) / (hi - lo));
  }

  static HRVMetrics computeHRV(List<double> rrMs) {
    if (rrMs.length < 2) {
      return const HRVMetrics(meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0);
    }
    final diffs = <double>[];
    double pnn50 = 0.0;
    for (int i = 1; i < rrMs.length; i++) {
      final d = (rrMs[i] - rrMs[i - 1]).abs();
      diffs.add(d);
      if (d > 50) pnn50++;
    }
    final meanRR = mean(rrMs);
    final sdnn = stddev(rrMs);
    final rmssd = diffs.isNotEmpty ? math.sqrt(mean(diffs.map((d) => d * d).toList())) : 0.0;
    final pnn = diffs.isNotEmpty ? (pnn50 / diffs.length) * 100 : 0.0;
    return HRVMetrics(meanRR: meanRR, sdnn: sdnn, rmssd: rmssd, pnn50: pnn);
  }

  static String classifyRhythm(List<double> rrMs) {
    if (rrMs.length < 5) return 'unknown';
    final sd = stddev(rrMs);
    final m = mean(rrMs);
    if (m == 0) return 'unknown';
    final cv = (sd / m) * 100;
    if (cv < 20) return 'regular';
    return 'irregular';
  }

  static VitalAnalysis analyzeVitals(
      List<double> ppg, double fs,
      List<double> redSignal, List<double> greenSignal, List<double> blueSignal,
      [AnalyzeOptions? options]) {
    final opts = options ?? const AnalyzeOptions();
    final dcRef = opts.dcRef ?? dcLevel(greenSignal.isNotEmpty ? greenSignal : ppg);
    final heart = analyzeHeartRate(ppg, fs, dcRef);
    final resp = _estimateRespRate(ppg, fs);
    final bp = _estimateBloodPressure(ppg, heart.peaks, heart.rrIntervals, heart.hrv, fs);
    final glucose = _estimateGlucose(ppg, heart.peaks, redSignal, greenSignal, blueSignal, fs, heart.signalQuality);
    final creatinine = estimateCreatinine(
      ppg, redSignal, greenSignal, blueSignal, heart.peaks, heart.bpm, heart.signalQuality, fs,
      opts.cameraMode ?? 'rear', opts.ageYears, opts.sex,
    );

    return VitalAnalysis(
      heart: heart,
      resp: resp,
      bpm: heart.bpm,
      respRate: resp.respRate,
      hrv: heart.hrv,
      sdnn: heart.hrv?.sdnn ?? 0,
      rmssd: heart.hrv?.rmssd ?? 0,
      pnn50: heart.hrv?.pnn50 ?? 0,
      rhythm: heart.rhythm,
      bp: bp,
      bpDeep: null,
      glucose: glucose,
      creatinine: creatinine,
    );
  }

  static BloodPressureMetrics _estimateBloodPressure(
      List<double> ppg, List<int> peakIndices, List<double> rrMs, HRVMetrics? hrv, double fs) {
    if (ppg.length < fs * 5 || peakIndices.length < 2) {
      return const BloodPressureMetrics(confidence: 0);
    }

    final features = _extractPPGFeatures(ppg, peakIndices, fs);
    final heartRate = rrMs.isNotEmpty ? 60000 / mean(rrMs) : 70.0;

    final stiffnessFactor = (features['stiffnessIndex'] as double) / 100.0;
    final reflectionFactor = (features['reflectionIndex'] as double) / 2.0;

    var baseSystolic = 110 + (heartRate - 70) * 0.3;
    var baseDiastolic = 70 + (heartRate - 70) * 0.1;

    final sysAdjust = (stiffnessFactor * 15) - (reflectionFactor * 10) + (hrv?.sdnn ?? 0) * 0.1;
    final diaAdjust = (stiffnessFactor * 5) - (reflectionFactor * 5) + (hrv?.rmssd ?? 0) * 0.05;

    baseSystolic += sysAdjust;
    baseDiastolic += diaAdjust;

    final pulsePressure = baseSystolic - baseDiastolic;
    final map = baseDiastolic + pulsePressure / 3;
    final confidence = math.min(0.7, 0.3 + stiffnessFactor * 0.2 + reflectionFactor * 0.2).toDouble();

    return BloodPressureMetrics(
      systolic: math.max(80, math.min(200, baseSystolic)).toDouble(),
      diastolic: math.max(50, math.min(130, baseDiastolic)).toDouble(),
      map: math.max(60, math.min(150, map)).toDouble(),
      pulsePressure: math.max(20, math.min(100, pulsePressure)).toDouble(),
      confidence: confidence,
    );
  }

  static Map<String, double> _extractPPGFeatures(List<double> ppg, List<int> peakIndices, double fs) {
    final n = ppg.length;
    final filtered = applyBandpassFilterCustom(detrend(ppg), cardiacLo, cardiacHi, fs);
    final min = filtered.reduce(math.min);
    final max = filtered.reduce(math.max);
    final range = max - min;
    final norm = filtered.map((v) => range > 1e-9 ? (v - min) / range : 0.0).toList();

    int peakIdx = peakIndices.isNotEmpty ? peakIndices.first : (n ~/ 2);
    if (peakIdx >= n) peakIdx = n - 1;

    final systolicArea = filtered.sublist(0, math.min(peakIdx + 1, n).toInt()).fold<double>(0, (a, b) => a + b.abs());
    final diastolicArea = filtered.sublist(math.min(peakIdx + 1, n).toInt()).fold<double>(0, (a, b) => a + b.abs());

    int troughIdx = peakIdx;
    for (int i = peakIdx + 1; i < math.min(n, peakIdx + (fs / 2).floor()).toInt(); i++) {
      if (filtered[i] < filtered[troughIdx]) troughIdx = i;
    }

    final systolicTime = (peakIdx / fs) * 1000;
    final diastolicTime = ((n - peakIdx) / fs) * 1000;
    final stiffnessIndex = systolicTime > 0 ? (max - min) / (systolicTime / 1000) : 0.0;
    final reflectionIndex = systolicTime > 0 && diastolicTime > 0 ? diastolicTime / systolicTime : 1.0;

    return {
      'pulseWidth': n / fs,
      'peakTime': peakIdx / n,
      'decayTime': (n - peakIdx) / n,
      'inflectionDiff': norm[peakIdx] - norm[0],
      'systolicArea': systolicArea,
      'diastolicArea': diastolicArea,
      'stiffnessIndex': stiffnessIndex,
      'reflectionIndex': reflectionIndex,
    };
  }

  static GlucoseMetrics _estimateGlucose(
      List<double> ppg, List<int> peakIndices, List<double> redSignal, List<double> greenSignal, List<double> blueSignal,
      double fs, double signalQuality) {
    if (ppg.length < fs * 10 || peakIndices.length < 3 || signalQuality < 50) {
      return const GlucoseMetrics(confidence: 0);
    }

    final redPI = filteredPerfusionIndex(redSignal.isNotEmpty ? redSignal : ppg, fs);
    final greenPI = filteredPerfusionIndex(greenSignal.isNotEmpty ? greenSignal : ppg, fs);
    final bluePI = filteredPerfusionIndex(blueSignal.isNotEmpty ? blueSignal : ppg, fs);

    final rRatio = redPI > 0 && greenPI > 0 ? redPI / greenPI : 1.0;
    final gRatio = greenPI > 0 && bluePI > 0 ? greenPI / bluePI : 1.0;

    var baseGlucose = 95 + (rRatio - 1) * 20 + (gRatio - 1) * 15;

    final filtered = applyBandpassFilterCustom(detrend(ppg), cardiacLo, cardiacHi, fs);
    final filteredPeaks = detectPeaks(filtered, fs);
    if (filteredPeaks.length >= 3) {
      final rrIntervals = <double>[];
      for (int i = 1; i < filteredPeaks.length; i++) {
        rrIntervals.add(((filteredPeaks[i].index - filteredPeaks[i - 1].index) / fs) * 1000);
      }
      final cv = rrIntervals.length > 1 ? (stddev(rrIntervals) / mean(rrIntervals)) * 100 : 0.0;
      baseGlucose += (cv - 5) * 0.5;
    }

    final piConfidence = math.min((redPI + greenPI + bluePI) / 6, 1.0).toDouble();
    final confidence = math.min(0.6, 0.2 + piConfidence * 0.4).toDouble();

    return GlucoseMetrics(
      glucose: math.max(70, math.min(200, baseGlucose.roundToDouble())).toDouble(),
      confidence: confidence,
    );
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

class Peak {
  final int index;
  final double value;
  const Peak({required this.index, required this.value});
}

class HRVMetrics {
  final double meanRR;
  final double sdnn;
  final double rmssd;
  final double pnn50;
  final double? lfHfRatio;
  const HRVMetrics({
    required this.meanRR, required this.sdnn, required this.rmssd,
    required this.pnn50, this.lfHfRatio,
  });
}

class HeartMetrics {
  final double? bpm;
  final List<int> peaks;
  final List<double> rrIntervals;
  final HRVMetrics? hrv;
  final String rhythm;
  final double signalQuality;
  const HeartMetrics({
    required this.bpm, required this.peaks, required this.rrIntervals,
    required this.hrv, required this.rhythm, required this.signalQuality,
  });
}

class RespMetrics {
  final double? respRate;
  final String method;
  final double confidence;
  final List<double>? waveform;
  const RespMetrics({
    required this.respRate, required this.method, required this.confidence, this.waveform,
  });
}

class BloodPressureMetrics {
  final double? systolic;
  final double? diastolic;
  final double? map;
  final double? pulsePressure;
  final double confidence;
  const BloodPressureMetrics({
    this.systolic, this.diastolic, this.map, this.pulsePressure, required this.confidence,
  });
}

class GlucoseMetrics {
  final double? glucose;
  final double confidence;
  const GlucoseMetrics({this.glucose, required this.confidence});
}

class CreatinineClass {
  static const String low = 'low';
  static const String normal = 'normal';
  static const String high = 'high';
}

class CreatinineMetrics {
  final double? creatinine;
  final double? creatinineUmol;
  final String? creatinineClass;
  final double? egfr;
  final String? stage;
  final double confidence;
  final String method;
  const CreatinineMetrics({
    this.creatinine, this.creatinineUmol, this.creatinineClass,
    this.egfr, this.stage, required this.confidence, required this.method,
  });
}

class CreatinineFeatures {
  final double peakAmp;
  final double dicroticAmp;
  final double pulseInterval;
  final double rrCv;
  final double augmentationIndex;
  final double stiffnessRatio;
  final double diastolicPeakPos;
  final double upstrokeMidpoint;
  final double riseSlope;
  final double secondDerivTime;
  final double diastolicNotchDuration;
  final double systolicPeakDuration;
  final double inflectionAreaRatio;
  final double firstDerivMinTime;
  final double perfusionIndex;
  final double redGreenRatio;
  final double redBlueRatio;
  final double harmonicRatio;
  final int beats;

  const CreatinineFeatures({
    required this.peakAmp, required this.dicroticAmp, required this.pulseInterval,
    required this.rrCv, required this.augmentationIndex, required this.stiffnessRatio,
    required this.diastolicPeakPos, required this.upstrokeMidpoint, required this.riseSlope,
    required this.secondDerivTime, required this.diastolicNotchDuration,
    required this.systolicPeakDuration, required this.inflectionAreaRatio,
    required this.firstDerivMinTime, required this.perfusionIndex,
    required this.redGreenRatio, required this.redBlueRatio, required this.harmonicRatio,
    required this.beats,
  });

  factory CreatinineFeatures.empty() => const CreatinineFeatures(
    peakAmp: 0, dicroticAmp: 0, pulseInterval: 0, rrCv: 0,
    augmentationIndex: 0, stiffnessRatio: 0, diastolicPeakPos: 0,
    upstrokeMidpoint: 0, riseSlope: 0, secondDerivTime: 0,
    diastolicNotchDuration: 0, systolicPeakDuration: 0,
    inflectionAreaRatio: 0, firstDerivMinTime: 0, perfusionIndex: 0,
    redGreenRatio: 1, redBlueRatio: 1, harmonicRatio: 0, beats: 0,
  );
}

class SpectralSnr {
  final double snr;
  final double signalPower;
  final double noiseDensity;
  final int sigBins;
  final int noiseBins;
  final bool hasReference;
  const SpectralSnr({
    required this.snr, required this.signalPower, required this.noiseDensity,
    required this.sigBins, required this.noiseBins, required this.hasReference,
  });
}

class PowerSpectrumResult {
  final List<double> freqs;
  final List<double> psd;
  const PowerSpectrumResult({required this.freqs, required this.psd});
}

class VitalAnalysis {
  final HeartMetrics heart;
  final RespMetrics resp;
  final double? bpm;
  final double? respRate;
  final HRVMetrics? hrv;
  final double? sdnn;
  final double? rmssd;
  final double? pnn50;
  final String rhythm;
  final BloodPressureMetrics bp;
  final double? bpDeep;
  final GlucoseMetrics glucose;
  final CreatinineMetrics creatinine;

  const VitalAnalysis({
    required this.heart, required this.resp, this.bpm, this.respRate,
    this.hrv, this.sdnn, this.rmssd, this.pnn50, required this.rhythm,
    required this.bp, this.bpDeep, required this.glucose, required this.creatinine,
  });
}

class AnalyzeOptions {
  final String? cameraMode;
  final int? ageYears;
  final String? sex;
  final List<double>? respSource;
  final double? dcRef;

  const AnalyzeOptions({
    this.cameraMode, this.ageYears, this.sex, this.respSource, this.dcRef,
  });
}
