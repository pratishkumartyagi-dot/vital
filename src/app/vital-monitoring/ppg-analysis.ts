export interface Peak {
  index: number;
  value: number;
}

export interface HRVMetrics {
  meanRR: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  lfHfRatio: number | null;
}

export interface HeartMetrics {
  bpm: number | null;
  peaks: number[];
  rrIntervals: number[];
  hrv: HRVMetrics | null;
  rhythm: "regular" | "irregular" | "unknown";
  signalQuality: number;
}

export interface RespMetrics {
  respRate: number | null;
  method: string;
  confidence: number;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.reduce((a, b) => a + (b - m) * (b - m), 0) / (arr.length - 1);
  return Math.sqrt(v);
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return signal.slice();
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = mean(x);
  const yMean = mean(signal);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - xMean) * (signal[i] - yMean);
    den += (x[i] - xMean) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = yMean - slope * xMean;
  return signal.map((y, i) => y - (slope * i + intercept));
}

interface BiquadCoeffs {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

/// Second-order Butterworth lowpass biquad (Q = 1/sqrt(2)) via the RBJ
/// audio-EQ cookbook, which is the exact bilinear-transform design and therefore
/// remains correct at any sampling rate (unlike the previous hardcoded SOS,
/// whose passband was centred near 0.5 Hz and amplified it ~13x).
function lowpassBiquad(fc: number, fs: number, q = Math.SQRT1_2): BiquadCoeffs {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cw) / 2 / a0,
    b1: (1 - cw) / a0,
    b2: (1 - cw) / 2 / a0,
    a1: (-2 * cw) / a0,
    a2: (1 - alpha) / a0,
  };
}

/// Second-order Butterworth highpass biquad (Q = 1/sqrt(2)).
function highpassBiquad(fc: number, fs: number, q = Math.SQRT1_2): BiquadCoeffs {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cw) / 2 / a0,
    b1: -(1 + cw) / a0,
    b2: (1 + cw) / 2 / a0,
    a1: (-2 * cw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function applyBiquad(signal: number[], c: BiquadCoeffs, seedX = 0, seedY = 0): number[] {
  const out: number[] = new Array(signal.length);
  let x1 = seedX, x2 = seedX, y1 = seedY, y2 = seedY;
  for (let i = 0; i < signal.length; i++) {
    const y = c.b0 * signal[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y;
    x2 = x1; x1 = signal[i];
    y2 = y1; y1 = y;
  }
  return out;
}

/// Butterworth bandpass as a cascade of a 2nd-order highpass at `low` and a
/// 2nd-order lowpass at `high`. With `zeroPhase` (default) the cascade is run
/// forward and backward (filtfilt) so pulse peaks and the AC/DC ratios are not
/// shifted by filter phase. The biquad states are seeded with the steady-state
/// response to the signal's DC level, which eliminates the otherwise massive
/// edge transient (a zero-state highpass reacts to the DC step with a spike of
/// nearly the full DC amplitude, inflating every RMS-based AC measurement).
export function butterworthBandpass(
  signal: number[],
  low: number,
  high: number,
  fs: number,
  zeroPhase = true
): number[] {
  if (signal.length < 10 || !(fs > 0) || !(low > 0) || high <= low || high >= fs / 2) {
    return signal.slice();
  }
  const hp = highpassBiquad(low, fs);
  const lp = lowpassBiquad(high, fs);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  // Highpass: steady state for a constant input dc is y = 0. Lowpass input
  // (the highpass output) is zero-mean, so its steady state is x = y = 0.
  let y = applyBiquad(signal, hp, dc, 0);
  y = applyBiquad(y, lp, 0, 0);
  if (zeroPhase) {
    y = y.reverse();
    y = applyBiquad(y, hp, 0, 0);
    y = applyBiquad(y, lp, 0, 0);
    y = y.reverse();
  }
  return y;
}

function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): { peaks: Peak[]; filtered: number[] } {
  const filtered = butterworthBandpass(detrend(signal), 0.5, 3.5, fs, true);
  const peaks: Peak[] = [];
  const minDist = Math.max(1, Math.floor((60 / maxBpm) * fs));
  const sorted = filtered.slice().sort((a, b) => a - b);
  const n = sorted.length;
  const p10 = sorted[Math.floor(n * 0.10)];
  const p90 = sorted[Math.floor(n * 0.90)];
  const p97 = sorted[Math.floor(n * 0.97)];
  // Require peaks to be well above the interquartile baseline, not merely above
  // the 20th percentile. This rejects dicrotic notches and residual noise that
  // otherwise inflate the peak count and destroy HRV.
  const minHeight = Math.max(p10 + 0.5 * (p90 - p10), 0.45 * p97);
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] >= filtered[i + 1]) {
      if (filtered[i] > minHeight) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1].index >= minDist) {
          peaks.push({ index: i, value: filtered[i] });
        } else if (filtered[i] > peaks[peaks.length - 1].value) {
          peaks[peaks.length - 1] = { index: i, value: filtered[i] };
        }
      }
    }
  }
  return { peaks, filtered };
}

/// Plane-Orthogonal-to-Skin (POS) pulse extraction (Wang, den Brinker, Stuijk,
/// de Haan, "Algorithmic Principles of Remote PPG", IEEE TBME 64(7), 2017,
/// Algorithm 1). Temporally mean-normalized RGB is projected onto the two axes
/// S1 = G - B and S2 = -2R + G + B, both orthogonal to the (1,1,1) skin/specular
/// direction and both giving in-phase pulse signals, combined as
/// h = S1 + alpha*S2 with alpha = std(S1)/std(S2). Common-mode illumination and
/// motion artifacts cancel in both projections while the pulse boosts, which
/// far outperforms any single raw channel for front-camera facial rPPG.
/// Windows of ~1 s are Hann-weighted and overlap-added so block edges do not
/// inject discontinuities into the downstream bandpass.
export function posPulse(red: number[], green: number[], blue: number[], winLen = 32): number[] {
  const n = Math.min(red.length, green.length, blue.length);
  if (n < winLen || winLen < 8) return [];
  const hop = Math.max(1, Math.floor(winLen / 2));
  const sqrt6 = Math.sqrt(6);
  const acc = new Array<number>(n).fill(0);
  const wsum = new Array<number>(n).fill(0);
  for (let start = 0; start + winLen <= n; start += hop) {
    let mr = 0;
    let mg = 0;
    let mb = 0;
    for (let i = 0; i < winLen; i++) {
      mr += red[start + i];
      mg += green[start + i];
      mb += blue[start + i];
    }
    mr /= winLen;
    mg /= winLen;
    mb /= winLen;
    if (mr <= 1 || mg <= 1 || mb <= 1) continue;
    const cnR = new Array<number>(winLen);
    const cnG = new Array<number>(winLen);
    const cnB = new Array<number>(winLen);
    for (let i = 0; i < winLen; i++) {
      cnR[i] = red[start + i] / mr;
      cnG[i] = green[start + i] / mg;
      cnB[i] = blue[start + i] / mb;
    }
    const s1 = new Array<number>(winLen);
    const s2 = new Array<number>(winLen);
    let m1 = 0;
    let m2 = 0;
    for (let i = 0; i < winLen; i++) {
      s1[i] = (cnG[i] - cnB[i]) / sqrt6;
      s2[i] = (-2 * cnR[i] + cnG[i] + cnB[i]) / sqrt6;
      m1 += s1[i];
      m2 += s2[i];
    }
    m1 /= winLen;
    m2 /= winLen;
    let v1 = 0;
    let v2 = 0;
    for (let i = 0; i < winLen; i++) {
      v1 += (s1[i] - m1) ** 2;
      v2 += (s2[i] - m2) ** 2;
    }
    const alpha = v2 > 1e-12 ? Math.sqrt(v1 / v2) : 0;
    for (let i = 0; i < winLen; i++) {
      const h = s1[i] - m1 + alpha * (s2[i] - m2);
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (winLen - 1)));
      acc[start + i] += h * w;
      wsum[start + i] += w;
    }
  }
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = wsum[i] > 1e-9 ? acc[i] / wsum[i] : 0;
  }
  return out;
}

function computePerfusionIndex(signal: number[]): number {
  if (signal.length < 10) return 0;
  let m = 0;
  let absMean = 0;
  for (const v of signal) {
    m += v;
    absMean += Math.abs(v);
  }
  m /= signal.length;
  absMean /= signal.length;
  // Camera signals are strictly positive, so |mean| is the DC. Zero-mean
  // derived signals (e.g. the POS pulse) fall back to mean(|x|) so the ratio
  // stays meaningful instead of collapsing to 0.
  const denom = Math.abs(m) > 1e-9 ? m : absMean;
  if (denom <= 1e-9) return 0;
  let ac = 0;
  for (const v of signal) ac += Math.abs(v - m);
  ac /= signal.length;
  return (ac / denom) * 100;
}

/// Perfusion index computed on the bandpass-filtered (0.5-3.5 Hz) pulse:
/// AC is the RMS of the filtered signal and DC the mean of the raw signal.
/// Unlike the raw MAD/mean variant above, baseline wander and respiration do
/// not inflate AC, which is what the creatinine/SpO2-style ratios need.
function filteredPerfusionIndex(signal: number[], fs: number): number {
  if (signal.length < fs * 2) return 0;
  const filtered = butterworthBandpass(signal, 0.5, 3.5, fs, true);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  if (dc <= 1e-9) return 0;
  let sq = 0;
  for (const v of filtered) sq += v * v;
  const ac = Math.sqrt(sq / filtered.length);
  return (ac / dc) * 100;
}

function computeSignalQuality(filtered: number[], raw: number[], peaks: Peak[], fs: number): number {
  if (peaks.length < 2) return 0;

  // 1. Perfusion Index (AC/DC ratio) - normalized to 0-1
  const pi = computePerfusionIndex(raw);
  const piScore = Math.min(1, pi / 1.5); // PI > 1.5% is good for smartphone camera PPG

  // 2. Spectral SNR, measured on the RAW signal.
  const { freqs, psd } = powerSpectrum(raw, fs);
  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) signalPower += psd[i];
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noisePower += psd[i];
  }
  const snr = noisePower > 1e-12 ? signalPower / noisePower : signalPower > 1e-12 ? 8 : 0;
  const snrScore = Math.min(1, snr / 4); // SNR > 4 is good for smartphone PPG

  // 3. Pulse regularity (CV of RR intervals)
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    rrMs.push(((peaks[i].index - peaks[i - 1].index) / fs) * 1000);
  }
  const cv = rrMs.length > 1 ? (stddev(rrMs) / mean(rrMs)) * 100 : 100;
  const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30); // regular rhythm scores 1.0; decreases for CV > 10%

  // 4. Structured SNR-based gating (blueprint for Android cPPG)
  // SNR > 5.0: Clean / Ideal - keep high SQI
  // SNR 3.0-5.0: Minor Noise - apply boost (recovery zone)
  // SNR 1.0-3.0: Highly Unstable - force downgrade (cap below 50)
  // SNR < 1.0: Pure Artifacts - clamp to 0
  let snrGate = 1.0;
  if (snr < 1.0) {
    // Pure artifacts: clamp to 0, bypass all analysis
    snrGate = 0;
  } else if (snr < 2.5) {
    // SNR 1.0-2.5: Highly unstable - force heavy penalty
    // Linear ramp from 0 at SNR=1.0 to 0.4 at SNR=2.5 (max SQI ~40)
    snrGate = clamp01((snr - 1.0) / 1.5) * 0.4;
  } else if (snr < 3.0) {
    // SNR 2.5-3.0: Transition zone - moderate penalty
    // Linear ramp from 0.4 at SNR=2.5 to 0.7 at SNR=3.0 (max SQI ~55-70)
    snrGate = 0.4 + 0.3 * clamp01((snr - 2.5) / 0.5);
  } else if (snr < 5.0) {
    // SNR 3.0-5.0: Minor noise - recovery zone (allow near-full score)
    // Ramp from 0.7 at SNR=3.0 to 1.0 at SNR=5.0
    snrGate = 0.7 + 0.3 * clamp01((snr - 3.0) / 2.0);
  }
  // SNR >= 5.0: snrGate = 1.0 (no penalty)

  // 5. Pulse-presence gate: filtered SNR to reject pure noise that passes raw SNR
  // (e.g., structured noise with cardiac-band energy but no real pulse)
  const { psd: psdFiltered } = powerSpectrum(filtered, fs);
  let sigF = 0, noiseF = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigF += psdFiltered[i];
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseF += psdFiltered[i];
  }
  const snrF = noiseF > 1e-12 ? sigF / noiseF : (sigF > 1e-12 ? 8 : 0);
  const pulseGate = clamp01((snrF - 10) / 20); // zero below snrF=10, full above 30

  // Combined gate: both must pass
  const gate = snrGate * pulseGate;

  // Combined score (weighted average)
  return Math.round((0.45 * piScore + 0.35 * snrScore + 0.2 * regScore) * gate * 100);
}

function computeHRV(rrMs: number[]): HRVMetrics {
  if (rrMs.length < 2) {
    return { meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, lfHfRatio: null };
  }
  const diffs: number[] = [];
  let pnn50 = 0;
  for (let i = 1; i < rrMs.length; i++) {
    const d = Math.abs(rrMs[i] - rrMs[i - 1]);
    diffs.push(d);
    if (d > 50) pnn50++;
  }
  const meanRR = mean(rrMs);
  const sdnn = stddev(rrMs);
  const rmssd = diffs.length ? Math.sqrt(mean(diffs.map((d) => d * d))) : 0;
  const pnn = diffs.length ? (pnn50 / diffs.length) * 100 : 0;
  return { meanRR, sdnn, rmssd, pnn50: pnn, lfHfRatio: null };
}

function classifyRhythm(rrMs: number[]): "regular" | "irregular" | "unknown" {
  if (rrMs.length < 5) return "unknown";
  const sd = stddev(rrMs);
  const m = mean(rrMs);
  if (m === 0) return "unknown";
  const cv = (sd / m) * 100;
  if (cv < 10) return "regular";
  if (cv < 20) return "regular";
  return "irregular";
}

export function powerSpectrum(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  const freqs: number[] = [];
  const psd: number[] = [];
  if (n < 8) return { freqs, psd };
  const detrended = detrend(signal);
  // Hann window: pulse tones rarely sit exactly on a DFT bin, and the unwindowed
  // DFT smeared their energy across many bins (leakage), corrupting the SNR and
  // harmonic-ratio scores. Normalized by mean(w^2) so windowed power stays
  // comparable to the unwindowed scale.
  const win = new Array<number>(n);
  let winSqSum = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    win[i] = w;
    winSqSum += w * w;
  }
  const wNorm = winSqSum / n || 1;
  for (let k = 0; k < Math.floor(n / 2); k++) {
    let real = 0;
    let imag = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      real += detrended[i] * win[i] * Math.cos(angle);
      imag -= detrended[i] * win[i] * Math.sin(angle);
    }
    freqs.push((k * fs) / n);
    psd.push((real * real + imag * imag) / n / wNorm);
  }
  return { freqs, psd };
}

export function analyzeHeartRate(ppg: number[], fs = 30): HeartMetrics {
  if (ppg.length < fs * 5) {
    return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: 0 };
  }
  const { peaks, filtered } = detectPeaks(ppg, fs);
  if (peaks.length < 3) {
    return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: 0 };
  }
  // Accept a peak only if its interval from the last accepted peak is
  // physiologic, so one spurious detection cannot corrupt all later intervals.
  const valid: Peak[] = [];
  for (const p of peaks) {
    if (valid.length === 0) {
      valid.push(p);
      continue;
    }
    const dtMs = ((p.index - valid[valid.length - 1].index) / fs) * 1000;
    if (dtMs >= 300 && dtMs <= 2000) valid.push(p);
  }
  if (valid.length < 3) {
    return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: 0 };
  }
  const rrAll: number[] = [];
  for (let i = 1; i < valid.length; i++) {
    rrAll.push(((valid[i].index - valid[i - 1].index) / fs) * 1000);
  }
  // Reject ectopic / motion intervals (>30% from the median) before HRV, which
  // is what previously produced SDNN/RMSSD values in the thousands of ms.
  const med = median(rrAll);
  const filteredRr = med > 0 ? rrAll.filter((d) => Math.abs(d - med) / med <= 0.3) : rrAll.slice();
  const rrMs = filteredRr.length >= 2 ? filteredRr : rrAll;
  const meanRR = mean(rrMs);
  const bpm = meanRR > 0 ? 60000 / meanRR : 0;
  const hrv = computeHRV(rrMs);
  const rhythm = classifyRhythm(rrMs);
  const quality = computeSignalQuality(filtered, ppg, valid, fs);
  return {
    bpm: bpm > 30 && bpm < 220 ? bpm : null,
    peaks: valid.map((p) => p.index),
    rrIntervals: rrMs,
    hrv,
    rhythm,
    signalQuality: quality,
  };
}

function autocorrelate(signal: number[]): number[] {
  const n = signal.length;
  const m = mean(signal);
  const centered = signal.map((v) => v - m);
  const norm = centered.reduce((a, b) => a + b * b, 0) || 1;
  const out: number[] = [];
  for (let lag = 0; lag < n; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += centered[i] * centered[i + lag];
    out.push(s / norm);
  }
  return out;
}

function findDominantFreqInBand(signal: number[], fs: number, fLow: number, fHigh: number): { freq: number; power: number } {
  const { freqs, psd } = powerSpectrum(signal, fs);
  let bestIdx = -1;
  let bestPower = -Infinity;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= fLow && freqs[i] <= fHigh) {
      if (psd[i] > bestPower) {
        bestPower = psd[i];
        bestIdx = i;
      }
    }
  }
  if (bestIdx < 0) return { freq: 0, power: 0 };
  return { freq: freqs[bestIdx], power: bestPower };
}

export function analyzeRespRate(ppg: number[], fs = 30): RespMetrics {
  if (ppg.length < fs * 10) {
    return { respRate: null, method: "insufficient-data", confidence: 0 };
  }
  const detrended = detrend(ppg);
  const lowpass = butterworthBandpass(detrended, 0.1, 0.5, fs);
  const ac = autocorrelate(lowpass);
  const minLag = Math.floor(fs / 0.5);
  const maxLag = Math.floor(fs / 0.1);
  let bestLag = -1;
  let bestVal = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, ac.length - 1); lag++) {
    if (ac[lag] > bestVal) {
      bestVal = ac[lag];
      bestLag = lag;
    }
  }
  let respBpm = 0;
  let method = "autocorr";
  let confidence = Math.max(0, Math.min(1, bestVal));
  if (bestLag > 0) {
    respBpm = (60 * fs) / bestLag;
  }
  const { freq: fftFreq, power: fftPower } = findDominantFreqInBand(lowpass, fs, 0.1, 0.5);
  if (fftFreq > 0) {
    const fftBpm = fftFreq * 60;
    if (respBpm === 0 || Math.abs(fftBpm - respBpm) < 8) {
      const combined = respBpm === 0 ? fftBpm : 0.5 * (respBpm + fftBpm);
      respBpm = combined;
      method = "autocorr+FFT";
    } else {
      const combined = 0.5 * (respBpm + fftBpm);
      respBpm = combined;
      method = "autocorr+FFT(fused)";
    }
    const totalPower = lowpass.reduce((a, b) => a + b * b, 0) / lowpass.length || 1;
    confidence = Math.max(confidence, Math.min(1, fftPower / totalPower));
  }
  if (respBpm < 5 || respBpm > 50) {
    return { respRate: null, method, confidence: 0 };
  }
  return { respRate: respBpm, method, confidence };
}

export interface PPGFeatures {
  pulseWidth: number;
  peakTime: number;
  decayTime: number;
  inflectionDiff: number;
  systolicArea: number;
  diastolicArea: number;
  stiffnessIndex: number;
  reflectionIndex: number;
}

function extractPPGFeatures(ppg: number[], peakIndices: number[], fs: number): PPGFeatures {
  const n = ppg.length;
  if (n < 30) {
    return { pulseWidth: 0, peakTime: 0, decayTime: 0, inflectionDiff: 0, systolicArea: 0, diastolicArea: 0, stiffnessIndex: 0, reflectionIndex: 0 };
  }

  const filtered = butterworthBandpass(detrend(ppg), 0.5, 3.5, fs);
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  const range = max - min || 1;
  const norm = filtered.map(v => (v - min) / range);

  let peakIdx = peakIndices.length > 0 ? peakIndices[0] : Math.floor(n / 2);
  if (peakIdx >= n) peakIdx = n - 1;

  const systolicArea = filtered.slice(0, peakIdx + 1).reduce((a, b) => a + Math.abs(b - min), 0);
  const diastolicArea = filtered.slice(peakIdx + 1).reduce((a, b) => a + Math.abs(b - min), 0);

  let troughIdx = peakIdx;
  for (let i = peakIdx + 1; i < Math.min(n, peakIdx + Math.floor(fs / 2)); i++) {
    if (filtered[i] < filtered[troughIdx]) troughIdx = i;
  }

  const systolicTime = (peakIdx / fs) * 1000;
  const diastolicTime = ((n - peakIdx) / fs) * 1000;
  const stiffnessIndex = systolicTime > 0 ? (max - min) / (systolicTime / 1000) : 0;
  const reflectionIndex = diastolicTime > 0 && systolicTime > 0 ? diastolicTime / systolicTime : 1;

  return {
    pulseWidth: n / fs,
    peakTime: peakIdx / n,
    decayTime: (n - peakIdx) / n,
    inflectionDiff: norm[peakIdx] - norm[0],
    systolicArea,
    diastolicArea,
    stiffnessIndex,
    reflectionIndex,
  };
}

export interface BloodPressureMetrics {
  systolic: number | null;
  diastolic: number | null;
  map: number | null;
  pulsePressure: number | null;
  confidence: number;
}

export interface GlucoseMetrics {
  glucose: number | null;
  confidence: number;
}

function estimateBloodPressure(
  ppg: number[],
  peakIndices: number[],
  rrMs: number[],
  hrv: HRVMetrics | null,
  fs: number
): BloodPressureMetrics {
  if (ppg.length < fs * 5 || peakIndices.length < 2) {
    return { systolic: null, diastolic: null, map: null, pulsePressure: null, confidence: 0 };
  }

  const features = extractPPGFeatures(ppg, peakIndices, fs);
  const heartRate = rrMs.length > 0 ? 60000 / mean(rrMs) : 70;

  const stiffnessFactor = Math.min(features.stiffnessIndex / 100, 1);
  const reflectionFactor = Math.min(features.reflectionIndex / 2, 1);

  let baseSystolic = 110 + (heartRate - 70) * 0.3;
  let baseDiastolic = 70 + (heartRate - 70) * 0.1;

  const sysAdjust = (stiffnessFactor * 15) - (reflectionFactor * 10) + (hrv?.sdnn ?? 0) * 0.1;
  const diaAdjust = (stiffnessFactor * 5) - (reflectionFactor * 5) + (hrv?.rmssd ?? 0) * 0.05;

  baseSystolic += sysAdjust;
  baseDiastolic += diaAdjust;

  const pulsePressure = baseSystolic - baseDiastolic;
  const map = baseDiastolic + pulsePressure / 3;

  const confidence = Math.min(0.7, 0.3 + (stiffnessFactor * 0.2) + (reflectionFactor * 0.2));

  return {
    systolic: Math.round(Math.max(80, Math.min(200, baseSystolic))),
    diastolic: Math.round(Math.max(50, Math.min(130, baseDiastolic))),
    map: Math.round(Math.max(60, Math.min(150, map))),
    pulsePressure: Math.round(Math.max(20, Math.min(100, pulsePressure))),
    confidence,
  };
}

function estimateGlucose(
  ppg: number[],
  peakIndices: number[],
  redSignal: number[],
  greenSignal: number[],
  blueSignal: number[],
  fs: number,
  signalQuality: number
): GlucoseMetrics {
  // Blood glucose is NOT directly measurable from RGB camera PPG (no glucose
  // absorption band in the visible spectrum); this is a weak research surrogate
  // at best. Gate it hard on signal quality so weak/noisy signals never
  // produce a number instead of a misleading artifact.
  if (ppg.length < fs * 10 || peakIndices.length < 3 || signalQuality < 50) {
    return { glucose: null, confidence: 0 };
  }

  const redPI = computePerfusionIndex(redSignal.length > 0 ? redSignal : ppg);
  const greenPI = computePerfusionIndex(greenSignal.length > 0 ? greenSignal : ppg);
  const bluePI = computePerfusionIndex(blueSignal.length > 0 ? blueSignal : ppg);

  const rRatio = redPI > 0 && greenPI > 0 ? redPI / greenPI : 1;
  const gRatio = greenPI > 0 && bluePI > 0 ? greenPI / bluePI : 1;

  let baseGlucose = 95 + (rRatio - 1) * 20 + (gRatio - 1) * 15;

  const filtered = butterworthBandpass(detrend(ppg), 0.5, 3.5, fs);
  const { peaks: filteredPeaks } = detectPeaks(filtered, fs);
  if (filteredPeaks.length >= 3) {
    const rrIntervals: number[] = [];
    for (let i = 1; i < filteredPeaks.length; i++) {
      rrIntervals.push(((filteredPeaks[i].index - filteredPeaks[i - 1].index) / fs) * 1000);
    }
    const cv = rrIntervals.length > 1 ? (stddev(rrIntervals) / mean(rrIntervals)) * 100 : 0;
    baseGlucose += (cv - 5) * 0.5;
  }

  const piConfidence = Math.min((redPI + greenPI + bluePI) / 6, 1);
  const confidence = Math.min(0.6, 0.2 + piConfidence * 0.4);

  return {
    glucose: Math.max(70, Math.min(200, Math.round(baseGlucose))),
    confidence,
  };
}

export type CreatinineClass = "low" | "normal" | "high";

export interface CreatinineMetrics {
  creatinine: number | null;
  creatinineUmol: number | null;
  creatinineClass: CreatinineClass | null;
  egfr: number | null;
  stage: string | null;
  confidence: number;
  method: "ppg-rear-ftir" | "rppg-front";
}

export interface CreatinineFeatures {
  peakAmp: number;
  dicroticAmp: number;
  pulseInterval: number;
  rrCv: number;
  augmentationIndex: number;
  stiffnessRatio: number;
  diastolicPeakPos: number;
  upstrokeMidpoint: number;
  riseSlope: number;
  secondDerivTime: number;
  diastolicNotchDuration: number;
  systolicPeakDuration: number;
  inflectionAreaRatio: number;
  firstDerivMinTime: number;
  perfusionIndex: number;
  redGreenRatio: number;
  redBlueRatio: number;
  harmonicRatio: number;
  beats: number;
}

const EMPTY_CREATININE_FEATURES: CreatinineFeatures = {
  peakAmp: 0,
  dicroticAmp: 0,
  pulseInterval: 0,
  rrCv: 0,
  augmentationIndex: 0,
  stiffnessRatio: 0,
  diastolicPeakPos: 0,
  upstrokeMidpoint: 0,
  riseSlope: 0,
  secondDerivTime: 0,
  diastolicNotchDuration: 0,
  systolicPeakDuration: 0,
  inflectionAreaRatio: 0,
  firstDerivMinTime: 0,
  perfusionIndex: 0,
  redGreenRatio: 1,
  redBlueRatio: 1,
  harmonicRatio: 0,
  beats: 0,
};

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

function normalizeRange(v: number, lo: number, hi: number): number {
  if (hi === lo) return 0;
  return clamp01((v - lo) / (hi - lo));
}

/// Extracts the 45-feature PPG morphology family used by both Sridevi et al.
/// creatinine studies:
///  - "Machine Learning Based Noninvasive Prediction of Creatinine from
///    Fingertip Videos" (Smart Health, 2025) - rear-camera fingertip PPG
///    regression, RF MAE ~0.10 mg/dL, selected F0-F5/F10/F11/F13/F22.
///  - "An integrated machine learning and hyperparameter optimization
///    framework for noninvasive creatinine estimation using photoplethysmography
///    signals" (Healthcare Analytics 7, 2025, 100395) - MIMIC-III ICU PPG,
///    XGBoost+Optuna 3-class low/normal/high (85.2% acc), selected
///    F3/F8/F11/F13/F20/F22/F25.
/// This extractor keeps the intersection/union of the selected morphology
/// descriptors plus perfusion and harmonic terms.
export function extractCreatinineFeatures(
  ppg: number[],
  red: number[],
  green: number[],
  blue: number[],
  peakIndices: number[],
  fs: number
): CreatinineFeatures {
  if (ppg.length < fs * 5 || peakIndices.length < 3) {
    return { ...EMPTY_CREATININE_FEATURES };
  }

  const filtered = butterworthBandpass(detrend(ppg), 0.5, 3.5, fs);
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  const range = max - min || 1;
  const norm = filtered.map((v) => (v - min) / range);

  const intervals: number[] = [];
  for (let i = 1; i < peakIndices.length; i++) {
    const len = peakIndices[i] - peakIndices[i - 1];
    if (len >= fs * 0.3 && len <= fs * 2) intervals.push(len);
  }
  if (intervals.length < 2) return { ...EMPTY_CREATININE_FEATURES };

  const cycleLen = Math.round(mean(intervals));
  if (cycleLen < 4) return { ...EMPTY_CREATININE_FEATURES };

  const cycles: number[][] = [];
  for (let i = 1; i < peakIndices.length; i++) {
    const start = peakIndices[i - 1];
    if (start + cycleLen <= norm.length) cycles.push(norm.slice(start, start + cycleLen));
  }
  if (cycles.length < 2) return { ...EMPTY_CREATININE_FEATURES };

  const avgCycle = new Array<number>(cycleLen).fill(0);
  for (const c of cycles) {
    for (let j = 0; j < cycleLen; j++) avgCycle[j] += c[j];
  }
  for (let j = 0; j < cycleLen; j++) avgCycle[j] /= cycles.length;

  let sysIdx = 0;
  for (let j = 0; j < cycleLen; j++) if (avgCycle[j] > avgCycle[sysIdx]) sysIdx = j;
  const peakAmp = avgCycle[sysIdx];

  let notchIdx = sysIdx;
  const notchSearchEnd = Math.min(cycleLen, sysIdx + Math.floor(fs * 0.5));
  for (let j = sysIdx + 1; j < notchSearchEnd; j++) {
    if (avgCycle[j] < avgCycle[notchIdx]) notchIdx = j;
  }
  const dicroticAmp = avgCycle[notchIdx];

  let diasIdx = notchIdx;
  for (let j = notchIdx + 1; j < cycleLen; j++) {
    if (avgCycle[j] > avgCycle[diasIdx]) diasIdx = j;
  }

  const augmentationIndex = peakAmp > 1e-6 ? avgCycle[diasIdx] / peakAmp : 0;
  const stiffnessRatio = peakAmp > 1e-6 ? (peakAmp - dicroticAmp) / peakAmp : 0;
  const riseSpan = Math.max(1, sysIdx);
  const riseSlope = (avgCycle[sysIdx] - avgCycle[0]) / riseSpan;

  const midLevel = (avgCycle[0] + avgCycle[sysIdx]) / 2;
  let midIdx = 0;
  for (let j = 1; j <= sysIdx; j++) {
    if (avgCycle[j] >= midLevel) {
      midIdx = j;
      break;
    }
  }
  const upstrokeMidpoint = midIdx / cycleLen;
  const diastolicPeakPos = diasIdx / cycleLen;

  const d1: number[] = [];
  for (let j = 1; j < cycleLen; j++) d1.push(avgCycle[j] - avgCycle[j - 1]);
  let firstMax = -1;
  let secondMax = -1;
  for (let j = 1; j < d1.length - 1; j++) {
    if (d1[j] > d1[j - 1] && d1[j] > d1[j + 1]) {
      if (firstMax < 0) firstMax = j;
      else {
        secondMax = j;
        break;
      }
    }
  }
  const secondDerivTime = secondMax > 0 ? secondMax / cycleLen : 0;

  let firstMin = -1;
  for (let j = 1; j < d1.length - 1; j++) {
    if (d1[j] < d1[j - 1] && d1[j] < d1[j + 1]) {
      firstMin = j;
      break;
    }
  }
  const firstDerivMinTime = firstMin > 0 ? firstMin / cycleLen : 0;

  const diastolicNotchDuration = Math.max(0, (diasIdx - notchIdx) / cycleLen);
  const systolicPeakDuration = sysIdx / cycleLen;

  let totalArea = 0;
  const cumulative: number[] = [0];
  for (let j = 1; j < cycleLen; j++) {
    const area = (avgCycle[j] + avgCycle[j - 1]) / 2;
    totalArea += area;
    cumulative.push(totalArea);
  }
  const inflectionIdx = firstMax > 0 ? firstMax : sysIdx;
  const inflectionAreaRatio = totalArea > 1e-9 ? cumulative[Math.min(inflectionIdx, cumulative.length - 1)] / totalArea : 0;

  const pulseInterval = mean(intervals) / fs;
  const rrMs = intervals.map((v) => (v / fs) * 1000);
  const rrCv = mean(rrMs) > 0 ? stddev(rrMs) / mean(rrMs) : 0;

  // Per-channel perfusion from the bandpass-filtered pulse (RMS AC over DC).
  // The previous raw MAD/mean PI was dominated by baseline wander and used a
  // nonsensical `redPI || greenPI` fallback for the primary term.
  const ppgPI = filteredPerfusionIndex(ppg, fs);
  const redPI = filteredPerfusionIndex(red.length > 0 ? red : ppg, fs);
  const greenPI = filteredPerfusionIndex(green.length > 0 ? green : ppg, fs);
  const bluePI = filteredPerfusionIndex(blue.length > 0 ? blue : ppg, fs);
  const perfusionIndex = ppgPI || redPI || greenPI;
  const redGreenRatio = greenPI > 0 ? redPI / greenPI : 1;
  const redBlueRatio = bluePI > 0 ? redPI / bluePI : 1;

  const { psd } = powerSpectrum(avgCycle, fs);
  const harmonicRatio = psd[1] ? (psd[2] + psd[3]) / psd[1] : 0;

  return {
    peakAmp,
    dicroticAmp,
    pulseInterval,
    rrCv,
    augmentationIndex,
    stiffnessRatio,
    diastolicPeakPos,
    upstrokeMidpoint,
    riseSlope,
    secondDerivTime,
    diastolicNotchDuration,
    systolicPeakDuration,
    inflectionAreaRatio,
    firstDerivMinTime,
    perfusionIndex,
    redGreenRatio,
    redBlueRatio,
    harmonicRatio,
    beats: cycles.length,
  };
}

/// CKD-EPI 2021 (race-free) estimated glomerular filtration rate.
export function estimateEgfr(
  creatinineMgDl: number | null,
  ageYears: number | null,
  sex: "male" | "female" | null
): number | null {
  if (creatinineMgDl === null || creatinineMgDl <= 0 || !Number.isFinite(creatinineMgDl)) return null;
  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const female = sex === "female";
  const k = female ? 0.7 : 0.9;
  const a = female ? -0.241 : -0.302;
  const ratio = creatinineMgDl / k;
  const egfr =
    142 *
    Math.pow(Math.min(ratio, 1), a) *
    Math.pow(Math.max(ratio, 1), -1.2) *
    Math.pow(0.9938, age) *
    (female ? 1.012 : 1);
  return egfr;
}

function kdigoStage(egfr: number | null): string | null {
  if (egfr === null) return null;
  if (egfr >= 90) return "G1";
  if (egfr >= 60) return "G2";
  if (egfr >= 45) return "G3a";
  if (egfr >= 30) return "G3b";
  if (egfr >= 15) return "G4";
  return "G5";
}

/// Non-invasive serum creatinine estimate. The regression is a transparent
/// surrogate for the Random Forest model reported by Sridevi et al. (2025),
/// which achieved MAE ~0.10 mg/dL on 213 subjects with an 850/910 nm NIR
/// finger rig. That model requires the paper's trained weights and NIR
/// excitation; this implementation approximates the same feature->creatinine
/// relationship from RGB smartphone PPG/rPPG and is NOT a clinical assay.
/// Design notes:
///  - Age is intentionally NOT a predictor term. The papers predict
///    creatinine from PPG waveform morphology only; vascular stiffness and
///    augmentation descriptors already carry the age signal, and age re-enters
///    downstream through CKD-EPI eGFR. An explicit age term double-counted
///    aging and biased eGFR artificially low for older subjects.
///  - Front facial rPPG has far weaker pulsatile amplitudes than rear
///    fingertip PPG, so instead of halving every morphology term (which
///    collapsed the front-mode estimate onto its demographic base value) the
///    descriptor normalization bands are mode-specific and the signal terms
///    keep full weight; the wider front-mode uncertainty is expressed through
///    the confidence cap.
///  - The estimate is gated on the composite SQI: below 40 the waveform
///    morphology is too unreliable to interpret, so no number is produced.
function estimateCreatinine(
  ppg: number[],
  red: number[],
  green: number[],
  blue: number[],
  peakIndices: number[],
  bpm: number | null,
  signalQuality: number,
  fs: number,
  cameraMode: "front" | "rear",
  ageYears: number | null,
  sex: "male" | "female" | null
): CreatinineMetrics {
  const method: CreatinineMetrics["method"] = cameraMode === "front" ? "rppg-front" : "ppg-rear-ftir";
  const features = extractCreatinineFeatures(ppg, red, green, blue, peakIndices, fs);
  const invalid: CreatinineMetrics = {
    creatinine: null,
    creatinineUmol: null,
    creatinineClass: null,
    egfr: null,
    stage: null,
    confidence: 0,
    method,
  };
  if (ppg.length < fs * 10 || features.beats < 3 || bpm === null || signalQuality < 40) return invalid;

  const isMale = sex === "male";
  const isFront = cameraMode === "front";

  const amplitude = normalizeRange(features.peakAmp, 0.05, 0.95);
  const perfusion = isFront
    ? normalizeRange(features.perfusionIndex, 0.02, 0.8)
    : normalizeRange(features.perfusionIndex, 0.15, 3);
  const redGreen = normalizeRange(features.redGreenRatio, 0.6, 1.8);
  const stiffness = normalizeRange(features.stiffnessRatio, 0.05, 0.9);
  const augmentation = normalizeRange(features.augmentationIndex, 0.1, 1.2);
  const harmonic = normalizeRange(features.harmonicRatio, 0.05, 1.5);
  const rrCv = normalizeRange(features.rrCv, 0, 0.3);
  const notchTime = normalizeRange(features.diastolicNotchDuration, 0.02, 0.5);
  const systolicTime = normalizeRange(features.systolicPeakDuration, 0.05, 0.5);
  const ipa = normalizeRange(features.inflectionAreaRatio, 0.1, 0.7);
  const derivMin = normalizeRange(features.firstDerivMinTime, 0.05, 0.6);

  // Beer-Lambert intuition: weaker reflected pulsatile amplitude / perfusion at
  // a fixed blood volume is associated with higher chromophore concentration.
  // The selected-descriptor terms mirror the F3/F8/F11/F13/F20/F22/F25 feature
  // importances reported for the MIMIC-III XGBoost+Optuna model.
  let creatinine =
    0.85 +
    (isMale ? 0.15 : 0) -
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

  creatinine = Math.min(12, Math.max(0.4, creatinine));
  creatinine = Math.round(creatinine * 100) / 100;

  // Encoding used by Sridevi et al. (Healthcare Analytics 2025): normal band is
  // 0.6-1.3 mg/dL, above is high, below is low.
  const creatinineClass: CreatinineClass =
    creatinine > 1.3 ? "high" : creatinine < 0.6 ? "low" : "normal";

  const sqNorm = clamp01(signalQuality / 100);
  const beatAdequacy = clamp01(features.beats / 15);
  const confidence = Math.min(isFront ? 0.55 : 0.8, 0.12 + 0.55 * sqNorm + 0.15 * beatAdequacy);

  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const egfr = estimateEgfr(creatinine, age, sex);

  return {
    creatinine,
    creatinineUmol: Math.round(creatinine * 88.4 * 10) / 10,
    creatinineClass,
    egfr: egfr === null ? null : Math.round(egfr),
    stage: kdigoStage(egfr),
    confidence: Math.round(confidence * 100) / 100,
    method,
  };
}

export interface VitalAnalysis {
  heart: HeartMetrics;
  resp: RespMetrics;
  bpm: number | null;
  respRate: number | null;
  hrv: HRVMetrics | null;
  sdnn: number | null;
  rmssd: number | null;
  pnn50: number | null;
  rhythm: "regular" | "irregular" | "unknown";
  bp: BloodPressureMetrics;
  glucose: GlucoseMetrics;
  creatinine: CreatinineMetrics;
}

export interface AnalyzeOptions {
  cameraMode?: "front" | "rear";
  ageYears?: number | null;
  sex?: "male" | "female" | null;
  respSource?: number[];
}

export function analyzeVitals(
  ppg: number[],
  fs = 30,
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = [],
  options: AnalyzeOptions = {}
): VitalAnalysis {
  const heart = analyzeHeartRate(ppg, fs);
  const resp = analyzeRespRate(options.respSource ?? ppg, fs);
  const bp = estimateBloodPressure(ppg, heart.peaks, heart.rrIntervals, heart.hrv, fs);
  const glucose = estimateGlucose(ppg, heart.peaks, redSignal, greenSignal, blueSignal, fs, heart.signalQuality);
  const creatinine = estimateCreatinine(
    ppg,
    redSignal,
    greenSignal,
    blueSignal,
    heart.peaks,
    heart.bpm,
    heart.signalQuality,
    fs,
    options.cameraMode ?? "rear",
    options.ageYears ?? null,
    options.sex ?? null
  );

  return {
    heart,
    resp,
    bpm: heart.bpm,
    respRate: resp.respRate,
    hrv: heart.hrv,
    sdnn: heart.hrv?.sdnn ?? null,
    rmssd: heart.hrv?.rmssd ?? null,
    pnn50: heart.hrv?.pnn50 ?? null,
    rhythm: heart.rhythm,
    bp,
    glucose,
    creatinine,
  };
}
