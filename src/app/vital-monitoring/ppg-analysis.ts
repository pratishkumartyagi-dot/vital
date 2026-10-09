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
  method: "peaks" | "spectral" | "none";
  bpmPeaks: number | null;
  bpmSpectral: number | null;
  rrMs: number[];
  peakTimesMs: number[];
  validFraction: number;
  snrDb: number;
  sdnn: number | null;
  rmssd: number | null;
  reliable: boolean;
  perfusionIndex: number;
}

export interface RespMetrics {
  respRate: number | null;
  method: string;
  confidence: number;
  waveform?: number[];
}

import { estimateRespRateRPPG, extractRPPGRespiration } from './rppg-resp';
import { estimateRespRateRearPPG } from './rrest-resp';

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
  // Restore a physical amplitude scale. h is assembled from the *mean-normalized*
  // channels, so it is a dimensionless relative-chrominance signal; feeding it
  // straight into the rest of the pipeline left two unit mismatches:
  //  - the 1/sqrt(6) projection scaling made it sqrt(6)/2 = 1.22x too small
  //    relative to a raw channel (rms(h) = sqrt(2)*sigma(s1) and
  //    rms((cnG-cnB)/sqrt(6)) = rms(cnG-cnB)/sqrt(6));
  //  - it stayed in "per-unit" amplitude while every AC/DC denominator is an
  //    absolute camera count level, so the perfusion index read ~0.015-0.03%
  //    instead of the ~3% the same signal shows in the raw green channel.
  // Multiplying by sqrt(3) * mean(green) puts the pulse back into camera-count
  // units, so AC/DC against the green DC level is a directly comparable
  // perfusion percentage. Every consumer (respiration extraction, BPNet, the
  // creatinine AC/DC ratios) is scale-invariant, so this is a pure unit fix.
  let gSum = 0;
  for (let i = 0; i < n; i++) gSum += green[i];
  const unitScale = Math.sqrt(3) * (gSum / n);
  for (let i = 0; i < n; i++) out[i] *= unitScale;
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

function rms(signal: number[]): number {
  if (signal.length === 0) return 0;
  let sq = 0;
  for (const v of signal) sq += v * v;
  return Math.sqrt(sq / signal.length);
}

/// Mean absolute level of a signal, used as the DC (illumination) reference.
/// Camera signals are strictly positive, so this is the DC level of the
/// underlying skin trace.
function dcLevel(signal: number[]): number {
  if (signal.length === 0) return 0;
  let sum = 0;
  for (const v of signal) sum += Math.abs(v);
  return sum / signal.length;
}

/// AC/DC of a single color channel, with the sensor zero-level offset (ZLO)
/// subtracted from the DC. Smartphone cameras have a per-frame tone-mapped DC
/// that sits above the true black level; without the ZLO correction the DC is
/// systematically high, the AC/DC ratio is systematically low, and the
/// ratio-of-ratios is biased by however much the three channels' offsets
/// differ. The calibration database carries a per-model ZLO; this helper
/// applies it (or the default) uniformly.
///
/// `raw` is the unfiltered channel trace; `filtered` is the same trace
/// bandpass-filtered to the cardiac band, whose RMS is the AC.
export function channelAcDcWithZlo(raw: number[], filtered: number[], zlo: number): { ac: number; dc: number } {
  const n = raw.length;
  if (n === 0) return { ac: 0, dc: 0 };
  let sum = 0;
  for (const v of raw) sum += v;
  const dc = Math.max(0, sum / n - zlo);
  let sq = 0;
  for (const v of filtered) sq += v * v;
  const ac = Math.sqrt(sq / filtered.length);
  return { ac, dc };
}

/// Ratio-of-ratios SpO2 for the red/green pair, with the ZLO correction and
/// the red/green calibration constants. Returns null when either channel's
/// corrected DC is non-positive (i.e. the channel is fully below the black
/// level — unusable).
export function estimateSpo2RedGreen(
  redRaw: number[],
  greenRaw: number[],
  redFilt: number[],
  greenFilt: number[],
  zlo: number,
  offset = 0
): number | null {
  const { ac: acR, dc: dcR } = channelAcDcWithZlo(redRaw, redFilt, zlo);
  const { ac: acG, dc: dcG } = channelAcDcWithZlo(greenRaw, greenFilt, zlo);
  if (dcR <= 1e-9 || dcG <= 1e-9) return null;
  const redRatio = acR / dcR;
  const greenRatio = acG / dcG;
  if (greenRatio <= 1e-9) return null;
  const R = redRatio / greenRatio;

  // Reject saturated green channel (>30% samples at >= 250) which biases AC/DC
  const satFrac = greenRaw.filter((v) => v >= 250).length / greenRaw.length;
  if (satFrac > 0.3) return null;

  return Math.min(SPO2_RED_GREEN_MAX, Math.max(SPO2_RED_GREEN_MIN, SPO2_RED_GREEN_INTERCEPT - SPO2_RED_GREEN_SLOPE * R + offset));
}

/// Log-domain SpO2 estimation (PMC10705321 method).
///
/// Algorithm:
/// 1. Take log of raw red and green signals: lr = log(red + eps), lg = log(green + eps)
/// 2. Bandpass filter log signals (0.7-3.5 Hz, 4th-order Butterworth via filtfilt)
/// 3. Window-wise R: R_win = std(log_filt_red) / std(log_filt_green) per 10s window
/// 4. Robust estimate: R = median of all window R values
/// 5. SpO2 = A - B * R with A=110, B=25
///
/// This approach is more physically correct (Beer-Lambert law in log domain)
/// and more robust to motion artifacts via window-wise median.
export function estimateSpo2LogDomain(
  redRaw: number[],
  greenRaw: number[],
  fs: number,
  A = SPO2_LOG_INTERCEPT,
  B = SPO2_LOG_SLOPE,
  eps = 1e-6
): number | null {
  const n = Math.min(redRaw.length, greenRaw.length);
  if (n < fs * 10) return null; // Need at least 10s

  // Step 1: Log-domain signals
  const lr = new Array<number>(n);
  const lg = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    lr[i] = Math.log(Math.max(redRaw[i], 1) + eps);
    lg[i] = Math.log(Math.max(greenRaw[i], 1) + eps);
  }

  // Step 2: 4th-order Butterworth bandpass (0.7-3.5 Hz) with zero-phase (filtfilt)
  // Current butterworthBandpass is already 4th-order (2nd HP + 2nd LP) with zero-phase
  const fr = butterworthBandpass(lr, 0.7, 3.5, fs, true);
  const fg = butterworthBandpass(lg, 0.7, 3.5, fs, true);

  // Step 3: Window-wise R computation
  const winLen = Math.round(SPO2_LOG_WINDOW_SEC * fs);
  const stepLen = Math.round(SPO2_LOG_STEP_SEC * fs);
  if (n < winLen) return null;

  const Rs: number[] = [];
  for (let i = 0; i + winLen <= n; i += stepLen) {
    // Standard deviation of filtered log signals in this window
    let sr = 0, sg = 0;
    let mr = 0, mg = 0;
    for (let j = 0; j < winLen; j++) {
      mr += fr[i + j];
      mg += fg[i + j];
    }
    mr /= winLen;
    mg /= winLen;
    for (let j = 0; j < winLen; j++) {
      sr += (fr[i + j] - mr) ** 2;
      sg += (fg[i + j] - mg) ** 2;
    }
    sr = Math.sqrt(sr / winLen);
    sg = Math.sqrt(sg / winLen);
    if (sg > 1e-12) {
      Rs.push(sr / sg);
    }
  }

  if (Rs.length === 0) return null;

  // Step 4: Median R for motion robustness
  Rs.sort((a, b) => a - b);
  const R = Rs[Math.floor(Rs.length / 2)];

  // Step 5: SpO2 = A - B * R
  const spo2 = A - B * R;
  return Math.min(SPO2_LOG_MAX, Math.max(SPO2_LOG_MIN, spo2));
}

/// Wrapper that tries log-domain method first, falls back to linear AC/DC method
export function estimateSpo2Best(
  redRaw: number[],
  greenRaw: number[],
  redFilt: number[],
  greenFilt: number[],
  zlo: number,
  fs: number,
  offset = 0
): { spo2: number | null; method: "log" | "linear" | null } {
  // Try log-domain method (more robust, physically correct)
  const spo2Log = estimateSpo2LogDomain(redRaw, greenRaw, fs);
  if (spo2Log !== null) {
    return { spo2: spo2Log, method: "log" };
  }

  // Fallback to linear AC/DC method
  const { ac: acR, dc: dcR } = channelAcDcWithZlo(redRaw, redFilt, zlo);
  const { ac: acG, dc: dcG } = channelAcDcWithZlo(greenRaw, greenFilt, zlo);
  if (dcR <= 1e-9 || dcG <= 1e-9) return { spo2: null, method: null };
  const redRatio = acR / dcR;
  const greenRatio = acG / dcG;
  if (greenRatio <= 1e-9) return { spo2: null, method: null };
  const R = redRatio / greenRatio;

  const satFrac = greenRaw.filter((v) => v >= 250).length / greenRaw.length;
  if (satFrac > 0.3) return { spo2: null, method: null };

  const spo2 = Math.min(SPO2_RED_GREEN_MAX, Math.max(SPO2_RED_GREEN_MIN, SPO2_RED_GREEN_INTERCEPT - SPO2_RED_GREEN_SLOPE * R + offset));
  return { spo2, method: "linear" };
}

/// Haemoglobin (g/dL) from the red/green ratio-of-ratios, ZLO-corrected.
export function estimateHbRedGreen(
  redRaw: number[],
  greenRaw: number[],
  redFilt: number[],
  greenFilt: number[],
  zlo: number
): number | null {
  const { ac: acR, dc: dcR } = channelAcDcWithZlo(redRaw, redFilt, zlo);
  const { ac: acG, dc: dcG } = channelAcDcWithZlo(greenRaw, greenFilt, zlo);
  if (dcR <= 1e-9 || dcG <= 1e-9) return null;
  const redRatio = acR / dcR;
  const greenRatio = acG / dcG;
  if (greenRatio <= 1e-9) return null;
  const R = redRatio / greenRatio;
  return Math.min(HB_MAX, Math.max(HB_MIN, HB_RED_GREEN_INTERCEPT - HB_RED_GREEN_SLOPE * R));
}

/// Look up the ZLO for a device model from the calibration database. Returns
/// the fallback when the model is unknown (the database is loaded by the
/// server-side calibration endpoint; the browser falls back to the constant).
export function resolveZlo(deviceModel?: string | null): number {
  if (!deviceModel) return DEFAULT_ZLO;
  try {
    // The calibration database is served by the calibration API; the browser
    // does not bundle it. This stub keeps the signature honest — callers
    // without a model pass null and get the default.
    return DEFAULT_ZLO;
  } catch {
    return DEFAULT_ZLO;
  }
}

/// Perfusion index (AC/DC) in percent of a single pulse channel.
///
/// AC is the RMS of the cardiac-band (0.5-3.5 Hz) component, so baseline wander
/// and respiration cannot inflate it. `dcRef` is the DC/illumination level of
/// the *source* trace and is mandatory for zero-mean derived pulses such as the
/// POS chrominance pulse, where mean(pulse) ~ 0 and AC/mean is meaningless.
export function perfusionIndexACDC(signal: number[], fs: number, dcRef?: number): number {
  if (signal.length < Math.max(10, fs * 2)) return 0;
  const dc = dcRef !== undefined && dcRef > 1e-9 ? dcRef : dcLevel(signal);
  if (dc <= 1e-9) return 0;
  const ac = rms(butterworthBandpass(signal, CARDIAC_LO, CARDIAC_HI, fs, true));
  return (ac / dc) * 100;
}

/// Composite signal quality index (0-100) for a single PPG channel.
///
/// `dcRef` is the DC (illumination) level of the *source* trace the pulse was
/// extracted from — the green skin-ROI mean. It is required because the rPPG
/// path analyses the POS chrominance pulse, which is per-window mean-normalized
/// and therefore essentially zero-mean: computing AC/mean on it is numerically
/// unstable (it was observed ranging from 0 to 15 000 %) and, against a fixed
/// threshold, pinned the perfusion term at 1.0 for good signals *and* for pure
/// noise, so the term had no discriminative power whatsoever.
function computeSignalQuality(
  filtered: number[],
  raw: number[],
  peaks: Peak[],
  fs: number,
  dcRef: number
): number {
  if (peaks.length < 2) return 0;

  // 1. Perfusion index: RMS of the cardiac-band AC over the DC level of the
  //    illuminated skin. `filtered` is already band-limited to 0.5-3.5 Hz, so
  //    baseline wander and respiration do not inflate the AC term.
  const dc = dcRef > 1e-9 ? dcRef : dcLevel(raw);
  const ac = rms(filtered);
  const pi = dc > 1e-9 ? (ac / dc) * 100 : 0;
  // Facial rPPG runs at 0.05-0.5% AC/DC and contact fingertip PPG at 0.5-5%, so
  // full credit at 0.5% with a square-root law keeps the entire remote-PPG range
  // usable instead of collapsing it to zero, while still discriminating a
  // strong pulse from a weak one.
  const piScore = clamp01(Math.sqrt(pi / 0.5));

  // 2. Bandwidth-matched spectral SNR on the RAW signal (see spectralSnr).
  //    Log-scaled because the usable range spans orders of magnitude:
  //    snr 0.5 (-3 dB) -> 0, snr 2 (+3 dB) -> 0.5, snr >= 10 (+10 dB) -> 1.
  const { snr } = spectralSnr(raw, fs);
  const snrScore = snr > 0 ? clamp01((Math.log10(snr) + 0.3) / 1.2) : 0;

  // 3. Pulse regularity (CV of RR intervals).
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    rrMs.push(((peaks[i].index - peaks[i - 1].index) / fs) * 1000);
  }
  // Apply ectopic/motion interval rejection (same as HRV computation) so that
  // a single outlier interval does not destroy the regularity score.
  const med = median(rrMs);
  const filteredRr = med > 0 ? rrMs.filter((d) => Math.abs(d - med) / med <= 0.3) : rrMs.slice();
  const rrMsClean = filteredRr.length >= 2 ? filteredRr : rrMs;
  // A single interval is no evidence of irregularity; the old code substituted
  // cv = 100 there and scored it 0.
  const cv = rrMsClean.length > 1 ? (stddev(rrMsClean) / mean(rrMsClean)) * 100 : 0;
  const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30);

  // 4. Pulse-presence gate. The previous band-power version compared the
  //    cardiac band of an already-bandpassed signal against its own (numerically
  //    empty) 3.5-12 Hz tail, and its zero-floor fallback of 8 sat *below* the
  //    gate's own zero crossing of 10, so a perfectly clean pulse scored
  //    pulseGate 0 and SQI 0. Narrowband prominence is the physically
  //    meaningful test and has no such degenerate branch.
  const prominence = cardiacSpectralProminence(raw, fs);
  const pulseGate = clamp01((prominence - 4) / 12);

  // Weighted average, still scaled down multiplicatively when no credible pulse
  // is present so that structured artifacts cannot report a healthy score.
  const base = 0.4 * snrScore + 0.35 * piScore + 0.25 * regScore;
  return Math.round(clamp01(base * pulseGate) * 100);
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


function spectralHR(f: number[], fs: number): { bpm: number | null; snrDb: number } {
  const n = f.length;
  const w = f.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))));
  const freqs: number[] = [];
  const power: number[] = [];
  for (let fr = 0.5; fr <= 3.5 + 1e-9; fr += 0.005) {
    let re = 0, im = 0;
    const k = (2 * Math.PI * fr) / fs;
    for (let i = 0; i < n; i++) {
      re += w[i] * Math.cos(k * i);
      im -= w[i] * Math.sin(k * i);
    }
    freqs.push(fr);
    power.push(re * re + im * im);
  }
  let pk = 0;
  for (let i = 1; i < power.length; i++) if (power[i] > power[pk]) pk = i;
  const f0 = freqs[pk];
  let sig = 0, tot = 0;
  for (let i = 0; i < freqs.length; i++) {
    tot += power[i];
    const near = Math.abs(freqs[i] - f0) <= 0.15;
    const harm = Math.abs(freqs[i] - 2 * f0) <= 0.15;
    if (near || harm) sig += power[i];
  }
  const noise = Math.max(tot - sig, 1e-12);
  const snrDb = 10 * Math.log10(Math.max(sig, 1e-12) / noise);
  return { bpm: f0 * 60, snrDb };
}

function detectPeaksAdaptive(f: number[], fs: number, seedRrMs: number, minFloorMs: number, thrWinSec: number): number[] {
  const n = f.length;
  const half = Math.round((thrWinSec * fs) / 2);
  const peaksIdx: number[] = [];
  const peaksAmp: number[] = [];
  const recentRR: number[] = [];

  const thresholdAt = (i: number): number => {
    const seg = f.slice(Math.max(0, i - half), Math.min(n, i + half + 1)).sort((a, b) => a - b);
    const p10 = seg[Math.floor(seg.length * 0.10)];
    const iqr = seg[Math.floor(seg.length * 0.75)] - seg[Math.floor(seg.length * 0.10)];
    const p97 = seg[Math.floor(seg.length * 0.97)];
    return Math.max(p10 + 0.5 * iqr, 0.45 * p97);
  };

  for (let i = 1; i < n - 1; i++) {
    if (!(f[i] > f[i - 1] && f[i] >= f[i + 1])) continue;
    if (f[i] < thresholdAt(i)) continue;

    const medRR = recentRR.length >= 3 ? median(recentRR.slice(-8)) : seedRrMs;
    const minInt = Math.max(minFloorMs, 0.6 * medRR);

    if (peaksIdx.length) {
      const last = peaksIdx[peaksIdx.length - 1];
      const dtMs = ((i - last) / fs) * 1000;
      if (dtMs < minInt) {
        if (f[i] > peaksAmp[peaksAmp.length - 1]) {
          peaksIdx[peaksIdx.length - 1] = i;
          peaksAmp[peaksAmp.length - 1] = f[i];
        }
        continue;
      }
      recentRR.push(dtMs);
    }
    peaksIdx.push(i);
    peaksAmp.push(f[i]);
  }

  return peaksIdx.map((i) => {
    const y0 = f[i - 1], y1 = f[i], y2 = f[i + 1];
    const den = y0 - 2 * y1 + y2;
    const d = Math.abs(den) > 1e-12 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0;
    return ((i + d) / fs) * 1000;
  });
}


export function analyzeHeartRate(ppg: number[], fs = 30, dcRef?: number): HeartMetrics {
  const minFloorMs = 300;
  const thresholdWindowSec = 8;
  const maxPeakSpectralDiffBpm = 8;
  const minValidFraction = 0.7;
  const minSnrDb = 0;

  if (!ppg || ppg.length < fs * 5) {
    return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: 0, method: "none", bpmPeaks: null, bpmSpectral: null, rrMs: [], peakTimesMs: [], validFraction: 0, snrDb: 0, sdnn: null, rmssd: null, reliable: false, perfusionIndex: 0 };
  }

  // Signal quality helper for early returns (uses raw signal)
  const computeBasicSQI = (signal: number[], fs: number, dcRef?: number): number => {
    const rawFiltered = butterworthBandpass(detrend(signal), 0.5, 3.5, fs, true);
    const dc = dcRef && dcRef > 1e-9 ? dcRef : dcLevel(signal);
    const ac = rms(rawFiltered);
    const pi = dc > 1e-9 ? (ac / dc) * 100 : 0;
    const piScore = clamp01(Math.sqrt(pi / 0.5));
    const { snr } = spectralSnr(signal, fs);
    const snrScore = snr > 0 ? clamp01((Math.log10(snr) + 0.3) / 1.2) : 0;
    const prominence = cardiacSpectralProminence(signal, fs);
    const pulseGate = clamp01((prominence - 4) / 12);
    const baseQuality = 0.4 * snrScore + 0.35 * piScore + 0.25 * 0; // no regularity without peaks
    return Math.round(clamp01(baseQuality * pulseGate) * 100);
  };

  // Normalise
  let x = dcRef && dcRef > 0 ? ppg.map((v) => v / dcRef - 1) : ppg.slice();
  const xMean = x.reduce((a, b) => a + b, 0) / x.length;
  x = x.map((v) => -(v - xMean));

  // Band-pass using existing butterworthBandpass
  const f = butterworthBandpass(x, 0.5, 3.5, fs, true);

  // Spectral estimate
  const spec = spectralHR(f, fs);
  const bpmSpectral = spec.bpm;
  const seedRr = bpmSpectral ? 60000 / bpmSpectral : 800;

  // Peak detection with new algorithm
  const peakTimesMs = detectPeaksAdaptive(f, fs, seedRr, minFloorMs, thresholdWindowSec);
  if (peakTimesMs.length < 4) {
    const basicSQI = computeBasicSQI(ppg, fs, dcRef);
    const basicPi = dcRef && dcRef > 1e-9 ? (rms(butterworthBandpass(detrend(ppg), 0.5, 3.5, fs, true)) / dcRef) * 100 : 0;
    return { bpm: bpmSpectral && bpmSpectral > 30 && bpmSpectral < 220 ? bpmSpectral : null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: basicSQI, method: "spectral", bpmPeaks: null, bpmSpectral: bpmSpectral && bpmSpectral > 30 && bpmSpectral < 220 ? bpmSpectral : null, rrMs: [], peakTimesMs: [], validFraction: 0, snrDb: spec.snrDb, sdnn: null, rmssd: null, reliable: false, perfusionIndex: basicPi };
  }

  // Convert peak times to sample indices for backward compatibility
  const peaks = peakTimesMs.map((t) => Math.round((t / 1000) * fs));

  // RR intervals
  const rrAll: number[] = [];
  for (let i = 1; i < peakTimesMs.length; i++) rrAll.push(peakTimesMs[i] - peakTimesMs[i - 1]);
  const med = median(rrAll);
  const ok = rrAll.map((r) => r >= 0.7 * med && r <= 1.3 * med);
  const rrMs = rrAll.filter((_, i) => ok[i]);
  if (rrMs.length < 3) {
    const basicSQI = computeBasicSQI(ppg, fs, dcRef);
    const basicPi = dcRef && dcRef > 1e-9 ? (rms(butterworthBandpass(detrend(ppg), 0.5, 3.5, fs, true)) / dcRef) * 100 : 0;
    return { bpm: bpmSpectral && bpmSpectral > 30 && bpmSpectral < 220 ? bpmSpectral : null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: basicSQI, method: "spectral", bpmPeaks: null, bpmSpectral: bpmSpectral && bpmSpectral > 30 && bpmSpectral < 220 ? bpmSpectral : null, rrMs: [], peakTimesMs: [], validFraction: 0, snrDb: spec.snrDb, sdnn: null, rmssd: null, reliable: false, perfusionIndex: basicPi };
  }

  const bpmVal = 60000 / median(rrMs);
  const bpm = bpmVal > 30 && bpmVal < 220 ? bpmVal : null;
  const hrv = computeHRV(rrMs);
  const rhythm = classifyRhythm(rrMs);

  // Signal quality using refined computation (like old computeSignalQuality)
  // Use raw signal for perfusion index and SNR (matches old computeSignalQuality)
  const rawFiltered = butterworthBandpass(detrend(ppg), 0.5, 3.5, fs, true);
  const dc = dcRef && dcRef > 1e-9 ? dcRef : dcLevel(ppg);
  const ac = rms(rawFiltered);
  const pi = dc > 1e-9 ? (ac / dc) * 100 : 0;
  const piScore = clamp01(Math.sqrt(pi / 0.5));

  // 2. Bandwidth-matched spectral SNR on raw signal
  const { snr } = spectralSnr(ppg, fs);
  const snrScore = snr > 0 ? clamp01((Math.log10(snr) + 0.3) / 1.2) : 0;

  // 3. Pulse regularity (CV of RR intervals)
  const rrMsClean = rrMs.length >= 2 ? rrMs : [];
  const cv = rrMsClean.length > 1 ? (stddev(rrMsClean) / mean(rrMsClean)) * 100 : 0;
  const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30);

  // 4. Pulse-presence gate using cardiac spectral prominence
  const prominence = cardiacSpectralProminence(ppg, fs);
  const pulseGate = clamp01((prominence - 4) / 12);

  // Weighted average, scaled by pulse gate
  const baseQuality = 0.4 * snrScore + 0.35 * piScore + 0.25 * regScore;
  const signalQuality = Math.round(clamp01(baseQuality * pulseGate) * 100);

  // New reliability gate for SpO2/glucose/creatinine gating
  const bpmForQuality = bpm ?? bpmSpectral ?? 60;
  const durSec = f.length / fs;
  const expectedBeats = Math.max(1, (durSec * bpmForQuality) / 60 - 1);
  const validFraction = Math.min(1, rrMs.length / expectedBeats);
  const agree = bpm !== null && bpmSpectral !== null && Math.abs(bpm - bpmSpectral) <= maxPeakSpectralDiffBpm;
  const snrOk = spec.snrDb >= minSnrDb;
  const reliable = validFraction >= minValidFraction && agree && snrOk;

  return {
    bpm,
    peaks,
    rrIntervals: rrMs,
    hrv,
    rhythm,
    signalQuality,
    method: reliable ? "peaks" : "spectral",
    bpmPeaks: bpm,
    bpmSpectral: bpmSpectral && bpmSpectral > 30 && bpmSpectral < 220 ? bpmSpectral : null,
    rrMs,
    peakTimesMs,
    validFraction,
    snrDb: spec.snrDb,
    sdnn: hrv?.sdnn ?? null,
    rmssd: hrv?.rmssd ?? null,
    reliable,
    perfusionIndex: pi,
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

export async function analyzeRespRate(
  ppg: number[],
  fs = 30,
  cameraMode: "front" | "rear" = "rear",
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = []
): Promise<RespMetrics> {
  if (ppg.length < fs * 10) {
    return { respRate: null, method: "insufficient-data", confidence: 0, waveform: [] };
  }

  if (cameraMode === "front") {
    const result = extractRPPGRespiration(redSignal, greenSignal, blueSignal, fs);
    return {
      respRate: result.respRate,
      method: result.method,
      confidence: result.confidence,
      waveform: result.waveform
    };
  } else {
    const result = estimateRespRateRearPPG(ppg, fs);
    return {
      respRate: result.respRate,
      method: result.method,
      confidence: result.confidence,
      waveform: result.waveform
    };
  }
}

export function analyzeRespRateSync(
  ppg: number[],
  fs = 30,
  cameraMode: "front" | "rear" = "rear",
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = []
): RespMetrics {
  if (ppg.length < fs * 10) {
    return { respRate: null, method: "insufficient-data", confidence: 0, waveform: [] };
  }

  if (cameraMode === "front") {
    const result = extractRPPGRespiration(redSignal, greenSignal, blueSignal, fs);
    return {
      respRate: result.respRate,
      method: result.method,
      confidence: result.confidence,
      waveform: result.waveform
    };
  } else {
    const result = estimateRespRateRearPPG(ppg, fs);
    return {
      respRate: result.respRate,
      method: result.method,
      confidence: result.confidence,
      waveform: result.waveform
    };
  }
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

  // Normalize features to [0,1] range for regression
  // Typical ranges for contact PPG (finger):
  // stiffnessIndex: 0.5-5 (upstroke slope/PWV proxy)
  // reflectionIndex (augmentation): 0.1-0.8
  const stiffnessFactor = clamp01((features.stiffnessIndex - 0.5) / 4.5);
  const reflectionFactor = clamp01((features.reflectionIndex - 0.1) / 0.7);

  // Population base values (mmHg)
  let baseSystolic = 115 + (heartRate - 70) * 0.4;
  let baseDiastolic = 75 + (heartRate - 70) * 0.15;

  // Morphology adjustments (mmHg)
  // Stiffer arteries → higher systolic, lower diastolic
  // Higher reflection → higher systolic, lower diastolic
  const sysAdjust = stiffnessFactor * 25 - reflectionFactor * 12 + (hrv?.sdnn ?? 0) * 0.15;
  const diaAdjust = -stiffnessFactor * 8 - reflectionFactor * 5 + (hrv?.rmssd ?? 0) * 0.08;

  baseSystolic += sysAdjust;
  baseDiastolic += diaAdjust;

  const pulsePressure = baseSystolic - baseDiastolic;
  const map = baseDiastolic + pulsePressure / 3;

  // Confidence based on feature quality and HRV
  const confidence = clamp01(0.35 + stiffnessFactor * 0.2 + reflectionFactor * 0.15 + (hrv?.sdnn ?? 0) / 100 * 0.1);

  return {
    systolic: Math.round(Math.min(190, Math.max(90, baseSystolic))),
    diastolic: Math.round(Math.min(120, Math.max(50, baseDiastolic))),
    map: Math.round(Math.min(150, Math.max(60, map))),
    pulsePressure: Math.round(Math.min(100, Math.max(25, pulsePressure))),
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

function channelAcDc(signal: number[]): number {
  if (signal.length < 3) return 0;
  const n = signal.length;
  const dc = signal.reduce((a, b) => a + b, 0) / n;
  if (dc <= 1e-9) return 0;
  const mean = dc;
  let sq = 0;
  for (const v of signal) sq += (v - mean) ** 2;
  const ac = Math.sqrt(sq / n);
  return ac / dc;
}

/// For rear-camera contact PPG through a fingertip, all three color channels
/// carry useful pulsatile signal. Green usually has the highest SNR, but red
/// and blue still contribute measurable AC, especially under strong torch
/// illumination. This helper fuses the three raw traces into one optimized
/// PPG signal by weighting each channel by its own AC/DC (perfusion index)
/// so the strongest channel dominates automatically.
export function combinePpgChannels(red: number[], green: number[], blue: number[]): number[] {
  const n = Math.min(red.length, green.length, blue.length);
  if (n === 0) return green.length > 0 ? green : red.length > 0 ? red : blue;

  const dcR = red.reduce((a, b) => a + b, 0) / n;
  const dcG = green.reduce((a, b) => a + b, 0) / n;
  const dcB = blue.reduce((a, b) => a + b, 0) / n;

  const piR = channelAcDc(red);
  const piG = channelAcDc(green);
  const piB = channelAcDc(blue);
  const totalPi = piR + piG + piB;

  let wR: number, wG: number, wB: number;
  if (totalPi > 1e-9) {
    wR = piR / totalPi;
    wG = piG / totalPi;
    wB = piB / totalPi;
  } else {
    wR = 0.25;
    wG = 0.5;
    wB = 0.25;
  }

  const result: number[] = [];
  for (let i = 0; i < n; i++) {
    const rNorm = (red[i] - dcR) / dcR;
    const gNorm = (green[i] - dcG) / dcG;
    const bNorm = (blue[i] - dcB) / dcB;
    const combined = wR * rNorm + wG * gNorm + wB * bNorm;
    result.push(dcG + combined * dcG);
  }
  return result;
}

/// Select the best single PPG channel for heart-rate
/// analysis based on per-recording spectral SNR. For rear-camera finger PPG, RED is preferred
/// because hemoglobin absorbs red less strongly than green, giving better
/// penetration through finger tissue and higher SNR. For front-camera rPPG,
/// GREEN is preferred because the POS algorithm uses green as the reference.
///
/// Saturation rejection: any channel with >10% samples at or above
/// 250 (out of 255) is treated as unusable and automatically loses.
///
/// Both channels are bandpass-filtered 0.5-8 Hz before SNR estimation
/// so the metric reflects usable cardiac content, not raw DC or
/// high-frequency sensor noise.
export function selectBestPpgChannel(
  red: number[],
  green: number[],
  blue: number[],
  fs: number,
  preferRed = false
): number[] {
  const n = Math.min(red.length, green.length);
  if (n < Math.floor(fs * 2)) return green.slice(-n);

  const satThreshold = 250;
  const redSatFrac = red.filter((v) => v >= satThreshold).length / red.length;
  const greenSatFrac = green.filter((v) => v >= satThreshold).length / green.length;

  const redFiltered = butterworthBandpass(red, 0.5, 8.0, fs, true);
  const greenFiltered = butterworthBandpass(green, 0.5, 8.0, fs, true);

  const redSnr = spectralSnr(redFiltered, fs);
  const greenSnr = spectralSnr(greenFiltered, fs);

  const redScore = redSatFrac > 0.1 ? -1 : redSnr.snr;
  const greenScore = greenSatFrac > 0.1 ? -1 : greenSnr.snr;

  // For rear camera (finger PPG): prefer RED unless green is significantly better
  // For front camera (rPPG): prefer GREEN as reference channel
  if (preferRed) {
    if (redScore >= greenScore || redScore >= greenScore * 0.8) {
      return redFiltered;
    }
    return greenFiltered;
  } else {
    // Higher SNR wins; stay on green when the two are close (within 20%).
    if (greenScore >= redScore || greenScore >= redScore * 0.8) {
      return greenFiltered;
    }
    return redFiltered;
  }
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

/// Cardiac band used everywhere (peak detection, POS analysis, SQI).
const CARDIAC_LO = 0.5;
const CARDIAC_HI = 3.5;

/// Respiratory band (0.05-0.5 Hz covers 3-30 breaths/min). Kept wide on the
/// low side because slow breathing and chest-impedance baseline drift share the
/// same sub-0.1 Hz energy; the estimator fuses ACF/FFT/peak methods and clamps
/// the final rate, so a wide band is more robust than a narrow one.
const RESP_LO = 0.05;
const RESP_HI = 0.5;

/// SpO2 / Hb calibration constants for the red/green ratio-of-ratios.
///
/// The textbook pulse-oximeter formula `SpO2 = A - B * RoR` is calibrated for
/// the RED (660 nm) / INFRARED (940 nm) pair, where RoR is ~0.4-0.8 across the
/// clinical range. A smartphone camera has no IR channel, so the accepted
/// contact-camera proxy is the red/green pair. Green (550 nm) is absorbed far
/// more strongly by hemoglobin than red is, so the red/green RoR is typically
/// **~0.5-2.0** rather than 0.4-0.8, and the old `110 - 25 * R` formula (which
/// expects R < 1) returned SpO2 in the 60-85% range for every healthy subject
/// with R > 1 — the "grossly low" values the user reported.
///
/// These constants are the linearised red/green calibration. They are a
/// **per-device estimate** — the real relationship depends on the camera's
/// spectral quantum efficiency, the flash spectrum, and the optical path — so
/// the UI keeps an `spo2Offset` slider and the validation dashboard exists to
/// calibrate a given phone. The slope is tuned for the red/green RoR which is
/// typically 0.5-1.3 for contact PPG; a healthy subject (SpO2 97-99%) has
/// R ~ 0.55-0.75, so slope ~ 20 and intercept ~ 110 map this to 97-99%.
///
/// **Log-domain method (from PMC10705321)**: Uses A=110, B=25 with window-wise
/// median of log-domain std ratios for motion robustness.
const SPO2_RED_GREEN_SLOPE = 20;
const SPO2_RED_GREEN_INTERCEPT = 110;
const SPO2_RED_GREEN_MIN = 88;
const SPO2_RED_GREEN_MAX = 100;

/// Log-domain SpO2 calibration (PMC10705321 method).
/// A=110, B=25 with window-wise median of std(log-filtered) ratios.
const SPO2_LOG_SLOPE = 25;
const SPO2_LOG_INTERCEPT = 110;
const SPO2_LOG_MIN = 70;
const SPO2_LOG_MAX = 100;

/// Log-domain window parameters (from paper: 10s window, 5s step @ 30Hz)
const SPO2_LOG_WINDOW_SEC = 10;
const SPO2_LOG_STEP_SEC = 5;

/// Haemoglobin calibration for the red/green ratio-of-ratios. The dual-
/// wavelength Hb literature (Wang et al.) fits Hb = a - b * R with the red/green
/// RoR; a healthy adult has R in ~0.5-2.0, so the intercept/slope pair must
/// put the result in ~12-18 g/dL there. The old `18.2 - 6.8 * R` returned
/// 5-10 g/dL for every healthy subject with R > 1.2 because it was calibrated
/// for the much smaller red/IR RoR.
const HB_RED_GREEN_SLOPE = 2.8;
const HB_RED_GREEN_INTERCEPT = 18.0;
const HB_MIN = 10;
const HB_MAX = 20;

/// Zero-level offset (ZLO) correction. Smartphone cameras apply per-frame tone
/// mapping and the sensor has a dark current / black level that varies with
/// exposure and temperature; subtracting the ZLO from the DC before the AC/DC
/// ratio is what makes the ratio-of-ratios reproducible across devices (see
/// Allen et al., Frontiers in Digital Health 2024). The calibration database
/// at `calibration_database.json` carries a per-model ZLO; this is the
/// fallback used when the device is unknown.
const DEFAULT_ZLO = 16.0;

/// Width of the out-of-band reference band above the cardiac band, in Hz. Kept
/// bounded (rather than running all the way to fs*0.4) so the density estimate
/// stays inside the flat part of the sensor/codec noise floor instead of
/// climbing into the steep low-frequency roll-off of the 1/f corner.
const NOISE_BAND_WIDTH = 6;

export interface SpectralSnr {
  /// Bandwidth-matched in-band / out-of-band power ratio. Independent of the
  /// reference band width and of the sampling rate.
  snr: number;
  signalPower: number;
  /// Mean per-bin power density of the reference band.
  noiseDensity: number;
  sigBins: number;
  noiseBins: number;
  /// False when the signal is band-limited by construction and therefore has
  /// no measurable out-of-band floor (e.g. an already-bandpassed pulse). The
  /// `snr` field is then Infinity and must not be used to penalise the signal.
  hasReference: boolean;
}

/// Bandwidth-matched cardiac-band SNR.
///
/// The previous SQI compared the summed power of the 0.5-3.5 Hz cardiac band
/// against the summed power of the whole 3.5 Hz .. fs*0.4 range (~8.5 Hz wide at
/// 30 fps, ~20 Hz wide at 60 fps). That is not an SNR: the noise side is ~3x
/// wider than the signal side, so broadband sensor/codec noise is amplified by
/// that ratio, and the result also drifts with the frame rate. On synthetic
/// facial rPPG the POS pulse recovered 72.0 bpm against a true 72 bpm while this
/// metric read 1.36, which the hard `snr < 1 -> clamp to 0` branch then turned
/// into SQI 7% — a fully usable signal reported as unusable.
///
/// Here the reference band is converted to its *equivalent in-band power*
/// (mean bin density x number of cardiac bins), so the ratio is a true SNR with
/// matched bandwidths and is therefore stable across frame rates and noise
/// colours.
export function spectralSnr(signal: number[], fs: number): SpectralSnr {
  const none: SpectralSnr = { snr: 0, signalPower: 0, noiseDensity: 0, sigBins: 0, noiseBins: 0, hasReference: false };
  const { freqs, psd } = powerSpectrum(signal, fs);
  if (freqs.length === 0) return none;

  const noiseHi = Math.min((fs / 2) * 0.9, CARDIAC_HI + NOISE_BAND_WIDTH);
  let signalPower = 0;
  let sigBins = 0;
  let noisePower = 0;
  let noiseBins = 0;
  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    if (f >= CARDIAC_LO && f <= CARDIAC_HI) {
      signalPower += psd[i];
      sigBins++;
    } else if (f > CARDIAC_HI && f <= noiseHi) {
      noisePower += psd[i];
      noiseBins++;
    }
  }
  if (sigBins === 0) return none;

  const noiseDensity = noiseBins > 0 ? noisePower / noiseBins : 0;
  const inBandNoise = noiseDensity * sigBins;
  if (inBandNoise <= 0) {
    // Band-limited signal: nothing to compare against, so do not penalise it.
    return { snr: Infinity, signalPower, noiseDensity, sigBins, noiseBins, hasReference: false };
  }
  return { snr: signalPower / inBandNoise, signalPower, noiseDensity, sigBins, noiseBins, hasReference: true };
}

/// Peak-to-median power ratio inside the cardiac band: how strongly the pulse
/// concentrates its energy into a narrow fundamental (plus dicrotic harmonic)
/// instead of spreading it over a flat floor. For a Gaussian noise field the
/// expected maximum/median of ~90 bins is only ~2.5-3.5, so this separates real
/// pulses from structured artifacts far more reliably than a band-power ratio.
export function cardiacSpectralProminence(signal: number[], fs: number): number {
  const { freqs, psd } = powerSpectrum(signal, fs);
  const band: number[] = [];
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= CARDIAC_LO && freqs[i] <= CARDIAC_HI) band.push(psd[i]);
  }
  if (band.length < 8) return 0;
  const sorted = band.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const peak = sorted[sorted.length - 1];
  return median > 0 ? peak / median : 0;
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
  const greenPI = filteredPerfusionIndex(green.length > 0 ? green : ppg, fs);
  const redPI = filteredPerfusionIndex(red.length > 0 ? red : ppg, fs);
  const bluePI = filteredPerfusionIndex(blue.length > 0 ? blue : ppg, fs);
  // `ppgPI` is 0 whenever the analysed pulse is zero-mean (the POS chrominance
  // rPPG pulse), and red is the *weakest* pulsatile channel on a face
  // (u_pbv ~ [0.33, 0.53, 0.77] for R/B/G), so the fallback order must prefer
  // green — otherwise the front-mode perfusion descriptor is read off the
  // least informative channel.
  const perfusionIndex = ppgPI || greenPI || redPI;
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

// BPNet type import only — the heavy tfjs module is loaded dynamically on demand
import type { BPResult } from './bpnet-inference';

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
  bpDeep: BPResult | null;
  glucose: GlucoseMetrics;
  creatinine: CreatinineMetrics;
}

export interface AnalyzeOptions {
  cameraMode?: "front" | "rear";
  ageYears?: number | null;
  sex?: "male" | "female" | null;
  respSource?: number[];
  /// DC (illumination) level of the source trace `ppg` was extracted from,
  /// used as the AC/DC denominator of the perfusion term in the SQI. Callers
  /// analysing a zero-mean derived pulse (POS chrominance rPPG) must pass the
  /// skin-ROI mean here, otherwise the perfusion term is meaningless.
  dcRef?: number;
}

export function analyzeVitals(
  ppg: number[],
  fs = 30,
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = [],
  options: AnalyzeOptions = {}
): VitalAnalysis {
  // Green is the highest-SNR camera channel, so it is the reference for the DC
  // level that the pulse's AC amplitude is expressed against.
  const dcRef = options.dcRef ?? dcLevel(greenSignal.length > 0 ? greenSignal : ppg);
  const heart = analyzeHeartRate(ppg, fs, dcRef);
  const resp = analyzeRespRateSync(
    options.respSource ?? ppg,
    fs,
    options.cameraMode ?? "rear",
    redSignal,
    greenSignal,
    blueSignal
  );
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
    bpDeep: null,
    glucose,
    creatinine,
  };
}

export async function analyzeVitalsWithDeepBP(
  ppg: number[],
  fs = 30,
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = [],
  options: AnalyzeOptions = {}
): Promise<VitalAnalysis> {
  const base = analyzeVitals(ppg, fs, redSignal, greenSignal, blueSignal, options);
  const { estimateBPDeep } = await import('./bpnet-inference');
  const bpDeep = await estimateBPDeep(ppg, fs);
  return { ...base, bpDeep };
}
