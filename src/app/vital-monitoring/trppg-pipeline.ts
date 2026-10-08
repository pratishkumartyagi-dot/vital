/**
 * TrPPG — True rPPG Pipeline (front camera)
 *
 * A complete, self-contained remote photoplethysmography pipeline for
 * front-camera facial video. Implements the 10-stage specification:
 *
 *   1. Preprocessing      — ROI spatial mean, normalisation, detrend, bandpass
 *   2. Pulse signal       — Green / CHROM / POS extraction
 *   3. Heart rate + SQI   — Welch PSD, SNR quality index
 *   4. Respiratory rate   — baseline wander + AM + FM, SQI-weighted fusion
 *   5. HRV                — cubic-spline upsampling, NN intervals, Lomb-Scargle
 *   6. SpO2               — AC/DC ratio-of-ratios with empirical calibration
 *   7. Blood pressure     — single-site morphology + two-site PTT
 *   8. Hemoglobin         — absorbance features, regularised regression
 *   9. Glucose/creatinine — trained-model interface with covariate baseline
 *  10. Evaluation         — Bland-Altman, MAE/RMSE/ARMS, Lin CCC, LOO-CV
 *
 * Every stage is pure and deterministic; no DOM, no camera, no I/O.
 * The caller supplies raw per-frame ROI colour means and receives a
 * TrPPGResult with all derived vitals plus the evaluation scaffolding.
 */

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export type PulseMethod = "green" | "chrom" | "pos";

export interface ROIRegion {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface TrPPGConfig {
  /** Sampling rate of the input frames (Hz). */
  fs: number;
  /** Region of interest in frame coordinates. */
  roi: ROIRegion;
  /** Pulse extraction method. */
  method: PulseMethod;
  /** Analysis window length in seconds (default 30). */
  windowSec: number;
  /** POS/CHROM window length in seconds (default 1.6). */
  pulseWindowSec: number;
  /** Cardiac band (Hz). */
  cardiacLow: number;
  cardiacHigh: number;
  /** Respiratory band (Hz). */
  respLow: number;
  respHigh: number;
  /** HRV upsampling target rate (Hz, default 120). */
  hrvUpsampleHz: number;
  /** Per-device SpO2 calibration coefficients. */
  spo2Cal?: SpO2Calibration;
  /** Per-device Hb calibration coefficients. */
  hbCal?: HbCalibration;
  /** Per-user BP calibration offset (mmHg). */
  bpOffset?: BPOffset;
  /** Skin tone covariate (Fitzpatrick 1-6 or 0=unknown). */
  skinTone?: number;
  /** Ambient light level (lux, 0=unknown). */
  ambientLux?: number;
}

export interface SpO2Calibration {
  /** Linear: SpO2 = a + b*R */
  a: number;
  b: number;
  /** Optional quadratic term: + c*R^2 */
  c?: number;
}

export interface HbCalibration {
  a: number;
  b: number;
  c?: number;
}

export interface BPOffset {
  systolic: number;
  diastolic: number;
}

/** Per-frame ROI colour means (the only required input). */
export interface FrameSample {
  r: number;
  g: number;
  b: number;
  /** Optional timestamp in seconds; defaults to index/fs. */
  t?: number;
}

export interface TrPPGResult {
  hr: number | null;
  sqi: number;
  pulse: number[];
  pulseMethod: PulseMethod;
  respRate: number | null;
  respConfidence: number;
  hrv: HRVResult;
  spo2: number | null;
  spo2R: number | null;
  bp: BPResult;
  hb: number | null;
  glucose: number | null;
  creatinine: number | null;
  /** Diagnostics for validation. */
  diagnostics: TrPPGDiagnostics;
}

export interface HRVResult {
  sdnn: number | null;
  rmssd: number | null;
  pnn50: number | null;
  lf: number | null;
  hf: number | null;
  lfhf: number | null;
  nnIntervals: number[];
  primary: "rmssd" | "sdnn";
}

export interface BPResult {
  systolic: number | null;
  diastolic: number | null;
  map: number | null;
  confidence: number;
  method: "single-site" | "two-site";
  pttMs: number | null;
}

export interface TrPPGDiagnostics {
  welchFreqs: number[];
  welchPsd: number[];
  hrPeakFreq: number | null;
  snrDb: number;
  respPeaks: { baseline: number | null; am: number | null; fm: number | null };
  ibiMs: number[];
  nnMs: number[];
  calibrationUsed: boolean;
}

// ─────────────────────────────────────────────────────────────────────
// Stage 1 — Preprocessing
// ─────────────────────────────────────────────────────────────────────

/**
 * Spatial mean of each channel over the ROI per frame is supplied by the
 * caller as FrameSample[]. This stage normalises, detrends and bandpass
 * filters each channel.
 *
 * Normalise:  Cn(t) = C(t) / mean(C)
 * Detrend:    remove linear trend (least-squares fit)
 * Bandpass:   Butterworth 0.7-3.0 Hz (pulse band), zero-phase
 */
export function preprocessChannels(
  samples: FrameSample[],
  cfg: TrPPGConfig
): { r: number[]; g: number[]; b: number[]; rN: number[]; gN: number[]; bN: number[] } {
  const n = samples.length;
  const r = new Array<number>(n);
  const g = new Array<number>(n);
  const b = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    r[i] = samples[i].r;
    g[i] = samples[i].g;
    b[i] = samples[i].b;
  }

  // Normalise each channel by its mean
  const rN = normalise(r);
  const gN = normalise(g);
  const bN = normalise(b);

  // Detrend (linear) then bandpass 0.7-3.0 Hz
  const rF = butterworthBandpass(detrend(rN), cfg.cardiacLow, cfg.cardiacHigh, cfg.fs, true);
  const gF = butterworthBandpass(detrend(gN), cfg.cardiacLow, cfg.cardiacHigh, cfg.fs, true);
  const bF = butterworthBandpass(detrend(bN), cfg.cardiacLow, cfg.cardiacHigh, cfg.fs, true);

  return { r, g, b, rN: rF, gN: gF, bN: bF };
}

function normalise(x: number[]): number[] {
  const m = mean(x);
  if (m <= 1e-9) return x.slice();
  return x.map((v) => v / m);
}

function detrend(x: number[]): number[] {
  const n = x.length;
  if (n < 3) return x.slice();
  const xm = mean(x);
  const tm = (n - 1) / 2;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - tm) * (x[i] - xm);
    den += (i - tm) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = xm - slope * tm;
  return x.map((v, i) => v - (slope * i + intercept));
}

// ─────────────────────────────────────────────────────────────────────
// Stage 2 — Pulse signal extraction
// ─────────────────────────────────────────────────────────────────────

/**
 * Extract the pulse signal p(t) using the configured method.
 *
 * Green:  p(t) = Gn(t)
 * CHROM:  Xs = 3Rn − 2Gn, Ys = 1.5Rn + Gn − 1.5Bn, p = Xs − αYs, α = σ(Xs)/σ(Ys)
 * POS:    S1 = Gn − Bn, S2 = Gn + Bn − 2Rn, p = S1 + (σ(S1)/σ(S2))·S2
 *         overlap-added over ~1.6 s windows (Hann, 50% overlap)
 */
export function extractPulse(
  rN: number[],
  gN: number[],
  bN: number[],
  cfg: TrPPGConfig
): { pulse: number[]; method: PulseMethod } {
  switch (cfg.method) {
    case "green":
      return { pulse: gN.slice(), method: "green" };
    case "chrom":
      return { pulse: chromPulse(rN, gN, bN), method: "chrom" };
    case "pos":
      return { pulse: posPulse(rN, gN, bN, cfg), method: "pos" };
  }
}

/** CHROM (De Haan & Jeanne 2013). */
function chromPulse(rN: number[], gN: number[], bN: number[]): number[] {
  const n = Math.min(rN.length, gN.length, bN.length);
  const xs = new Array<number>(n);
  const ys = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    xs[i] = 3 * rN[i] - 2 * gN[i];
    ys[i] = 1.5 * rN[i] + gN[i] - 1.5 * bN[i];
  }
  const sx = stddev(xs);
  const sy = stddev(ys);
  const alpha = sy > 1e-12 ? sx / sy : 0;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = xs[i] - alpha * ys[i];
  return out;
}

/** POS (Wang et al. 2017), overlap-added over ~1.6 s Hann windows. */
function posPulse(rN: number[], gN: number[], bN: number[], cfg: TrPPGConfig): number[] {
  const n = Math.min(rN.length, gN.length, bN.length);
  const winLen = Math.max(8, Math.round(cfg.pulseWindowSec * cfg.fs));
  if (n < winLen) return new Array<number>(n).fill(0);
  const hop = Math.max(1, Math.floor(winLen / 2));
  const acc = new Array<number>(n).fill(0);
  const wsum = new Array<number>(n).fill(0);

  for (let start = 0; start + winLen <= n; start += hop) {
    // Window means for temporal normalisation
    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < winLen; i++) {
      mr += rN[start + i];
      mg += gN[start + i];
      mb += bN[start + i];
    }
    mr /= winLen; mg /= winLen; mb /= winLen;
    if (mr <= 1e-9 || mg <= 1e-9 || mb <= 1e-9) continue;

    // Normalised channels within window
    const cnR = new Array<number>(winLen);
    const cnG = new Array<number>(winLen);
    const cnB = new Array<number>(winLen);
    for (let i = 0; i < winLen; i++) {
      cnR[i] = rN[start + i] / mr;
      cnG[i] = gN[start + i] / mg;
      cnB[i] = bN[start + i] / mb;
    }

    // POS projections
    const s1 = new Array<number>(winLen);
    const s2 = new Array<number>(winLen);
    let m1 = 0, m2 = 0;
    for (let i = 0; i < winLen; i++) {
      s1[i] = cnG[i] - cnB[i];
      s2[i] = cnG[i] + cnB[i] - 2 * cnR[i];
      m1 += s1[i];
      m2 += s2[i];
    }
    m1 /= winLen; m2 /= winLen;

    let v1 = 0, v2 = 0;
    for (let i = 0; i < winLen; i++) {
      v1 += (s1[i] - m1) ** 2;
      v2 += (s2[i] - m2) ** 2;
    }
    const alpha = v2 > 1e-12 ? Math.sqrt(v1 / v2) : 0;

    for (let i = 0; i < winLen; i++) {
      const h = (s1[i] - m1) + alpha * (s2[i] - m2);
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (winLen - 1)));
      acc[start + i] += h * w;
      wsum[start + i] += w;
    }
  }

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = wsum[i] > 1e-9 ? acc[i] / wsum[i] : 0;
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Stage 3 — Heart rate and quality (Welch PSD)
// ─────────────────────────────────────────────────────────────────────

export interface WelchResult {
  freqs: number[];
  psd: number[];
  peakFreq: number | null;
  snrDb: number;
  snrLinear: number;
}

/**
 * Welch PSD of p(t): segment into overlapping Hann windows, average
 * periodograms. f_hr = argmax P(f) in [cardiacLow, cardiacHigh].
 * SNR = power within ±0.1 Hz of f_hr (+ 2nd harmonic) / remaining band power.
 */
export function welchPSD(signal: number[], fs: number, cfg: TrPPGConfig): WelchResult {
  const n = signal.length;
  if (n < 8) return { freqs: [], psd: [], peakFreq: null, snrDb: 0, snrLinear: 0 };

  // Segment: 256-sample windows (or n/4), 50% overlap
  const segLen = Math.min(256, Math.max(16, Math.floor(n / 4)));
  const hop = Math.max(1, Math.floor(segLen / 2));
  const nfft = segLen;
  const numBins = Math.floor(nfft / 2) + 1;
  const psdSum = new Array<number>(numBins).fill(0);
  let segCount = 0;

  for (let start = 0; start + segLen <= n; start += hop) {
    // Hann window
    const win = new Array<number>(segLen);
    let winSq = 0;
    for (let i = 0; i < segLen; i++) {
      win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (segLen - 1)));
      winSq += win[i] * win[i];
    }
    const wNorm = winSq / segLen || 1;

    // Detrend segment
    const seg = new Array<number>(segLen);
    for (let i = 0; i < segLen; i++) seg[i] = signal[start + i];
    const segD = detrend(seg);

    // DFT
    for (let k = 0; k < numBins; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < segLen; i++) {
        const angle = (2 * Math.PI * k * i) / nfft;
        re += segD[i] * win[i] * Math.cos(angle);
        im -= segD[i] * win[i] * Math.sin(angle);
      }
      psdSum[k] += (re * re + im * im) / (segLen * wNorm);
    }
    segCount++;
  }

  const psd = psdSum.map((v) => (segCount > 0 ? v / segCount : 0));
  const freqs: number[] = [];
  for (let k = 0; k < numBins; k++) freqs.push((k * fs) / nfft);

  // Peak in cardiac band
  let peakIdx = -1;
  let peakPow = -Infinity;
  for (let i = 0; i < numBins; i++) {
    if (freqs[i] >= cfg.cardiacLow && freqs[i] <= cfg.cardiacHigh && psd[i] > peakPow) {
      peakPow = psd[i];
      peakIdx = i;
    }
  }
  if (peakIdx < 0) return { freqs, psd, peakFreq: null, snrDb: 0, snrLinear: 0 };

  const fHr = freqs[peakIdx];

  // SNR: power within ±0.1 Hz of f_hr (+ 2nd harmonic) / remaining band power
  let sigPow = 0;
  let totPow = 0;
  for (let i = 0; i < numBins; i++) {
    if (freqs[i] < cfg.cardiacLow || freqs[i] > cfg.cardiacHigh) continue;
    totPow += psd[i];
    const near = Math.abs(freqs[i] - fHr) <= 0.1;
    const harm = Math.abs(freqs[i] - 2 * fHr) <= 0.1;
    if (near || harm) sigPow += psd[i];
  }
  const noisePow = Math.max(totPow - sigPow, 1e-12);
  const snrLinear = sigPow / noisePow;
  const snrDb = 10 * Math.log10(Math.max(snrLinear, 1e-12));

  return { freqs, psd, peakFreq: fHr, snrDb, snrLinear };
}

/**
 * Heart rate from Welch PSD. SQI = clamp01((SNR_dB + 6) / 12) so that
 * SNR = -6 dB → 0, SNR = +6 dB → 1. Every downstream parameter is
 * weighted by this SQI.
 */
export function computeHRAndSQI(pulse: number[], cfg: TrPPGConfig): { hr: number | null; sqi: number; welch: WelchResult } {
  const welch = welchPSD(pulse, cfg.fs, cfg);
  if (welch.peakFreq === null) return { hr: null, sqi: 0, welch };
  const hr = 60 * welch.peakFreq;
  if (hr < 30 || hr > 220) return { hr: null, sqi: 0, welch };
  // SQI from SNR: -6 dB → 0, +6 dB → 1
  const sqi = clamp01((welch.snrDb + 6) / 12);
  return { hr, sqi, welch };
}

// ─────────────────────────────────────────────────────────────────────
// Stage 4 — Respiratory rate (three modulations, SQI-weighted fusion)
// ─────────────────────────────────────────────────────────────────────

export interface RespResult {
  respRate: number | null;
  confidence: number;
  peaks: { baseline: number | null; am: number | null; fm: number | null };
}

/**
 * Three respiratory-induced modulations of p(t):
 *   1. Baseline wander — 0.1-0.5 Hz band of the raw trace
 *   2. Amplitude modulation — Hilbert envelope of p(t)
 *   3. Frequency modulation — inter-beat interval series
 * PSD peak of each, fuse by SQI-weighted median, RR = 60·f_resp.
 */
export function computeRespiratory(
  pulse: number[],
  rawGreen: number[],
  cfg: TrPPGConfig,
  sqi: number
): RespResult {
  const peaks: { baseline: number | null; am: number | null; fm: number | null } = {
    baseline: null,
    am: null,
    fm: null,
  };

  // 1. Baseline wander: bandpass raw trace 0.1-0.5 Hz
  const bw = butterworthBandpass(rawGreen, cfg.respLow, cfg.respHigh, cfg.fs, true);
  peaks.baseline = psdPeakFreq(bw, cfg.fs, cfg.respLow, cfg.respHigh);

  // 2. Amplitude modulation: Hilbert envelope of p(t)
  const envelope = hilbertEnvelope(pulse);
  const envFilt = butterworthBandpass(envelope, cfg.respLow, cfg.respHigh, cfg.fs, true);
  peaks.am = psdPeakFreq(envFilt, cfg.fs, cfg.respLow, cfg.respHigh);

  // 3. Frequency modulation: inter-beat interval series
  const ibi = interBeatIntervals(pulse, cfg.fs);
  if (ibi.length >= 4) {
    // Resample IBI to uniform grid, then bandpass
    const ibiUniform = resampleUniform(ibi, cfg.fs, cfg.hrvUpsampleHz);
    const ibiFilt = butterworthBandpass(ibiUniform, cfg.respLow, cfg.respHigh, cfg.hrvUpsampleHz, true);
    peaks.fm = psdPeakFreq(ibiFilt, cfg.hrvUpsampleHz, cfg.respLow, cfg.respHigh);
  }

  // Fuse by SQI-weighted median
  const candidates: number[] = [];
  const weights: number[] = [];
  for (const [peak, w] of [
    [peaks.baseline, 0.4],
    [peaks.am, 0.35],
    [peaks.fm, 0.25],
  ] as const) {
    if (peak !== null && peak >= cfg.respLow && peak <= cfg.respHigh) {
      candidates.push(peak);
      weights.push(w * (0.5 + sqi)); // SQI-weighted
    }
  }

  if (candidates.length === 0) {
    return { respRate: null, confidence: 0, peaks };
  }

  // Weighted median
  const paired = candidates.map((f, i) => ({ f, w: weights[i] }));
  paired.sort((a, b) => a.f - b.f);
  let cumW = 0;
  const totalW = weights.reduce((a, b) => a + b, 0);
  let fResp = paired[paired.length - 1].f;
  for (const p of paired) {
    cumW += p.w;
    if (cumW >= totalW / 2) {
      fResp = p.f;
      break;
    }
  }

  const respRate = 60 * fResp;
  // Confidence: spread of candidates weighted by SQI
  const spread = candidates.length > 1 ? stddev(candidates) / (mean(candidates) || 1) : 0;
  const confidence = clamp01((1 - spread) * (0.3 + 0.7 * sqi));

  return {
    respRate: respRate >= 5 && respRate <= 50 ? respRate : null,
    confidence,
    peaks,
  };
}

/** PSD peak frequency in a band (simple periodogram). */
function psdPeakFreq(signal: number[], fs: number, fLow: number, fHigh: number): number | null {
  const n = signal.length;
  if (n < 8) return null;
  const sd = detrend(signal);
  // Hann window
  const win = new Array<number>(n);
  let winSq = 0;
  for (let i = 0; i < n; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    winSq += win[i] * win[i];
  }
  const wNorm = winSq / n || 1;

  let bestF = 0;
  let bestP = -Infinity;
  const df = fs / n;
  for (let k = 0; k <= Math.floor(n / 2); k++) {
    const f = k * df;
    if (f < fLow || f > fHigh) continue;
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      re += sd[i] * win[i] * Math.cos(angle);
      im -= sd[i] * win[i] * Math.sin(angle);
    }
    const p = (re * re + im * im) / (n * wNorm);
    if (p > bestP) {
      bestP = p;
      bestF = f;
    }
  }
  return bestP > 0 ? bestF : null;
}

/** Hilbert envelope via analytic signal (Hilbert transform). */
function hilbertEnvelope(x: number[]): number[] {
  const n = x.length;
  if (n < 4) return x.slice();
  // FFT-based Hilbert transform
  const h = hilbertTransform(x);
  const env = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    env[i] = Math.sqrt(x[i] * x[i] + h[i] * h[i]);
  }
  return env;
}

/** Discrete Hilbert transform via FFT. */
function hilbertTransform(x: number[]): number[] {
  const n = x.length;
  // Pad to power of 2 for FFT
  let nfft = 1;
  while (nfft < n) nfft *= 2;
  const re = new Array<number>(nfft).fill(0);
  const im = new Array<number>(nfft).fill(0);
  for (let i = 0; i < n; i++) re[i] = x[i];

  // FFT
  fft(re, im);

  // Multiply by H(f): 1 for f=0, 2 for 0<f<nyquist, 1 for f=nyquist, 0 otherwise
  for (let k = 0; k < nfft; k++) {
    if (k === 0 || k === nfft / 2) {
      // keep as is (multiply by 1)
    } else if (k < nfft / 2) {
      re[k] *= 2;
      im[k] *= 2;
    } else {
      re[k] = 0;
      im[k] = 0;
    }
  }

  // IFFT
  ifft(re, im);

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = im[i]; // imaginary part is the Hilbert transform
  return out;
}

/** In-place radix-2 FFT. */
function fft(re: number[], im: number[]): void {
  const n = re.length;
  if (n <= 1) return;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Cooley-Tukey
  for (let len = 2; len <= n; len *= 2) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curR = 1, curI = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = re[i + j], ui = im[i + j];
        const vr = re[i + j + len / 2] * curR - im[i + j + len / 2] * curI;
        const vi = re[i + j + len / 2] * curI + im[i + j + len / 2] * curR;
        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr;
        im[i + j + len / 2] = ui - vi;
        const nextR = curR * wr - curI * wi;
        curI = curR * wi + curI * wr;
        curR = nextR;
      }
    }
  }
}

/** In-place radix-2 IFFT. */
function ifft(re: number[], im: number[]): void {
  const n = re.length;
  // Conjugate
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

/** Inter-beat intervals from pulse peaks (ms). */
function interBeatIntervals(pulse: number[], fs: number): number[] {
  const filtered = butterworthBandpass(detrend(pulse), 0.7, 3.0, fs, true);
  const peaks = detectPeaksSimple(filtered, fs);
  const ibi: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    ibi.push(((peaks[i] - peaks[i - 1]) / fs) * 1000);
  }
  return ibi;
}

function detectPeaksSimple(signal: number[], fs: number): number[] {
  const n = signal.length;
  const minDist = Math.max(1, Math.floor((60 / 180) * fs));
  const sorted = signal.slice().sort((a, b) => a - b);
  const p10 = sorted[Math.floor(n * 0.1)];
  const p90 = sorted[Math.floor(n * 0.9)];
  const minHeight = p10 + 0.5 * (p90 - p10);
  const peaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1] && signal[i] > minHeight) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      } else if (signal[i] > signal[peaks[peaks.length - 1]]) {
        peaks[peaks.length - 1] = i;
      }
    }
  }
  return peaks;
}

/** Resample an unevenly-spaced series to a uniform grid (linear interp). */
function resampleUniform(values: number[], originalFs: number, targetFs: number): number[] {
  const n = values.length;
  if (n < 2) return values.slice();
  const duration = n / originalFs;
  const targetN = Math.max(2, Math.round(duration * targetFs));
  const out = new Array<number>(targetN);
  for (let i = 0; i < targetN; i++) {
    const t = i / targetFs;
    const srcPos = t * originalFs;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, n - 1);
    const frac = srcPos - i0;
    out[i] = values[i0] * (1 - frac) + values[i1] * frac;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Stage 5 — HRV (cubic-spline upsampling, NN intervals, Lomb-Scargle)
// ─────────────────────────────────────────────────────────────────────

/**
 * Upsample p(t) to ≥120 Hz (cubic spline) before peak detection, because
 * 30 fps gives ±16 ms timing error that biases RMSSD. Then IBI series,
 * ectopic/artefact removal → NN intervals. SDNN, RMSSD, pNN50.
 * LF/HF by Lomb-Scargle on the unevenly sampled NN series.
 * Short windows make LF unreliable, so RMSSD is the primary HRV metric
 * for a 60 s capture.
 */
export function computeHRV(
  pulse: number[],
  cfg: TrPPGConfig
): HRVResult {
  const empty: HRVResult = {
    sdnn: null,
    rmssd: null,
    pnn50: null,
    lf: null,
    hf: null,
    lfhf: null,
    nnIntervals: [],
    primary: "rmssd",
  };

  // Cubic-spline upsample to target Hz
  const up = cubicSplineUpsample(pulse, cfg.fs, cfg.hrvUpsampleHz);
  if (up.length < cfg.hrvUpsampleHz * 2) return empty;

  // Peak detection on upsampled signal
  const filtered = butterworthBandpass(detrend(up), 0.7, 3.0, cfg.hrvUpsampleHz, true);
  const peaks = detectPeaksSimple(filtered, cfg.hrvUpsampleHz);
  if (peaks.length < 3) return empty;

  // IBI series (ms)
  const ibi: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    ibi.push(((peaks[i] - peaks[i - 1]) / cfg.hrvUpsampleHz) * 1000);
  }
  if (ibi.length < 2) return empty;

  // Ectopic/artefact removal: reject intervals outside ±20% of median
  const med = median(ibi);
  const nn = ibi.filter((v) => Math.abs(v - med) / med <= 0.2);
  if (nn.length < 2) return empty;

  // Time-domain HRV
  const sdnn = stddev(nn);
  const diffs: number[] = [];
  let pnn50Count = 0;
  for (let i = 1; i < nn.length; i++) {
    const d = Math.abs(nn[i] - nn[i - 1]);
    diffs.push(d);
    if (d > 50) pnn50Count++;
  }
  const rmssd = diffs.length > 0 ? Math.sqrt(mean(diffs.map((d) => d * d))) : 0;
  const pnn50 = (pnn50Count / diffs.length) * 100;

  // LF/HF by Lomb-Scargle on unevenly sampled NN series
  const { lf, hf } = lombScargleLFHF(nn, cfg.hrvUpsampleHz);

  return {
    sdnn,
    rmssd,
    pnn50,
    lf,
    hf,
    lfhf: hf > 0 ? lf / hf : null,
    nnIntervals: nn,
    primary: "rmssd", // RMSSD primary for short (60 s) captures
  };
}

/** Cubic spline upsampling (natural cubic spline, Catmull-Rom style). */
function cubicSplineUpsample(x: number[], originalFs: number, targetFs: number): number[] {
  const n = x.length;
  if (n < 4) return x.slice();
  const targetN = Math.max(2, Math.round((n / originalFs) * targetFs));
  const out = new Array<number>(targetN);

  // Natural cubic spline coefficients
  const h = new Array<number>(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = 1 / originalFs;

  // Solve tridiagonal system for second derivatives
  const alpha = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    alpha[i] = (3 / h[i]) * (x[i + 1] - x[i]) - (3 / h[i - 1]) * (x[i] - x[i - 1]);
  }
  const l = new Array<number>(n).fill(1);
  const mu = new Array<number>(n).fill(0);
  const z = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    l[i] = 2 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }
  const c = new Array<number>(n).fill(0);
  const b = new Array<number>(n - 1);
  const d = new Array<number>(n - 1);
  for (let j = n - 2; j >= 0; j--) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (x[j + 1] - x[j]) / h[j] - (h[j] * (c[j + 1] + 2 * c[j])) / 3;
    d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
  }

  // Evaluate spline at target times
  for (let i = 0; i < targetN; i++) {
    const t = i / targetFs;
    const srcPos = t * originalFs;
    let j = Math.floor(srcPos);
    if (j >= n - 1) j = n - 2;
    const dt = srcPos - j;
    out[i] = x[j] + b[j] * dt + c[j] * dt * dt + d[j] * dt * dt * dt;
  }
  return out;
}

/** Lomb-Scargle periodogram for LF/HF on unevenly sampled NN series. */
function lombScargleLFHF(nnMs: number[], fs: number): { lf: number; hf: number } {
  // NN series is unevenly sampled (each NN interval is a sample at cumulative time)
  const n = nnMs.length;
  if (n < 4) return { lf: 0, hf: 0 };

  // Build time series: t[i] = cumulative time, y[i] = NN interval
  const t: number[] = [];
  const y: number[] = [];
  let cumT = 0;
  for (let i = 0; i < n; i++) {
    t.push(cumT / 1000); // seconds
    y.push(nnMs[i]);
    cumT += nnMs[i];
  }
  const duration = cumT / 1000;
  if (duration < 20) return { lf: 0, hf: 0 }; // Need ≥20s for LF/HF

  const yMean = mean(y);
  const yVar = stddev(y) ** 2 || 1;

  // Frequency grid: 0.01 Hz resolution
  const df = 0.01;
  const fMax = 0.5;
  const freqs: number[] = [];
  for (let f = df; f <= fMax; f += df) freqs.push(f);

  // Lomb-Scargle power at each frequency
  const power: number[] = [];
  for (const f of freqs) {
    const w = 2 * Math.PI * f;
    // Time delay
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      num += Math.sin(2 * w * t[i]);
      den += Math.cos(2 * w * t[i]);
    }
    const tau = Math.atan2(num, den) / (2 * w);

    let c = 0, s = 0, cc = 0, ss = 0, cs = 0;
    for (let i = 0; i < n; i++) {
      const wt = w * (t[i] - tau);
      const cw = Math.cos(wt);
      const sw = Math.sin(wt);
      const dy = y[i] - yMean;
      c += dy * cw;
      s += dy * sw;
      cc += cw * cw;
      ss += sw * sw;
      cs += cw * sw;
    }
    const p = (c * c * ss + s * s * cc - 2 * c * s * cs) / (cc * ss - cs * cs);
    power.push(p / yVar);
  }

  // Integrate power in LF (0.04-0.15) and HF (0.15-0.40) bands
  let lf = 0, hf = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.04 && freqs[i] < 0.15) lf += power[i];
    if (freqs[i] >= 0.15 && freqs[i] <= 0.40) hf += power[i];
  }
  return { lf, hf };
}

// ─────────────────────────────────────────────────────────────────────
// Stage 6 — SpO2 (AC/DC ratio-of-ratios, empirical calibration)
// ─────────────────────────────────────────────────────────────────────

/**
 * AC_c = spectral amplitude at f_hr (±0.15 Hz) for channel c
 * DC_c = mean(C)
 * R = (AC_red/DC_red) / (AC_green/DC_green)  (green = reference)
 * SpO2 = a + b·R (+ c·R²)  — empirical, fitted per phone model
 *
 * Beer-Lambert theory (guide only, camera bands are broad):
 *   SpO2 = (ε_Hb(λ1) − R·ε_Hb(λ2)) /
 *          [(ε_Hb(λ1) − ε_HbO2(λ1)) + R(ε_HbO2(λ2) − ε_Hb(λ2))]
 */
export function computeSpO2(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  fHr: number | null,
  cfg: TrPPGConfig,
  sqi: number
): { spo2: number | null; R: number | null } {
  if (fHr === null || sqi < 0.3) return { spo2: null, R: null };

  // Spectral amplitude at f_hr (±0.15 Hz) for each channel
  const acR = spectralAmplitude(rRaw, cfg.fs, fHr, 0.15);
  const acG = spectralAmplitude(gRaw, cfg.fs, fHr, 0.15);
  const dcR = mean(rRaw);
  const dcG = mean(gRaw);

  if (dcR <= 1e-9 || dcG <= 1e-9 || acG <= 1e-12) return { spo2: null, R: null };

  const R = (acR / dcR) / (acG / dcG);

  // Empirical calibration (per-device, fitted against reference oximeter)
  const cal = cfg.spo2Cal ?? { a: 110, b: -20 };
  let spo2: number;
  if (cal.c !== undefined) {
    spo2 = cal.a + cal.b * R + cal.c * R * R;
  } else {
    spo2 = cal.a + cal.b * R;
  }

  // Skin tone / ambient light covariates (if validation shows they help)
  if (cfg.skinTone && cfg.skinTone > 0) {
    // Darker skin tones attenuate green more; small correction
    spo2 += (cfg.skinTone - 3.5) * 0.3;
  }
  if (cfg.ambientLux !== undefined && cfg.ambientLux > 0) {
    // High ambient light reduces contrast; small correction
    spo2 -= Math.min(2, cfg.ambientLux / 5000);
  }

  return {
    spo2: clamp(spo2, 70, 100),
    R,
  };
}

/** Spectral amplitude at a target frequency (± tolerance Hz). */
function spectralAmplitude(signal: number[], fs: number, targetFreq: number, tolHz: number): number {
  const n = signal.length;
  if (n < 8) return 0;
  const sd = detrend(signal);
  // Hann window
  const win = new Array<number>(n);
  let winSq = 0;
  for (let i = 0; i < n; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    winSq += win[i] * win[i];
  }
  const wNorm = winSq / n || 1;

  // Search bins near target frequency
  const df = fs / n;
  const kTarget = Math.round(targetFreq / df);
  const kTol = Math.max(1, Math.round(tolHz / df));
  let amp = 0;
  for (let k = Math.max(1, kTarget - kTol); k <= Math.min(Math.floor(n / 2), kTarget + kTol); k++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      re += sd[i] * win[i] * Math.cos(angle);
      im -= sd[i] * win[i] * Math.sin(angle);
    }
    amp += Math.sqrt(re * re + im * im) / (n * Math.sqrt(wNorm));
  }
  return amp;
}

// ─────────────────────────────────────────────────────────────────────
// Stage 7 — Blood pressure
// ─────────────────────────────────────────────────────────────────────

/**
 * Single-site (face only): waveform features (systolic upstroke time,
 * width at 50% amplitude, reflection/augmentation index, APG ratios)
 * + HR, age, sex → SBP = β0 + Σβi·xi (ridge / gradient boosting),
 * with per-user offset from cuff calibration.
 *
 * Two-site (face + finger): PTT = Δt between pulse feet,
 * PWV = L/PTT, BP = a + b·ln(PTT) (or a + b/PTT²).
 */
export function computeBP(
  pulse: number[],
  cfg: TrPPGConfig,
  hr: number | null,
  sqi: number,
  ageYears: number | null,
  sex: "male" | "female" | null,
  fingerPulse?: number[] // optional second site for PTT
): BPResult {
  const empty: BPResult = {
    systolic: null,
    diastolic: null,
    map: null,
    confidence: 0,
    method: "single-site",
    pttMs: null,
  };
  if (hr === null || sqi < 0.3 || pulse.length < cfg.fs * 5) return empty;

  // Two-site PTT if finger pulse provided
  if (fingerPulse && fingerPulse.length >= cfg.fs * 5) {
    const ptt = computePTT(pulse, fingerPulse, cfg);
    if (ptt !== null) {
      // BP = a + b·ln(PTT)  (Moens-Korteweg derived)
      // Default coefficients; per-user calibration overrides
      const a = 130, b = -25;
      const sbp = a + b * Math.log(Math.max(ptt, 50));
      const dbp = sbp * 0.6; // rough diastolic estimate
      return {
        systolic: clamp(Math.round(sbp), 70, 220),
        diastolic: clamp(Math.round(dbp), 40, 140),
        map: clamp(Math.round(dbp + (sbp - dbp) / 3), 50, 160),
        confidence: clamp01(0.4 + 0.4 * sqi),
        method: "two-site",
        pttMs: ptt,
      };
    }
  }

  // Single-site morphology features
  const feats = extractBPFeatures(pulse, cfg);
  if (!feats) return empty;

  // Linear model: SBP = β0 + Σβi·xi
  // Coefficients are placeholders; per-user cuff calibration provides offset
  const age = ageYears ?? 45;
  const sexFactor = sex === "male" ? 1 : 0;
  let sbp =
    90 +
    0.4 * hr +
    12 * feats.augmentationIndex +
    8 * feats.stiffnessIndex +
    0.15 * (age - 45) +
    2 * sexFactor;
  let dbp =
    55 +
    0.2 * hr +
    6 * feats.augmentationIndex +
    4 * feats.stiffnessIndex +
    0.1 * (age - 45);

  // Per-user cuff calibration offset
  if (cfg.bpOffset) {
    sbp += cfg.bpOffset.systolic;
    dbp += cfg.bpOffset.diastolic;
  }

  const pp = sbp - dbp;
  return {
    systolic: clamp(Math.round(sbp), 70, 220),
    diastolic: clamp(Math.round(dbp), 40, 140),
    map: clamp(Math.round(dbp + pp / 3), 50, 160),
    confidence: clamp01(0.25 + 0.35 * sqi + 0.2 * (cfg.bpOffset ? 1 : 0)),
    method: "single-site",
    pttMs: null,
  };
}

interface BPFeatures {
  systolicUpstrokeTime: number; // fraction of cycle
  widthAt50: number; // fraction of cycle
  augmentationIndex: number;
  stiffnessIndex: number;
  apgRatios: number[];
}

function extractBPFeatures(pulse: number[], cfg: TrPPGConfig): BPFeatures | null {
  const filtered = butterworthBandpass(detrend(pulse), 0.7, 3.0, cfg.fs, true);
  const peaks = detectPeaksSimple(filtered, cfg.fs);
  if (peaks.length < 2) return null;

  // Average cycle
  const intervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    const len = peaks[i] - peaks[i - 1];
    if (len >= cfg.fs * 0.3 && len <= cfg.fs * 2) intervals.push(len);
  }
  if (intervals.length < 2) return null;
  const cycleLen = Math.round(mean(intervals));
  if (cycleLen < 4) return null;

  const cycles: number[][] = [];
  for (let i = 1; i < peaks.length; i++) {
    const start = peaks[i - 1];
    if (start + cycleLen <= filtered.length) cycles.push(filtered.slice(start, start + cycleLen));
  }
  if (cycles.length < 2) return null;

  const avg = new Array<number>(cycleLen).fill(0);
  for (const c of cycles) for (let j = 0; j < cycleLen; j++) avg[j] += c[j] / cycles.length;

  const min = Math.min(...avg);
  const max = Math.max(...avg);
  const range = max - min || 1;
  const norm = avg.map((v) => (v - min) / range);

  const sysIdx = norm.indexOf(max);
  // Systolic upstroke time: from cycle start to peak
  const systolicUpstrokeTime = sysIdx / cycleLen;

  // Width at 50% amplitude
  const halfLevel = 0.5;
  let wStart = 0, wEnd = cycleLen - 1;
  for (let j = 0; j <= sysIdx; j++) if (norm[j] >= halfLevel) { wStart = j; break; }
  for (let j = sysIdx; j < cycleLen; j++) if (norm[j] < halfLevel) { wEnd = j; break; }
  const widthAt50 = (wEnd - wStart) / cycleLen;

  // Dicrotic notch / diastolic peak for augmentation index
  let notchIdx = sysIdx;
  const searchEnd = Math.min(cycleLen, sysIdx + Math.floor(cfg.fs * 0.5));
  for (let j = sysIdx + 1; j < searchEnd; j++) if (norm[j] < norm[notchIdx]) notchIdx = j;
  let diasIdx = notchIdx;
  for (let j = notchIdx + 1; j < cycleLen; j++) if (norm[j] > norm[diasIdx]) diasIdx = j;
  const augmentationIndex = max > 1e-9 ? norm[diasIdx] / norm[sysIdx] : 0;

  // Stiffness index: ΔP/Δt of upstroke
  const stiffnessIndex = systolicUpstrokeTime > 0 ? 1 / systolicUpstrokeTime : 0;

  // APG (second-derivative) ratios
  const d2: number[] = [];
  for (let j = 1; j < cycleLen - 1; j++) d2.push(norm[j + 1] - 2 * norm[j] + norm[j - 1]);
  const apgRatios = d2.length > 0 ? [Math.max(...d2.map(Math.abs)) / (range || 1)] : [0];

  return { systolicUpstrokeTime, widthAt50, augmentationIndex, stiffnessIndex, apgRatios };
}

/** Pulse transit time between face and finger pulse feet (ms). */
function computePTT(facePulse: number[], fingerPulse: number[], cfg: TrPPGConfig): number | null {
  const facePeaks = detectPeaksSimple(butterworthBandpass(detrend(facePulse), 0.7, 3.0, cfg.fs, true), cfg.fs);
  const fingerPeaks = detectPeaksSimple(butterworthBandpass(detrend(fingerPulse), 0.7, 3.0, cfg.fs, true), cfg.fs);
  if (facePeaks.length < 3 || fingerPeaks.length < 3) return null;

  // Match peaks within ±0.5s and compute mean time difference
  const diffs: number[] = [];
  for (const fp of facePeaks) {
    let best = -1, bestDist = Infinity;
    for (const pp of fingerPeaks) {
      const d = Math.abs(pp - fp);
      if (d < bestDist) { bestDist = d; best = pp; }
    }
    if (best >= 0 && bestDist < cfg.fs * 0.5) {
      diffs.push((best - fp) / cfg.fs * 1000);
    }
  }
  if (diffs.length < 3) return null;
  return median(diffs);
}

// ─────────────────────────────────────────────────────────────────────
// Stage 8 — Hemoglobin (absorbance features)
// ─────────────────────────────────────────────────────────────────────

/**
 * Absorbance per channel: A_c = −ln(I_c / I0_c)
 * Features: AC/DC ratios and log-ratios across R, G, B, skin reflectance
 * (DC levels), and HR. Hb = f(features) by regularised regression or
 * gradient boosting with subject-wise cross-validation.
 * Calibrate per device and skin tone.
 */
export function computeHb(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  fHr: number | null,
  cfg: TrPPGConfig,
  sqi: number
): number | null {
  if (fHr === null || sqi < 0.3) return null;

  const dcR = mean(rRaw), dcG = mean(gRaw), dcB = mean(bRaw);
  if (dcR <= 1e-9 || dcG <= 1e-9 || dcB <= 1e-9) return null;

  // Absorbance: A_c = -ln(I_c / I0_c), I0 = max channel DC (reference)
  const i0 = Math.max(dcR, dcG, dcB);
  const aR = -Math.log(dcR / i0);
  const aG = -Math.log(dcG / i0);
  const aB = -Math.log(dcB / i0);

  // AC/DC ratios per channel
  const acR = spectralAmplitude(rRaw, cfg.fs, fHr, 0.15) / dcR;
  const acG = spectralAmplitude(gRaw, cfg.fs, fHr, 0.15) / dcG;
  const acB = spectralAmplitude(bRaw, cfg.fs, fHr, 0.15) / dcB;

  // Log-ratios across channels
  const logRG = Math.log((acR * dcG) / (acG * dcR));
  const logRB = Math.log((acR * dcB) / (acB * dcR));
  const logGB = Math.log((acG * dcB) / (acB * dcG));

  // Feature vector
  const features = [
    aR, aG, aB,           // absorbance
    acR, acG, acB,        // AC/DC ratios
    logRG, logRB, logGB,  // log-ratios
    dcR / i0, dcG / i0, dcB / i0, // skin reflectance (relative DC)
    fHr,                  // HR
  ];

  // Regularised regression (ridge). Coefficients are placeholders;
  // per-device/skin-tone calibration is required.
  const weights = [
    0.5, 0.8, 0.3,   // absorbance R,G,B
    -1.2, 0.4, -0.6, // AC/DC R,G,B
    0.8, 0.5, -0.3,  // log-ratios
    -0.4, -0.6, -0.2, // reflectance
    0.01,            // HR
  ];
  let hb = 14.0; // intercept
  for (let i = 0; i < features.length; i++) hb += weights[i] * features[i];

  // Skin tone covariate
  if (cfg.skinTone && cfg.skinTone > 0) {
    hb += (cfg.skinTone - 3.5) * 0.2;
  }

  return clamp(hb, 8, 22);
}

// ─────────────────────────────────────────────────────────────────────
// Stage 9 — Glucose and creatinine (trained-model interface)
// ─────────────────────────────────────────────────────────────────────

/**
 * No direct optical formula exists. These are y = f(waveform features, HR,
 * HRV, age, sex, BMI) from a trained model. Always compare against a
 * covariate-only baseline model, and evaluate with MAE, Clarke/Parkes
 * error grid (glucose) and eGFR category agreement (creatinine).
 *
 * This interface accepts a trained model function; a covariate-only
 * baseline is provided for comparison.
 */
export interface ModelInput {
  waveformFeatures: number[];
  hr: number | null;
  hrvRmssd: number | null;
  hrvSdnn: number | null;
  age: number | null;
  sex: "male" | "female" | null;
  bmi: number | null;
  spo2: number | null;
  perfusionIndex: number | null;
}

export type TrainedModel = (input: ModelInput) => number | null;

/** Covariate-only baseline (no waveform features) — the null model. */
export function covariateBaseline(input: ModelInput, targetMean: number): number {
  // Baseline: predict population mean adjusted for age/sex
  let pred = targetMean;
  if (input.age !== null) pred += (input.age - 45) * 0.05;
  if (input.sex === "male") pred += 0.3;
  return pred;
}

/** Glucose via trained model (mg/dL). */
export function computeGlucose(
  input: ModelInput,
  model: TrainedModel,
  sqi: number
): number | null {
  if (sqi < 0.4) return null;
  const pred = model(input);
  return pred !== null ? clamp(pred, 40, 400) : null;
}

/** Creatinine via trained model (mg/dL). */
export function computeCreatinine(
  input: ModelInput,
  model: TrainedModel,
  sqi: number
): number | null {
  if (sqi < 0.4) return null;
  const pred = model(input);
  return pred !== null ? clamp(pred, 0.2, 15) : null;
}

/** eGFR from creatinine (CKD-EPI 2021, race-free). */
export function egfr(creatinineMgDl: number | null, ageYears: number | null, sex: "male" | "female" | null): number | null {
  if (creatinineMgDl === null || creatinineMgDl <= 0) return null;
  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const female = sex === "female";
  const k = female ? 0.7 : 0.9;
  const a = female ? -0.241 : -0.302;
  const ratio = creatinineMgDl / k;
  return (
    142 *
    Math.pow(Math.min(ratio, 1), a) *
    Math.pow(Math.max(ratio, 1), -1.2) *
    Math.pow(0.9938, age) *
    (female ? 1.012 : 1)
  );
}

// ─────────────────────────────────────────────────────────────────────
// Stage 10 — Evaluation and fitting
// ─────────────────────────────────────────────────────────────────────

export interface BlandAltmanResult {
  bias: number;
  loaLower: number;
  loaUpper: number;
  sd: number;
  n: number;
}

/**
 * Bland-Altman analysis: d = app − reference; bias = mean(d);
 * LoA = bias ± 1.96·SD(d)
 */
export function blandAltman(app: number[], reference: number[]): BlandAltmanResult | null {
  const n = Math.min(app.length, reference.length);
  if (n < 3) return null;
  const d: number[] = [];
  for (let i = 0; i < n; i++) d.push(app[i] - reference[i]);
  const bias = mean(d);
  const sd = stddev(d);
  return {
    bias,
    loaLower: bias - 1.96 * sd,
    loaUpper: bias + 1.96 * sd,
    sd,
    n,
  };
}

/** MAE = mean|d|; RMSE = sqrt(mean(d²)); ARMS = sqrt(mean(d²)) for SpO2. */
export function errorMetrics(app: number[], reference: number[]): { mae: number; rmse: number; arms: number } | null {
  const n = Math.min(app.length, reference.length);
  if (n < 2) return null;
  let sae = 0, se = 0;
  for (let i = 0; i < n; i++) {
    const d = app[i] - reference[i];
    sae += Math.abs(d);
    se += d * d;
  }
  return {
    mae: sae / n,
    rmse: Math.sqrt(se / n),
    arms: Math.sqrt(se / n),
  };
}

/** Lin's concordance correlation coefficient: CCC = 2ρσxσy / (σx² + σy² + (μx − μy)²) */
export function concordanceCCC(x: number[], y: number[]): number | null {
  const n = Math.min(x.length, y.length);
  if (n < 3) return null;
  const mx = mean(x), my = mean(y);
  const sx = stddev(x), sy = stddev(y);
  if (sx <= 1e-12 || sy <= 1e-12) return null;
  let cov = 0;
  for (let i = 0; i < n; i++) cov += (x[i] - mx) * (y[i] - my);
  cov /= n - 1;
  const rho = cov / (sx * sy);
  return (2 * rho * sx * sy) / (sx * sx + sy * sy + (mx - my) ** 2);
}

/** Clarke error grid zones for glucose (mg/dL). */
export function clarkeErrorGrid(app: number[], reference: number[]): { zone: string; count: number }[] {
  const zones = new Map<string, number>();
  const n = Math.min(app.length, reference.length);
  for (let i = 0; i < n; i++) {
    const a = app[i], r = reference[i];
    const zone = clarkeZone(a, r);
    zones.set(zone, (zones.get(zone) ?? 0) + 1);
  }
  return Array.from(zones.entries()).map(([zone, count]) => ({ zone, count }));
}

function clarkeZone(a: number, r: number): string {
  // Clarke error grid zones A-E
  if (a >= 70 && a <= 180 && r >= 70 && r <= 180) return "A";
  if (a <= 70 && r <= 70) return "A";
  if (a >= 180 && r >= 180) return "A";
  if (a >= 70 && a <= 180 && (r < 70 || r > 180)) return "B";
  if (r >= 70 && r <= 180 && (a < 70 || a > 180)) return "B";
  if (a < 70 && r >= 70 && r <= 180) return "C";
  if (a > 180 && r >= 70 && r <= 180) return "C";
  if (a < 70 && r > 180) return "D";
  if (a > 180 && r < 70) return "D";
  return "E";
}

/**
 * Leave-one-subject-out cross-validation.
 * subjects: array of {app: number[], ref: number[]} per subject.
 * Returns per-subject predictions and aggregate metrics.
 */
export function leaveOneSubjectOut(
  subjects: { app: number[]; ref: number[] }[],
  fitModel: (trainApp: number[][], trainRef: number[][]) => (app: number[]) => number[]
): { predictions: number[]; references: number[]; mae: number; rmse: number; ccc: number | null } {
  const allPred: number[] = [];
  const allRef: number[] = [];

  for (let i = 0; i < subjects.length; i++) {
    const trainApp = subjects.filter((_, j) => j !== i).map((s) => s.app);
    const trainRef = subjects.filter((_, j) => j !== i).map((s) => s.ref);
    const model = fitModel(trainApp, trainRef);
    const pred = model(subjects[i].app);
    allPred.push(...pred);
    allRef.push(...subjects[i].ref);
  }

  const metrics = errorMetrics(allPred, allRef);
  return {
    predictions: allPred,
    references: allRef,
    mae: metrics?.mae ?? 0,
    rmse: metrics?.rmse ?? 0,
    ccc: concordanceCCC(allPred, allRef),
  };
}

// ─────────────────────────────────────────────────────────────────────
// Main TrPPG pipeline orchestrator
// ─────────────────────────────────────────────────────────────────────

export interface TrPPGOptions {
  ageYears?: number | null;
  sex?: "male" | "female" | null;
  bmi?: number | null;
  fingerPulse?: number[]; // second site for PTT
  glucoseModel?: TrainedModel;
  creatinineModel?: TrainedModel;
}

/**
 * Run the complete TrPPG pipeline on front-camera ROI colour means.
 *
 * @param samples Per-frame ROI colour means (R, G, B spatial averages)
 * @param cfg Pipeline configuration
 * @param opts Optional covariates and trained models
 */
export function runTrPPG(samples: FrameSample[], cfg: TrPPGConfig, opts: TrPPGOptions = {}): TrPPGResult {
  // Stage 1: Preprocessing
  const { r, g, b, rN, gN, bN } = preprocessChannels(samples, cfg);

  // Stage 2: Pulse signal
  const { pulse, method } = extractPulse(rN, gN, bN, cfg);

  // Stage 3: HR + SQI (Welch PSD)
  const { hr, sqi, welch } = computeHRAndSQI(pulse, cfg);

  // Stage 4: Respiratory rate (3 modulations, SQI-weighted fusion)
  const resp = computeRespiratory(pulse, g, cfg, sqi);

  // Stage 5: HRV (cubic-spline upsampling, NN intervals, Lomb-Scargle)
  const hrv = computeHRV(pulse, cfg);

  // Stage 6: SpO2 (AC/DC ratio-of-ratios, empirical calibration)
  const { spo2, R: spo2R } = computeSpO2(r, g, b, welch.peakFreq, cfg, sqi);

  // Stage 7: Blood pressure (single-site morphology + two-site PTT)
  const bp = computeBP(pulse, cfg, hr, sqi, opts.ageYears ?? null, opts.sex ?? null, opts.fingerPulse);

  // Stage 8: Hemoglobin (absorbance features)
  const hb = computeHb(r, g, b, welch.peakFreq, cfg, sqi);

  // Stage 9: Glucose / creatinine (trained-model interface)
  const perfusionIndex = computePerfusionIndex(pulse, g, cfg);
  const modelInput: ModelInput = {
    waveformFeatures: pulse.slice(-Math.min(pulse.length, cfg.fs * 10)),
    hr,
    hrvRmssd: hrv.rmssd,
    hrvSdnn: hrv.sdnn,
    age: opts.ageYears ?? null,
    sex: opts.sex ?? null,
    bmi: opts.bmi ?? null,
    spo2,
    perfusionIndex,
  };
  const glucose = opts.glucoseModel ? computeGlucose(modelInput, opts.glucoseModel, sqi) : null;
  const creatinine = opts.creatinineModel ? computeCreatinine(modelInput, opts.creatinineModel, sqi) : null;

  // Diagnostics
  const ibi = interBeatIntervals(pulse, cfg.fs);
  const diagnostics: TrPPGDiagnostics = {
    welchFreqs: welch.freqs,
    welchPsd: welch.psd,
    hrPeakFreq: welch.peakFreq,
    snrDb: welch.snrDb,
    respPeaks: resp.peaks,
    ibiMs: ibi,
    nnMs: hrv.nnIntervals,
    calibrationUsed: cfg.spo2Cal !== undefined || cfg.hbCal !== undefined || cfg.bpOffset !== undefined,
  };

  return {
    hr,
    sqi,
    pulse,
    pulseMethod: method,
    respRate: resp.respRate,
    respConfidence: resp.confidence,
    hrv,
    spo2,
    spo2R,
    bp,
    hb,
    glucose,
    creatinine,
    diagnostics,
  };
}

/** Perfusion index (AC/DC) of the pulse against the green DC reference. */
function computePerfusionIndex(pulse: number[], greenRaw: number[], cfg: TrPPGConfig): number {
  const filtered = butterworthBandpass(detrend(pulse), cfg.cardiacLow, cfg.cardiacHigh, cfg.fs, true);
  const dc = mean(greenRaw);
  if (dc <= 1e-9) return 0;
  const ac = rms(filtered);
  return (ac / dc) * 100;
}

// ─────────────────────────────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────────────────────────────

function mean(x: number[]): number {
  if (x.length === 0) return 0;
  return x.reduce((a, b) => a + b, 0) / x.length;
}

function stddev(x: number[]): number {
  if (x.length < 2) return 0;
  const m = mean(x);
  const v = x.reduce((a, b) => a + (b - m) * (b - m), 0) / (x.length - 1);
  return Math.sqrt(v);
}

function median(x: number[]): number {
  if (x.length === 0) return 0;
  const s = x.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function rms(x: number[]): number {
  if (x.length === 0) return 0;
  let sq = 0;
  for (const v of x) sq += v * v;
  return Math.sqrt(sq / x.length);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

/** Butterworth bandpass (2nd-order HP + LP biquads, zero-phase filtfilt). */
function butterworthBandpass(signal: number[], low: number, high: number, fs: number, zeroPhase = true): number[] {
  if (signal.length < 10 || !(fs > 0) || !(low > 0) || high <= low || high >= fs / 2) {
    return signal.slice();
  }
  const hp = highpassBiquad(low, fs);
  const lp = lowpassBiquad(high, fs);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
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

interface BiquadCoeffs { b0: number; b1: number; b2: number; a1: number; a2: number }

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
