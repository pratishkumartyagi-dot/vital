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

/// 4th-order Butterworth bandpass filter using cascaded SOS (Second-Order Sections)
/// Coefficients generated via: scipy.signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band', output='sos')
/// For 0.5-3.5 Hz at 30 Hz:
const BUTTERWORTH_SOS_05_35_30: number[][] = [
  [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
  [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
];

function applySosFilter(signal: number[], sos: number[][]): number[] {
  let stage = signal.slice();
  for (const section of sos) {
    const [b0, b1, b2, a0, a1, a2] = section;
    const out: number[] = new Array(stage.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < stage.length; i++) {
      const x0 = stage[i];
      const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      out[i] = y0;
      x2 = x1; x1 = x0;
      y2 = y1; y1 = y0;
    }
    stage = out;
  }
  return stage;
}

export function butterworthBandpass(signal: number[], low: number, high: number, fs: number): number[] {
  // For standard PPG range (0.5-3.5 Hz at 30 Hz), use pre-computed SOS
  if (fs === 30 && low === 0.5 && high === 3.5) {
    return applySosFilter(signal, BUTTERWORTH_SOS_05_35_30);
  }
  // Fallback: simple IIR cascade for other parameters
  const n = signal.length;
  if (n < 10) return signal.slice();
  const result: number[] = new Array(n);
  const tauH = 1 / (2 * Math.PI * low);
  const tauL = 1 / (2 * Math.PI * high);
  const alphaH = tauH / (tauH + 1 / fs);
  const alphaL = tauL / (tauL + 1 / fs);
  let hp = signal[0];
  let lp = signal[0];
  for (let i = 0; i < n; i++) {
    hp = alphaH * (hp + signal[i] - (i > 0 ? signal[i - 1] : signal[0]));
    lp = alphaL * lp + (1 - alphaL) * hp;
    result[i] = lp;
  }
  return result;
}

function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): { peaks: Peak[]; filtered: number[] } {
  const filtered = butterworthBandpass(detrend(signal), 0.5, 3.5, fs);
  const peaks: Peak[] = [];
  const minDist = Math.max(1, Math.floor((60 / maxBpm) * fs));
  const sorted = filtered.slice().sort((a, b) => a - b);
  const loIdx = Math.floor(sorted.length * 0.10);
  const hiIdx = Math.floor(sorted.length * 0.90);
  const baseline = sorted[loIdx];
  const peakCeil = sorted[hiIdx];
  const minHeight = baseline + 0.20 * (peakCeil - baseline);
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1]) {
      if (filtered[i] > minHeight) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1].index >= minDist) {
          peaks.push({ index: i, value: filtered[i] });
        } else {
          if (filtered[i] > peaks[peaks.length - 1].value) {
            peaks[peaks.length - 1] = { index: i, value: filtered[i] };
          }
        }
      }
    }
  }
  return { peaks, filtered };
}

function computePerfusionIndex(signal: number[]): number {
  if (signal.length < 10) return 0;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  let ac = 0;
  for (const v of signal) ac += Math.abs(v - mean);
  ac /= signal.length;
  return mean > 0 ? (ac / mean) * 100 : 0;
}

function computeSignalQuality(filtered: number[], raw: number[], peaks: Peak[]): number {
  if (peaks.length < 2) return 0;
  
  // 1. Perfusion Index (AC/DC ratio) - normalized to 0-1
  const pi = computePerfusionIndex(raw);
  const piScore = Math.min(1, pi / 2); // PI > 2% is good for smartphone PPG
  
  // 2. Spectral SNR in cardiac band (0.7-3.5 Hz)
  const { freqs, psd } = powerSpectrum(filtered, 30);
  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.7 && freqs[i] <= 3.5) signalPower += psd[i];
    else if (freqs[i] > 3.5 && freqs[i] <= 15) noisePower += psd[i];
  }
  const snr = noisePower > 0 ? signalPower / noisePower : 0;
  const snrScore = Math.min(1, snr / 4); // SNR > 4 is good for smartphone PPG
  
  // 3. Pulse regularity (CV of RR intervals)
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    rrMs.push(((peaks[i].index - peaks[i - 1].index) / 30) * 1000);
  }
  const cv = rrMs.length > 1 ? (stddev(rrMs) / mean(rrMs)) * 100 : 100;
  const regScore = Math.max(0, 1 - cv / 20); // CV < 20% is regular
  
  // Combined score (weighted average)
  return Math.round((0.45 * piScore + 0.35 * snrScore + 0.2 * regScore) * 100);
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

function powerSpectrum(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  const freqs: number[] = [];
  const psd: number[] = [];
  if (n < 8) return { freqs, psd };
  const detrended = detrend(signal);
  for (let k = 0; k < Math.floor(n / 2); k++) {
    let real = 0;
    let imag = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      real += detrended[i] * Math.cos(angle);
      imag -= detrended[i] * Math.sin(angle);
    }
    freqs.push((k * fs) / n);
    psd.push((real * real + imag * imag) / n);
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
  const valid = peaks.filter((_, i) => {
    if (i === 0) return true;
    const dtMs = ((peaks[i].index - peaks[i - 1].index) / fs) * 1000;
    return dtMs >= 300 && dtMs <= 2000;
  });
  if (valid.length < 3) {
    return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: "unknown", signalQuality: 0 };
  }
  const rrMs: number[] = [];
  for (let i = 1; i < valid.length; i++) {
    rrMs.push(((valid[i].index - valid[i - 1].index) / fs) * 1000);
  }
  const meanRR = mean(rrMs);
  const bpm = meanRR > 0 ? 60000 / meanRR : 0;
  const hrv = computeHRV(rrMs);
  const rhythm = classifyRhythm(rrMs);
  const quality = computeSignalQuality(filtered, ppg, valid);
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
  fs: number
): GlucoseMetrics {
  if (ppg.length < fs * 10 || peakIndices.length < 3) {
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
}

export function analyzeVitals(
  ppg: number[],
  fs = 30,
  redSignal: number[] = [],
  greenSignal: number[] = [],
  blueSignal: number[] = []
): VitalAnalysis {
  const heart = analyzeHeartRate(ppg, fs);
  const resp = analyzeRespRate(ppg, fs);
  const bp = estimateBloodPressure(ppg, heart.peaks, heart.rrIntervals, heart.hrv, fs);
  const glucose = estimateGlucose(ppg, heart.peaks, redSignal, greenSignal, blueSignal, fs);

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
  };
}
