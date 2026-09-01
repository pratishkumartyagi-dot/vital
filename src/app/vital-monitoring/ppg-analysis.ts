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

function bandpass(signal: number[], low: number, high: number, fs: number): number[] {
  const n = signal.length;
  if (n < 10) return signal.slice();
  const result: number[] = new Array(n).fill(0);
  const tauH = 1 / (2 * Math.PI * high);
  const tauL = 1 / (2 * Math.PI * low);
  const alphaH = tauH / (tauH + 1 / fs);
  const alphaL = tauL / (tauL + 1 / fs);
  let prevH: number[] = new Array(n).fill(0);
  let prevL: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    prevH[i] = signal[i];
    prevL[i] = signal[i];
  }
  for (let i = 1; i < n; i++) {
    prevH[i] = alphaH * prevH[i] + (1 - alphaH) * prevH[i - 1];
  }
  for (let i = 1; i < n; i++) {
    prevL[i] = alphaL * prevL[i] + (1 - alphaL) * prevL[i - 1];
  }
  for (let i = 0; i < n; i++) {
    result[i] = prevH[i] - prevL[i];
  }
  return result;
}

function butterBandpass(signal: number[], low: number, high: number, fs: number, order = 2): number[] {
  let x = signal;
  for (let i = 0; i < order; i++) x = bandpass(x, low, high, fs);
  return x;
}

function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): { peaks: Peak[]; filtered: number[] } {
  const detrended = detrend(signal);
  const filtered = bandpass(detrended, 0.7, 3.5, fs);
  const peaks: Peak[] = [];
  const minDist = Math.max(1, Math.floor((60 / maxBpm) * fs));
  const fMin = Math.min(...filtered);
  const fMax = Math.max(...filtered);
  const fRange = fMax - fMin || 1;
  const minHeight = fMin + 0.15 * fRange;
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
  const meanFiltered = mean(filtered);
  let ac = 0;
  for (const v of filtered) ac += Math.abs(v - meanFiltered);
  ac /= filtered.length;
  const dc = mean(ppg);
  const quality = dc > 0 ? Math.min(1, (ac / dc) * 50) : 0;
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
  const lowpass = butterBandpass(detrended, 0.1, 0.5, fs, 2);
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
}

export function analyzeVitals(ppg: number[], fs = 30): VitalAnalysis {
  const heart = analyzeHeartRate(ppg, fs);
  const resp = analyzeRespRate(ppg, fs);
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
  };
}
