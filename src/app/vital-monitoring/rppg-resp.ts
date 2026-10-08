export interface RPPGRespResult {
  respRate: number | null;
  waveform: number[];
  confidence: number;
  method: string;
}

export interface RPPGRespConfig {
  fs: number;
  windowSec: number;
  bandLow: number;
  bandHigh: number;
}

const DEFAULT_CONFIG: RPPGRespConfig = {
  fs: 30,
  windowSec: 30,
  bandLow: 0.1,
  bandHigh: 0.5
};

function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return signal.slice();
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const yMean = signal.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - xMean) * (signal[i] - yMean);
    den += (x[i] - xMean) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = yMean - slope * xMean;
  return signal.map((y, i) => y - (slope * i + intercept));
}

function butterworthBandpass(signal: number[], low: number, high: number, fs: number): number[] {
  if (signal.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
    return signal.slice();
  }
  const w0Low = (2 * Math.PI * low) / fs;
  const w0High = (2 * Math.PI * high) / fs;
  const cwLow = Math.cos(w0Low);
  const swLow = Math.sin(w0Low);
  const cwHigh = Math.cos(w0High);
  const swHigh = Math.sin(w0High);
  const q = Math.SQRT1_2;
  const alphaLow = swLow / (2 * q);
  const alphaHigh = swHigh / (2 * q);
  const a0Low = 1 + alphaLow;
  const a0High = 1 + alphaHigh;
  const hp_b0 = (1 + cwLow) / 2 / a0Low;
  const hp_b1 = -(1 + cwLow) / a0Low;
  const hp_b2 = (1 + cwLow) / 2 / a0Low;
  const hp_a1 = (-2 * cwLow) / a0Low;
  const hp_a2 = (1 - alphaLow) / a0Low;
  const lp_b0 = (1 - cwHigh) / 2 / a0High;
  const lp_b1 = (1 - cwHigh) / a0High;
  const lp_b2 = (1 - cwHigh) / 2 / a0High;
  const lp_a1 = (-2 * cwHigh) / a0High;
  const lp_a2 = (1 - alphaHigh) / a0High;

  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  let x1 = dc, x2 = dc, y1 = 0, y2 = 0;
  const hpOut: number[] = new Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    const y = hp_b0 * signal[i] + hp_b1 * x1 + hp_b2 * x2 - hp_a1 * y1 - hp_a2 * y2;
    hpOut[i] = y;
    x2 = x1; x1 = signal[i];
    y2 = y1; y1 = y;
  }

  x1 = 0; x2 = 0; y1 = 0; y2 = 0;
  const lpOut: number[] = new Array(signal.length);
  for (let i = 0; i < hpOut.length; i++) {
    const y = lp_b0 * hpOut[i] + lp_b1 * x1 + lp_b2 * x2 - lp_a1 * y1 - lp_a2 * y2;
    lpOut[i] = y;
    x2 = x1; x1 = hpOut[i];
    y2 = y1; y1 = y;
  }

  const rev = lpOut.reverse();
  x1 = 0; x2 = 0; y1 = 0; y2 = 0;
  const hpOut2: number[] = new Array(rev.length);
  for (let i = 0; i < rev.length; i++) {
    const y = hp_b0 * rev[i] + hp_b1 * x1 + hp_b2 * x2 - hp_a1 * y1 - hp_a2 * y2;
    hpOut2[i] = y;
    x2 = x1; x1 = rev[i];
    y2 = y1; y1 = y;
  }

  x1 = 0; x2 = 0; y1 = 0; y2 = 0;
  const lpOut2: number[] = new Array(hpOut2.length);
  for (let i = 0; i < hpOut2.length; i++) {
    const y = lp_b0 * hpOut2[i] + lp_b1 * x1 + lp_b2 * x2 - lp_a1 * y1 - lp_a2 * y2;
    lpOut2[i] = y;
    x2 = x1; x1 = hpOut2[i];
    y2 = y1; y1 = y;
  }

  return lpOut2.reverse();
}

function autocorrelate(signal: number[]): number[] {
  const n = signal.length;
  const m = signal.reduce((a, b) => a + b, 0) / n;
  const centered = signal.map(v => v - m);
  const norm = centered.reduce((a, b) => a + b * b, 0) || 1;
  const out: number[] = [];
  for (let lag = 0; lag < n; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += centered[i] * centered[i + lag];
    out.push(s / norm);
  }
  return out;
}

function findPeaks(signal: number[], minDist: number): number[] {
  const peaks: number[] = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

export function estimateRespRateRPPG(
  ppg: number[],
  fs: number = 30,
  config: Partial<RPPGRespConfig> = {}
): RPPGRespResult {
  const cfg = { ...DEFAULT_CONFIG, ...config, fs };
  
  if (ppg.length < cfg.fs * 10) {
    return { respRate: null, waveform: [], confidence: 0, method: 'rppg-insufficient-data' };
  }

  const windowLen = Math.min(ppg.length, Math.floor(cfg.fs * cfg.windowSec));
  const signal = ppg.slice(-windowLen);

  const detrended = detrend(signal);
  const bandpassed = butterworthBandpass(detrended, cfg.bandLow, cfg.bandHigh, cfg.fs);

  const ac = autocorrelate(bandpassed);
  const minLag = Math.floor(cfg.fs / cfg.bandHigh);
  const maxLag = Math.floor(cfg.fs / cfg.bandLow);
  let bestLag = -1;
  let bestVal = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, ac.length - 1); lag++) {
    if (ac[lag] > bestVal) {
      bestVal = ac[lag];
      bestLag = lag;
    }
  }

  let respRate = 0;
  let method = 'rppg-autocorr';
  let confidence = 0;

  if (bestLag > 0) {
    respRate = (60 * cfg.fs) / bestLag;
    confidence = Math.max(0, Math.min(1, bestVal));
    method = 'rppg-autocorr';
  }

  const n = bandpassed.length;
  const freqs: number[] = [];
  const psd: number[] = [];
  const win = new Array(n);
  let winSqSum = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    win[i] = w;
    winSqSum += w * w;
  }
  const wNorm = winSqSum / n || 1;
  for (let k = 0; k < Math.floor(n / 2); k++) {
    let real = 0, imag = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      real += bandpassed[i] * win[i] * Math.cos(angle);
      imag -= bandpassed[i] * win[i] * Math.sin(angle);
    }
    freqs.push((k * cfg.fs) / n);
    psd.push((real * real + imag * imag) / n / wNorm);
  }

  let bestFreq = 0;
  let bestPower = -Infinity;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= cfg.bandLow && freqs[i] <= cfg.bandHigh) {
      if (psd[i] > bestPower) {
        bestPower = psd[i];
        bestFreq = freqs[i];
      }
    }
  }

  if (bestFreq > 0) {
    const fftRate = bestFreq * 60;
    if (respRate === 0 || Math.abs(fftRate - respRate) < 8) {
      respRate = respRate === 0 ? fftRate : 0.5 * (respRate + fftRate);
      method = 'rppg-autocorr+FFT';
      const totalPower = psd.reduce((a, b) => a + b, 0) || 1;
      confidence = Math.max(confidence, Math.min(1, bestPower / totalPower * 5));
    } else {
      respRate = 0.5 * (respRate + fftRate);
      method = 'rppg-autocorr+FFT(fused)';
    }
  }

  if (respRate < 5 || respRate > 50) {
    return { respRate: null, waveform: bandpassed, confidence: 0, method };
  }

  const minDist = Math.floor(cfg.fs / 0.5);
  const peaks = findPeaks(bandpassed, minDist);
  if (peaks.length >= 3) {
    const intervals: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push((peaks[i] - peaks[i - 1]) / cfg.fs);
    }
    const meanInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    if (meanInterval > 0) {
      const peakRate = 60 / meanInterval;
      if (Math.abs(peakRate - respRate) < 5) {
        respRate = 0.5 * (respRate + peakRate);
        method += '+peaks';
      }
    }
  }

  return {
    respRate: Math.round(respRate * 10) / 10,
    waveform: bandpassed,
    confidence: Math.round(confidence * 100) / 100,
    method
  };
}

export function extractRPPGRespiration(
  red: number[],
  green: number[],
  blue: number[],
  fs: number = 30
): RPPGRespResult {
  const n = Math.min(red.length, green.length, blue.length);
  if (n < fs * 10) {
    return { respRate: null, waveform: [], confidence: 0, method: 'rppg-insufficient-data' };
  }

  const pos = posPulse(red, green, blue, Math.max(16, Math.round(fs)));
  if (pos.length < fs * 10) {
    return { respRate: null, waveform: [], confidence: 0, method: 'rppg-pos-failed' };
  }

  return estimateRespRateRPPG(pos, fs);
}

function posPulse(red: number[], green: number[], blue: number[], winLen = 120): number[] {
  const n = Math.min(red.length, green.length, blue.length);
  if (n < winLen || winLen < 8) return [];
  const hop = Math.max(1, Math.floor(winLen / 2));
  const sqrt6 = Math.sqrt(6);
  const acc = new Array<number>(n).fill(0);
  const wsum = new Array<number>(n).fill(0);
  for (let start = 0; start + winLen <= n; start += hop) {
    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < winLen; i++) {
      mr += red[start + i];
      mg += green[start + i];
      mb += blue[start + i];
    }
    mr /= winLen; mg /= winLen; mb /= winLen;
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
    let m1 = 0, m2 = 0;
    for (let i = 0; i < winLen; i++) {
      s1[i] = (cnG[i] - cnB[i]) / sqrt6;
      s2[i] = (-2 * cnR[i] + cnG[i] + cnB[i]) / sqrt6;
      m1 += s1[i]; m2 += s2[i];
    }
    m1 /= winLen; m2 /= winLen;
    let v1 = 0, v2 = 0;
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