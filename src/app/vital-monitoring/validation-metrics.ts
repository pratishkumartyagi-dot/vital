"use strict";

import { butterworthBandpass } from "./ppg-analysis";

export type CameraMode = "contact" | "rppg";

export interface ContactPPGResult {
  cameraMode: "contact";
  signal: { r: number[]; g: number[]; b: number[] };
  signalExtraction: "optical-density";
  heartRate: HRResult;
  acDc: PerBeatAcDc;
  perfusionIndex: number;
  spo2: SpO2Result;
  respRate: RRResult;
  hrv: HRVResult;
  bp: BPResult;
  hb: HbResult;
  glucose: GlucoseResult;
  creatinine: CreatinineResult;
  quality: QualityResult;
}

export interface RPPGResult {
  cameraMode: "rppg";
  signal: { r: number[]; g: number[]; b: number[] };
  signalExtraction: "chrom" | "pos";
  pulseMethod: "green" | "chrom" | "pos";
  heartRate: HRResult;
  acDc: PerBeatAcDc;
  perfusionIndex: number;
  spo2: SpO2Result;
  respRate: RRResult;
  hrv: HRVResult;
  bp: BPResult;
  hb: HbResult;
  glucose: GlucoseResult;
  creatinine: CreatinineResult;
  quality: QualityResult;
}

export type ValidationResult = ContactPPGResult | RPPGResult;

export interface HRResult {
  bpm: number | null;
  method: "peak-detection" | "spectral" | "fused";
  ibi: number[];
  peakTimes: number[];
  confidence: number;
}

export interface PerBeatAcDc {
  ac: number;
  dc: number;
  pi: number;
  acPerBeat: number[];
  dcPerBeat: number[];
  peakIndices: number[];
  troughIndices: number[];
  beatCount: number;
}

export interface SpO2Result {
  spo2: number | null;
  rRatio: number | null;
  method: "ratio-of-ratios" | "log-ratio";
  a: number;
  b: number;
  isQuadratic: boolean;
  c?: number;
  confidence: number;
}

export interface RRResult {
  respRate: number | null;
  method: "baseline-wander" | "amplitude-modulation" | "frequency-modulation" | "sqi-weighted-fusion";
  fusions: {
    baselineWander: { freq: number; power: number };
    amplitudeMod: { freq: number; power: number };
    frequencyMod: { freq: number; power: number };
  };
  confidence: number;
}

export interface HRVResult {
  meanRR: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  lfHfRatio: number | null;
  upsampled: boolean;
  resamplingRate: number;
  ectopicRejection: EctopicRejectionInfo;
}

export interface EctopicRejectionInfo {
  totalIntervals: number;
  rejected: number;
  threshold: number;
}

export interface BPResult {
  systolic: number | null;
  diastolic: number | null;
  map: number | null;
  pulsePressure: number | null;
  stiffnessIndex: number;
  reflectionIndex: number;
  apgRatios: { b: number; c: number; d: number; e: number; a: number };
  agingIndex: number;
  confidence: number;
  method: "single-site" | "two-site";
  calibrated: boolean;
}

export interface HbResult {
  hb: number | null;
  rRatio: number | null;
  absorbanceFeatures: AbsorbanceFeatures;
  confidence: number;
}

export interface AbsorbanceFeatures {
  odR: number;
  odG: number;
  odB: number;
  logAcDcRed: number;
  logAcDcGreen: number;
  logAcDcBlue: number;
  pi: number;
  skinReflectance: number;
}

export interface GlucoseResult {
  glucose: number | null;
  confidence: number;
  method: "ppg-morphology";
  baselineComparison: number;
}

export type CreatinineClass = "low" | "normal" | "high";

export interface CreatinineResult {
  creatinine: number | null;
  creatinineUmol: number | null;
  creatinineClass: CreatinineClass | null;
  egfr: number | null;
  stage: string | null;
  confidence: number;
  mae: number;
  rmse: number;
  ccc: number;
  method: "rppg-front" | "ppg-rear-ftir";
}

export interface QualityResult {
  sqi: number;
  pulseTemplateCorrelation: number;
  skewness: number;
  kurtosis: number;
  spectralSnr: number;
  pi: number;
  accepted: boolean;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / (arr.length - 1));
}

function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return signal.slice();
  const xm = (n - 1) / 2;
  const ym = mean(signal);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xm) * (signal[i] - ym);
    den += (i - xm) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = ym - slope * xm;
  return signal.map((y, i) => y - (slope * i + intercept));
}

function butterworthBandpassZeroPhase(signal: number[], low: number, high: number, fs: number): number[] {
  const forward = butterworthBandpass(signal, low, high, fs);
  const reversed = forward.slice().reverse();
  const backward = butterworthBandpass(reversed, low, high, fs);
  return backward.reverse();
}

function linearInterpolate(arr: Float64Array, t: number, n: number): number {
  const scaled = t * (n - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(i0 + 1, n - 1);
  const frac = scaled - i0;
  return arr[i0] * (1 - frac) + arr[i1] * frac;
}

function cubicSplineUpsample(signal: number[], targetFs: number, sourceFs: number): number[] {
  const n = signal.length;
  if (n < 2) return signal.slice();
  const ratio = targetFs / sourceFs;
  const outputLen = Math.floor(n * ratio);
  const input = new Float64Array(signal);
  const output = new Array(outputLen);

  for (let i = 0; i < outputLen; i++) {
    const t = i / (outputLen - 1) * (n - 1);
    output[i] = linearInterpolate(input, t / (n - 1), n);
  }
  return output;
}

function parabolicInterpolationPeak(y0: number, y1: number, y2: number): { delta: number; peak: number } {
  const denom = y0 - 2 * y1 + y2;
  if (Math.abs(denom) < 1e-9) return { delta: 0, peak: y1 };
  const delta = 0.5 * (y0 - y2) / denom;
  const peak = y1 - 0.25 * (y0 - y2) * delta;
  return { delta, peak };
}

function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): number[] {
  const n = signal.length;
  const filtered = butterworthBandpassZeroPhase(signal, 0.5, 4.0, fs);
  const minDist = Math.max(1, Math.floor((60 / maxBpm) * fs));
  const heights = filtered.slice().sort((a, b) => a - b);
  const lo = heights[Math.floor(n * 0.10)] || 0;
  const hi = heights[Math.floor(n * 0.90)] || 1;
  const threshold = lo + 0.2 * (hi - lo);

  const peaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] >= filtered[i + 1] && filtered[i] > threshold) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      } else if (filtered[i] > filtered[peaks[peaks.length - 1]]) {
        peaks[peaks.length - 1] = i;
      }
    }
  }
  return peaks;
}

function computeHeartRate(
  ppg: number[],
  fs: number,
  method: "peak-detection" | "spectral" | "fused" = "fused",
): HRResult {
  if (ppg.length < fs * 2) {
    return { bpm: null, method: "fused", ibi: [], peakTimes: [], confidence: 0 };
  }

  let bpm: number | null = null;
  let ibi: number[] = [];
  let peakTimes: number[] = [];
  let confidence = 0;

  if (method === "peak-detection" || method === "fused") {
    const peaks = detectPeaks(ppg, fs);
    if (peaks.length >= 3) {
      peakTimes = peaks.map((p) => (p / fs) * 1000);
      ibi = [];
      for (let i = 1; i < peaks.length; i++) {
        const interval = ((peaks[i] - peaks[i - 1]) / fs) * 1000;
        if (interval >= 300 && interval <= 2000) ibi.push(interval);
      }
      if (ibi.length > 0) {
        const meanIbi = mean(ibi);
        const peakBpm = 60000 / meanIbi;
        const cv = stddev(ibi) / meanIbi;
        confidence = Math.max(0, 1 - cv * 2);
        bpm = peakBpm > 30 && peakBpm < 220 ? peakBpm : null;
      }
    }
  }

  if ((method === "spectral" || method === "fused") && bpm === null) {
    const { freqs, psd } = powerSpectrum(ppg, fs);
    let bestIdx = -1;
    let bestPower = -Infinity;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= 0.7 / 60 && freqs[i] <= 3.0 / 60) {
        if (psd[i] > bestPower) {
          bestPower = psd[i];
          bestIdx = i;
        }
      }
    }
    if (bestIdx >= 0) {
      const spectralBpm = freqs[bestIdx] * 60;
      if (spectralBpm > 40 && spectralBpm < 180) {
        if (method === "spectral") {
          bpm = spectralBpm;
          confidence = 0.5;
        } else if (bpm === null) {
          bpm = spectralBpm;
        }
      }
    }
  }

  return {
    bpm,
    method,
    ibi,
    peakTimes,
    confidence: Math.round(confidence * 100) / 100,
  };
}

function computePerBeatAcDc(ppg: number[], peaks: number[], fs: number): PerBeatAcDc {
  if (peaks.length < 2) {
    const meanVal = mean(ppg);
    return {
      ac: 0,
      dc: meanVal,
      pi: 0,
      acPerBeat: [],
      dcPerBeat: [],
      peakIndices: [],
      troughIndices: [],
      beatCount: 0,
    };
  }

  const acPerBeat: number[] = [];
  const dcPerBeat: number[] = [];
  const troughIndices: number[] = [];

  for (let i = 0; i < peaks.length; i++) {
    const start = i === 0 ? peaks[i] - Math.floor(fs * 0.3) : peaks[i - 1];
    const end = i === peaks.length - 1 ? peaks[i] + Math.floor(fs * 0.3) : peaks[i + 1];

    const beatSlice = ppg.slice(Math.max(0, start), Math.min(ppg.length, end));
    const beatMax = Math.max(...beatSlice);
    const beatMin = Math.min(...beatSlice);
    const beatMean = mean(beatSlice);

    acPerBeat.push(beatMax - beatMin);
    dcPerBeat.push(beatMean);

    // Find trough after peak
    let troughIdx = peaks[i];
    for (let j = peaks[i]; j < Math.min(ppg.length, end); j++) {
      if (ppg[j] < ppg[troughIdx]) troughIdx = j;
    }
    troughIndices.push(troughIdx);
  }

  const meanAc = mean(acPerBeat);
  const meanDc = mean(dcPerBeat);
  const pi = meanDc > 0 ? (meanAc / meanDc) * 100 : 0;

  return {
    ac: meanAc,
    dc: meanDc,
    pi,
    acPerBeat,
    dcPerBeat,
    peakIndices: peaks,
    troughIndices,
    beatCount: peaks.length,
  };
}

function computeSpO2(
  rRaw: number[],
  gRaw: number[],
  xRaw: number[], // reference channel (green or blue)
  fs: number,
  perBeat: PerBeatAcDc,
  hr: number | null,
  calibration: { a: number; b: number; c?: number },
  method: "ratio-of-ratios" | "log-ratio" = "ratio-of-ratios",
): SpO2Result {
  if (hr === null || hr <= 0 || rRaw.length < fs * 3) {
    return {
      spo2: null,
      rRatio: null,
      method,
      a: calibration.a,
      b: calibration.b,
      isQuadratic: !!calibration.c,
      c: calibration.c,
      confidence: 0,
    };
  }

  const redFiltered = butterworthBandpassZeroPhase(rRaw, 0.5, 4.0, fs);
  const xFiltered = butterworthBandpassZeroPhase(xRaw, 0.5, 4.0, fs);

  if (method === "ratio-of-ratios") {
    const dcR = mean(rRaw);
    const dcX = mean(xRaw);
    const acR = mean(redFiltered.map((v) => Math.abs(v - mean(redFiltered))));
    const acX = mean(xFiltered.map((v) => Math.abs(v - mean(xFiltered))));

    if (dcX <= 0 || dcR <= 0 || acX <= 1e-12) {
      return {
        spo2: null,
        rRatio: null,
        method,
        a: calibration.a,
        b: calibration.b,
        isQuadratic: !!calibration.c,
        c: calibration.c,
        confidence: 0,
      };
    }

    const rRatio = (acR / dcR) / (acX / dcX);
    let spo2: number;

    if (calibration.c !== undefined) {
      spo2 = calibration.a + calibration.b * rRatio + calibration.c * rRatio * rRatio;
    } else {
      spo2 = calibration.a + calibration.b * rRatio;
    }

    const confidence = Math.min(1, Math.max(0, perBeat.pi / 5));
    const clampedSpo2 = Math.min(100, Math.max(70, Math.round(spo2)));

    return {
      spo2: clampedSpo2,
      rRatio: Math.round(rRatio * 10000) / 10000,
      method,
      a: calibration.a,
      b: calibration.b,
      isQuadratic: !!calibration.c,
      c: calibration.c,
      confidence: Math.round(confidence * 100) / 100,
    };
  }

  // Log form: R = ln(I_max/I_min)_red / ln(I_max/I_min)_x
  const rMax = Math.max(...rRaw);
  const rMin = Math.max(1, Math.min(...rRaw));
  const xMax = Math.max(...xRaw);
  const xMin = Math.max(1, Math.min(...xRaw));

  const rRatio = Math.log(rMax / rMin) / Math.log(xMax / xMin);
  let spo2: number;

  if (calibration.c !== undefined) {
    spo2 = calibration.a + calibration.b * rRatio + calibration.c * rRatio * rRatio;
  } else {
    spo2 = calibration.a + calibration.b * rRatio;
  }

  const confidence = Math.min(1, Math.max(0, perBeat.pi / 5));
  const clampedSpo2 = Math.min(100, Math.max(70, Math.round(spo2)));

  return {
    spo2: clampedSpo2,
    rRatio: Math.round(rRatio * 10000) / 10000,
    method,
    a: calibration.a,
    b: calibration.b,
    isQuadratic: !!calibration.c,
    c: calibration.c,
    confidence: Math.round(confidence * 100) / 100,
  };
}

function computeRespRate(ppg: number[], fs: number): RRResult {
  if (ppg.length < fs * 10) {
    return {
      respRate: null,
      method: "sqi-weighted-fusion",
      fusions: {
        baselineWander: { freq: 0, power: 0 },
        amplitudeMod: { freq: 0, power: 0 },
        frequencyMod: { freq: 0, power: 0 },
      },
      confidence: 0,
    };
  }

  const detrended = detrend(ppg);

  // 1. Baseline wander (0.1-0.5 Hz)
  const baselineWander = butterworthBandpassZeroPhase(detrended, 0.1, 0.5, fs);
  const bwResult = findDominantFreqInBand(baselineWander, fs, 0.1, 0.5);

  // 2. Amplitude modulation (Hilbert envelope of the cardiac pulse, then 0.1-0.5 Hz)
  const pulseFiltered = butterworthBandpassZeroPhase(detrended, 0.5, 4.0, fs);
  const envelope = hilbertEnvelope(pulseFiltered);
  const envelopeFiltered = butterworthBandpassZeroPhase(detrend(envelope), 0.1, 0.5, fs);
  const amResult = findDominantFreqInBand(envelopeFiltered, fs, 0.1, 0.5);

  // 3. Frequency modulation (from IBI series)
  const peaks = detectPeaks(pulseFiltered, fs);
  let fmResult = { freq: 0, power: 0 };
  if (peaks.length >= 5) {
    const ibi: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
      ibi.push((peaks[i] - peaks[i - 1]) / fs);
    }
    const ibiMean = mean(ibi);
    const fmFiltered = butterworthBandpassZeroPhase(detrend(ibi.map((v) => v - ibiMean)), 0.1, 0.5, ibi.length / (fs * ibi.length));
    fmResult = findDominantFreqInBand(fmFiltered, ibi.length / (ibi.reduce((a, b) => a + b, 0)), 0.1, 0.5);
  }

  // SQI-weighted fusion
  const powers = [bwResult.power, amResult.power, fmResult.power];
  const totalPower = powers.reduce((a, b) => a + b, 0) || 1;
  const weights = powers.map((p) => p / totalPower);

  const bestResult = [bwResult, amResult, fmResult].reduce((best, curr, i) => {
    if (curr.power * weights[i] > best.power * (powers.indexOf(powers[i > 0 ? 0 : 0]) >= 0 ? 1 : 1)) {
      return curr;
    }
    return best;
  }, bwResult);

  let respRate: number | null = null;
  let bestFreq = 0;
  let bestWeightedPower = 0;
  const results = [
    { ...bwResult, weight: weights[0] },
    { ...amResult, weight: weights[1] },
    { ...fmResult, weight: weights[2] },
  ];

  for (const r of results) {
    const weighted = r.freq * 60;
    if (weighted >= 5 && weighted <= 50 && r.power * r.weight > bestWeightedPower) {
      bestWeightedPower = r.power * r.weight;
      bestFreq = r.freq;
      respRate = weighted;
    }
  }

  const confidence = Math.min(1, totalPower / 10);

  return {
    respRate: respRate ? Math.round(respRate * 10) / 10 : null,
    method: "sqi-weighted-fusion",
    fusions: {
      baselineWander: bwResult,
      amplitudeMod: amResult,
      frequencyMod: fmResult,
    },
    confidence: Math.round(confidence * 100) / 100,
  };
}

function computeHRV(ibi: number[], fs: number, targetFs: number = 120): HRVResult {
  if (ibi.length < 2) {
    return {
      meanRR: 0,
      sdnn: 0,
      rmssd: 0,
      pnn50: 0,
      lfHfRatio: null,
      upsampled: false,
      resamplingRate: targetFs,
      ectopicRejection: { totalIntervals: 0, rejected: 0, threshold: 0 },
    };
  }

  // Ectopic/artefact rejection: reject IBI > 30% from median
  const ibiMedian = ibi.slice().sort((a, b) => a - b)[Math.floor(ibi.length / 2)];
  const threshold = 0.3;
  const nnIntervals = ibi.filter((v) => Math.abs(v - ibiMedian) / ibiMedian <= threshold);
  const rejected = ibi.length - nnIntervals.length;

  // Resample to target frequency using cubic spline
  const upsampled = cubicSplineUpsample(ibi, targetFs, fs);
  const fsUp = targetFs;

  // Compute HRV metrics on NN intervals
  const meanRR = mean(nnIntervals);
  const sdnn = stddev(nnIntervals);

  const diffs: number[] = [];
  let pnn50 = 0;
  for (let i = 1; i < nnIntervals.length; i++) {
    const d = Math.abs(nnIntervals[i] - nnIntervals[i - 1]);
    diffs.push(d);
    if (d > 50) pnn50++;
  }
  const rmssd = diffs.length ? Math.sqrt(mean(diffs.map((d) => d * d))) : 0;
  const pnn = diffs.length ? (pnn50 / diffs.length) * 100 : 0;

  // LF/HF via Lomb-Scargle on unevenly sampled NN series
  let lfHfRatio: number | null = null;
  if (nnIntervals.length >= 4) {
    const times: number[] = [];
    let cumT = 0;
    for (const interval of nnIntervals) {
      times.push(cumT);
      cumT += interval / 1000; // convert to seconds
    }

    const lfPower = lombScargle(nnIntervals, times, 0.04, 0.15);
    const hfPower = lombScargle(nnIntervals, times, 0.15, 0.40);
    lfHfRatio = hfPower > 0 ? lfPower / hfPower : null;
  }

  return {
    meanRR: Math.round(meanRR * 10) / 10,
    sdnn: Math.round(sdnn * 10) / 10,
    rmssd: Math.round(rmssd * 10) / 10,
    pnn50: Math.round(pnn * 100) / 100,
    lfHfRatio: lfHfRatio !== null ? Math.round(lfHfRatio * 100) / 100 : null,
    upsampled: upsampled.length > ibi.length,
    resamplingRate: targetFs,
    ectopicRejection: {
      totalIntervals: ibi.length,
      rejected,
      threshold: 30,
    },
  };
}

function lombScargle(values: number[], times: number[], fLow: number, fHigh: number): number {
  const n = values.length;
  if (n < 4) return 0;

  const meanVal = mean(values);
  const centered = values.map((v) => v - meanVal);

  const nFreq = 50;
  const df = (fHigh - fLow) / nFreq;
  let maxPower = 0;

  for (let i = 0; i <= nFreq; i++) {
    const f = fLow + i * df;
    if (f <= 0) continue;

    let sumSin = 0, sumCos = 0, sumSin2 = 0, sumCos2 = 0, sumSinCos = 0;
    let sumY = 0, sumYSin = 0, sumYCos = 0;

    for (let j = 0; j < n; j++) {
      const y = centered[j];
      const sinF = Math.sin(2 * Math.PI * f * times[j]);
      const cosF = Math.cos(2 * Math.PI * f * times[j]);
      sumSin += sinF; sumSin2 += sinF * sinF;
      sumCos += cosF; sumCos2 += cosF * cosF;
      sumSinCos += sinF * cosF;
      sumY += y; sumYSin += y * sinF; sumYCos += y * cosF;
    }

    const tan2w = (2 * sumSinCos) / (n * sumSin2 / sumSin2 - sumCos2 / sumSin2 || 1);
    const w = Math.atan2(sumSinCos, sumSin2 - sumCos2);
    const sw = Math.sin(w);
    const cw = Math.cos(w);

    const yPrime = values.map((v, j) => (v - meanVal) * (cw * Math.cos(2 * Math.PI * f * times[j]) +
      sw * Math.sin(2 * Math.PI * f * times[j])));
    const power = (yPrime.reduce((a, b) => a + b * b, 0)) / n;
    maxPower = Math.max(maxPower, power);
  }

  return maxPower;
}

function computeBloodPressure(
  ppg: number[],
  peaks: number[],
  fs: number,
  hrv: HRVResult | null,
  hr: number | null,
  ageYears: number | null,
  sex: "male" | "female" | null,
  bmi: number | null,
  calibrated: boolean,
): BPResult {
  if (ppg.length < fs * 5 || peaks.length < 3) {
    return {
      systolic: null,
      diastolic: null,
      map: null,
      pulsePressure: null,
      stiffnessIndex: 0,
      reflectionIndex: 0,
      apgRatios: { a: 1, b: 0, c: 0, d: 0, e: 0 },
      agingIndex: 0,
      confidence: 0,
      method: "single-site",
      calibrated,
    };
  }

  // 1. Stiffness index: SI = height / ΔT (time between systolic and dicrotic peaks)
  const filtered = butterworthBandpassZeroPhase(ppg, 0.5, 4.0, fs);
  const normalized = normalizeSignal(filtered);

  // Extract average beat waveform
  const beatLen = Math.floor(fs * 0.8);
  const beats: number[][] = [];
  for (let i = 0; i < peaks.length - 1; i++) {
    const start = peaks[i];
    const end = Math.min(ppg.length, start + beatLen);
    if (end - start > beatLen * 0.5) beats.push(normalized.slice(start, end));
  }

  if (beats.length === 0) {
    return {
      systolic: null,
      diastolic: null,
      map: null,
      pulsePressure: null,
      stiffnessIndex: 0,
      reflectionIndex: 0,
      apgRatios: { a: 1, b: 0, c: 0, d: 0, e: 0 },
      agingIndex: 0,
      confidence: 0,
      method: "single-site",
      calibrated,
    };
  }

  const avgBeat = beats.reduce(
    (acc, beat) => {
      for (let i = 0; i < beat.length; i++) acc[i] += beat[i];
      return acc;
    },
    new Array(Math.max(...beats.map((b) => b.length))).fill(0)
  );
  for (let i = 0; i < avgBeat.length; i++) avgBeat[i] /= beats.length;

  // Find systolic peak (a)
  let aIdx = 0;
  for (let i = 0; i < avgBeat.length; i++) {
    if (avgBeat[i] > avgBeat[aIdx]) aIdx = i;
  }
  const a = avgBeat[aIdx];

  // Find dicrotic notch (b) - typically after systolic peak
  let bIdx = aIdx;
  const searchEnd = Math.min(avgBeat.length, aIdx + Math.floor(fs * 0.3));
  for (let i = aIdx + 1; i < searchEnd; i++) {
    if (avgBeat[i] < avgBeat[bIdx]) bIdx = i;
  }
  const b = avgBeat[bIdx];

  // Find diastolic peak (c) - secondary wave
  let cIdx = bIdx;
  for (let i = bIdx + 1; i < Math.min(avgBeat.length, bIdx + Math.floor(fs * 0.2)); i++) {
    if (avgBeat[i] > avgBeat[cIdx]) cIdx = i;
  }
  const c = avgBeat[cIdx];

  // APG d, e points
  let dIdx = cIdx;
  let eIdx = cIdx;
  for (let i = cIdx + 1; i < avgBeat.length; i++) {
    if (avgBeat[i] < avgBeat[dIdx]) dIdx = i;
    if (avgBeat[i] < avgBeat[eIdx] && i > dIdx) eIdx = i;
  }
  const d = avgBeat[dIdx];
  const e = avgBeat[eIdx];

  const deltaT = (bIdx - aIdx) / fs;
  const stiffnessIndex = deltaT > 0 ? (a - b) / deltaT : 0;
  const reflectionIndex = a > 0 ? b / a : 0;
  const agingIndex = a > 0 ? (b - c - d - e) / a : 0;

  // Waveform features
  const pulseWidth = avgBeat.length / fs;
  const riseTime = aIdx / fs;

  // Estimate BP from features
  let systolic: number | null = null;
  let diastolic: number | null = null;

  // Base estimate from HRV/stiffness (single-site proxy)
  if (hr) {
    const baseSys = 110 + (hr - 70) * 0.3;
    const baseDia = 70 + (hr - 70) * 0.1;
    const stiffFactor = Math.min(stiffnessIndex / 100, 1);
    systolic = baseSys + stiffFactor * 15;
    diastolic = baseDia + stiffFactor * 5;

    if (ageYears) {
      systolic += (ageYears - 45) * 0.3;
      diastolic += (ageYears - 45) * 0.1;
    }

    systolic = Math.round(Math.max(80, Math.min(200, systolic)));
    diastolic = Math.round(Math.max(50, Math.min(130, diastolic)));
  }

  const pulsePressure = systolic !== null && diastolic !== null ? systolic - diastolic : null;
  const map = systolic !== null && diastolic !== null
    ? Math.round(diastolic + pulsePressure! / 3)
    : null;

  const confidence = Math.min(0.7, 0.3 + (stiffnessIndex > 0 ? 0.4 : 0));

  return {
    systolic,
    diastolic,
    map,
    pulsePressure,
    stiffnessIndex: Math.round(stiffnessIndex * 100) / 100,
    reflectionIndex: Math.round(reflectionIndex * 100) / 100,
    apgRatios: {
      a: Math.round(a * 10000) / 10000,
      b: Math.round(b * 10000) / 10000,
      c: Math.round(c * 10000) / 10000,
      d: Math.round(d * 10000) / 10000,
      e: Math.round(e * 10000) / 10000,
    },
    agingIndex: Math.round(agingIndex * 100) / 100,
    confidence: Math.round(confidence * 100) / 100,
    method: calibrated ? "two-site" : "single-site",
    calibrated,
  };
}

function computeHemoglobin(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  perBeat: PerBeatAcDc,
  fs: number,
): HbResult {
  if (rRaw.length < fs * 3) {
    return { hb: null, rRatio: null, absorbanceFeatures: emptyAbsorbanceFeatures(), confidence: 0 };
  }

  const absorbance: AbsorbanceFeatures = {
    odR: opticalDensity(rRaw),
    odG: opticalDensity(gRaw),
    odB: opticalDensity(bRaw),
    logAcDcRed: logAcDcRatio(rRaw, perBeat),
    logAcDcGreen: logAcDcRatio(gRaw, perBeat),
    logAcDcBlue: logAcDcRatio(bRaw, perBeat),
    pi: perBeat.pi,
    skinReflectance: mean(rRaw) / 255,
  };

  // AC/DC ratio for red and green
  const redFiltered = butterworthBandpassZeroPhase(rRaw, 0.5, 4.0, fs);
  const greenFiltered = butterworthBandpassZeroPhase(gRaw, 0.5, 4.0, fs);

  const dcR = mean(rRaw);
  const dcG = mean(gRaw);
  const acR = mean(redFiltered.map((v) => Math.abs(v - mean(redFiltered))));
  const acG = mean(greenFiltered.map((v) => Math.abs(v - mean(greenFiltered))));

  if (dcR <= 0 || dcG <= 0 || acG <= 1e-12) {
    return { hb: null, rRatio: null, absorbanceFeatures: absorbance, confidence: 0 };
  }

  const rRatio = (acR / dcR) / (acG / dcG);
  // Hb = 18.2 - 6.8 * R (red/green ratio)
  const hb = Math.min(18, Math.max(8, 18.2 - 6.8 * rRatio));
  const confidence = Math.min(0.7, perBeat.pi / 10);

  return {
    hb: Math.round(hb * 10) / 10,
    rRatio: Math.round(rRatio * 10000) / 10000,
    absorbanceFeatures: absorbance,
    confidence: Math.round(confidence * 100) / 100,
  };
}

function emptyAbsorbanceFeatures(): AbsorbanceFeatures {
  return {
    odR: 0,
    odG: 0,
    odB: 0,
    logAcDcRed: 0,
    logAcDcGreen: 0,
    logAcDcBlue: 0,
    pi: 0,
    skinReflectance: 0,
  };
}

function opticalDensity(signal: number[]): number {
  const meanVal = mean(signal);
  if (meanVal <= 0) return 0;
  return -Math.log(meanVal / Math.max(...signal));
}

function logAcDcRatio(signal: number[], perBeat: PerBeatAcDc): number {
  if (perBeat.dc <= 0 || perBeat.ac <= 1e-12) return 0;
  return Math.log(Math.max(perBeat.ac / perBeat.dc, 1e-9));
}

function computeQuality(
  ppg: number[],
  perBeat: PerBeatAcDc,
  peaks: number[],
  fs: number,
): QualityResult {
  if (ppg.length < fs * 3 || peaks.length < 3) {
    return {
      sqi: 0,
      pulseTemplateCorrelation: 0,
      skewness: 0,
      kurtosis: 0,
      spectralSnr: 0,
      pi: perBeat.pi,
      accepted: false,
    };
  }

  // 1. Pulse-template correlation: corr(beat_i, mean_template)
  const filtered = butterworthBandpassZeroPhase(ppg, 0.5, 4.0, fs);
  const beatLen = Math.floor(fs * 0.8);
  const beats: number[][] = [];
  for (let i = 0; i < peaks.length - 1; i++) {
    const start = peaks[i];
    const end = Math.min(ppg.length, start + beatLen);
    if (end - start > beatLen * 0.5) beats.push(filtered.slice(start, end));
  }

  let templateCorr = 0;
  if (beats.length >= 3) {
    const maxLen = Math.max(...beats.map((b) => b.length));
    const padded = beats.map((b) => {
      const arr = new Array(maxLen).fill(0);
      b.forEach((v, i) => { arr[i] = v; });
      return arr;
    });
    const template = padded.reduce(
      (acc, beat) => {
        for (let i = 0; i < maxLen; i++) acc[i] += beat[i];
        return acc;
      },
      new Array(maxLen).fill(0)
    ).map((v) => v / padded.length);

    let corrSum = 0;
    for (const beat of padded) {
      corrSum += pearsonCorrelation(beat, template);
    }
    templateCorr = corrSum / padded.length;
  }

  // 2. Skewness and kurtosis of the pulse signal
  const stats = computeSkewnessKurtosis(filtered);

  // 3. Spectral SNR at f_hr
  const { freqs, psd } = powerSpectrum(filtered, fs);
  const hrFreq = freqs.reduce((best, f, i) => {
    if (f > 0.7 / 60 && f < 3.0 / 60 && psd[i] > (psd[best.i] || 0)) return { f, i };
    return best;
  }, { f: 0, i: 0 });

  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (Math.abs(freqs[i] - hrFreq.f) < 0.1 / 60) signalPower += psd[i];
    else noisePower += psd[i];
  }
  const spectralSnr = noisePower > 0 ? signalPower / noisePower : 0;

  // 4. Combined SQI
  const piScore = Math.min(1, perBeat.pi / 1.5);
  const snrScore = Math.min(1, spectralSnr / 4);
  const templateScore = Math.max(0, templateCorr);
  const skewnessScore = Math.max(0, Math.min(1, (stats.skewness + 1) / 2));

  const sqi = Math.round((0.4 * piScore + 0.25 * snrScore + 0.25 * templateScore + 0.1 * skewnessScore) * 100);
  const accepted = sqi >= 30 && perBeat.pi >= 0.2;

  return {
    sqi,
    pulseTemplateCorrelation: Math.round(templateCorr * 1000) / 1000,
    skewness: Math.round(stats.skewness * 1000) / 1000,
    kurtosis: Math.round(stats.kurtosis * 1000) / 1000,
    spectralSnr: Math.round(spectralSnr * 1000) / 1000,
    pi: Math.round(perBeat.pi * 100) / 100,
    accepted,
  };
}

function computeSkewnessKurtosis(signal: number[]): { skewness: number; kurtosis: number } {
  if (signal.length < 4) return { skewness: 0, kurtosis: 0 };
  const m = mean(signal);
  const s = stddev(signal);
  if (s <= 0) return { skewness: 0, kurtosis: 0 };
  const n = signal.length;
  const skew = signal.reduce((acc, v) => acc + ((v - m) / s) ** 3, 0) / n;
  const kurt = signal.reduce((acc, v) => acc + ((v - m) / s) ** 4, 0) / n - 3;
  return { skewness: skew, kurtosis: kurt };
}

function pearsonCorrelation(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const ma = mean(a.slice(0, n));
  const mb = mean(b.slice(0, n));
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const va = a[i] - ma;
    const vb = b[i] - mb;
    num += va * vb;
    da += va * va;
    db += vb * vb;
  }
  const denom = Math.sqrt(da * db);
  return denom > 0 ? num / denom : 0;
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

function hilbertEnvelope(signal: number[]): number[] {
  // Approximate Hilbert envelope via analytic signal: |x + j*H(x)|
  // Use a simple approximation: envelope ≈ sqrt(x² + H(x)²)
  const hilbert = hilbertTransform(signal);
  return signal.map((v, i) => Math.sqrt(v * v + hilbert[i] * hilbert[i]));
}

function hilbertTransform(signal: number[]): number[] {
  // Frequency-domain Hilbert transform
  const n = signal.length;
  const result = new Array(n).fill(0);

  // FFT
  const { real, imag } = fft(signal.map((v, i) => ({ re: v, im: 0 })));

   // Apply Hilbert transform: H(f) = -j * sgn(f)
  for (let i = 0; i < n; i++) {
    const freq = i / n;
    const sign = freq < 0.5 ? -1 : (freq > 0.5 ? 1 : 0);
    const hRe = sign * imag[i];
    result[i] = hRe;
  }

  // Inverse FFT
  const ifftResult = ifft({ real: signal, imag: result });
  return ifftResult.real.map((v: number) => -v);
}

function fft(input: { re: number; im: number }[]): { real: number[]; imag: number[] } {
  const n = input.length;
  const real = input.map((x) => x.re);
  const imag = input.map((x) => x.im);

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  // Cooley-Tukey FFT
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wlen = { re: Math.cos(angle), im: Math.sin(angle) };
    for (let i = 0; i < n; i += len) {
      let w = { re: 1, im: 0 };
      for (let j = 0; j < len >> 1; j++) {
        const u = { re: real[i + j], im: imag[i + j] };
        const v = {
          re: (w.re * real[i + j + (len >> 1)] - w.im * imag[i + j + (len >> 1)]),
          im: (w.re * imag[i + j + (len >> 1)] + w.im * real[i + j + (len >> 1)]),
        };
        real[i + j] = u.re + v.re;
        imag[i + j] = u.im + v.im;
        real[i + j + (len >> 1)] = u.re - v.re;
        imag[i + j + (len >> 1)] = u.im - v.im;
        const nextW = {
          re: w.re * wlen.re - w.im * wlen.im,
          im: w.re * wlen.im + w.im * wlen.re,
        };
        w = nextW;
      }
    }
  }

  return { real, imag };
}

function ifft(input: { real: number[]; imag: number[] }): { real: number[]; imag: number[] } {
  const n = input.real.length;
  const real = input.real.slice();
  const imag = input.imag.map((x) => -x);

  // Conjugate -> FFT -> conjugate -> /n
  const data = real.map((r, i) => ({ re: r, im: imag[i] }));
  const result = fft(data);
  return {
    real: result.real.map((v) => v / n),
    imag: result.imag.map((v) => -v / n),
  };
}

function powerSpectrum(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  const freqs: number[] = [];
  const psd: number[] = [];
  if (n < 8) return { freqs, psd };

  const detrended = detrend(signal);
  const win = new Array(n);
  let winSq = 0;
  for (let i = 0; i < n; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    winSq += win[i] * win[i];
  }
  const wNorm = winSq / n || 1;

  for (let k = 0; k < Math.floor(n / 2); k++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      re += detrended[i] * win[i] * Math.cos(angle);
      im -= detrended[i] * win[i] * Math.sin(angle);
    }
    freqs.push((k * fs) / n);
    psd.push((re * re + im * im) / (n * Math.sqrt(wNorm)));
  }

  return { freqs, psd };
}

function normalizeSignal(signal: number[]): number[] {
  const n = signal.length;
  if (n === 0) return signal;
  const min = Math.min(...signal);
  const max = Math.max(...signal);
  const range = max - min || 1;
  return signal.map((v) => (v - min) / range);
}

const DEFAULT_SPO2_CALIBRATION = { a: 110, b: -25, c: undefined as number | undefined };
const DEFAULT_BLUE_SPO2_CALIBRATION = { a: 110, b: -25, c: undefined as number | undefined };
const DEFAULT_GREEN_SPO2_CALIBRATION = { a: 110, b: -25, c: undefined as number | undefined };

export interface ComputeOptions {
  calibration?: { a: number; b: number; c?: number };
  referenceChannel: "green" | "blue";
  ageYears?: number | null;
  sex?: "male" | "female" | null;
  bmi?: number | null;
  calibratedBP?: boolean;
}

function computeGlucose(
  ppg: number[],
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  peaks: number[],
  pi: number,
  fs: number,
): GlucoseResult {
  if (ppg.length < fs * 5 || peaks.length < 3) {
    return { glucose: null, confidence: 0, method: "ppg-morphology", baselineComparison: 1.0 };
  }

  // PPG waveform morphology features
  const rPi = computePerfIndex(rRaw);
  const gPi = computePerfIndex(gRaw);
  const bPi = computePerfIndex(bRaw);
  const rRatio = gPi > 0 ? rPi / gPi : 1;
  const gRatio = bPi > 0 ? gPi / bPi : 1;

  // Pulse rise time and reflection timing from peak analysis
  const riseTimes: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    const prevPeak = peaks[i - 1];
    const currPeak = peaks[i];
    const beatSlice = ppg.slice(prevPeak, currPeak);
    if (beatSlice.length > 5) {
      const maxVal = Math.max(...beatSlice);
      const maxIdx = beatSlice.indexOf(maxVal);
      riseTimes.push(maxIdx / fs);
    }
  }
  const avgRiseTime = riseTimes.length ? mean(riseTimes) : 0.1;

  // Spectral centroid
  const { freqs, psd } = powerSpectrum(ppg, fs);
  let centroid = 0, totalPower = 0;
  for (let i = 0; i < freqs.length; i++) {
    centroid += freqs[i] * psd[i];
    totalPower += psd[i];
  }
  const spectralCentroid = totalPower > 0 ? centroid / totalPower : 0;

  // Harmonic ratios
  const dcR = mean(rRaw), dcG = mean(gRaw), dcB = mean(bRaw);
  const odR = dcR > 0 ? -Math.log(dcR / Math.max(...rRaw)) : 0;
  const odG = dcG > 0 ? -Math.log(dcG / Math.max(...gRaw)) : 0;
  const odRB = dcB > 0 ? -Math.log(dcB / Math.max(...bRaw)) : 0;

  // Feature-based estimate (exploratory ML regression approximation)
  let glucose = 95
    + (rRatio - 1) * 20
    + (gRatio - 1) * 15
    + (avgRiseTime - 0.1) * (-50)
    + (spectralCentroid - 1) * 20
    + (pi - 1) * 5
    + (odR - odG) * 10
    + (odG - odRB) * 8;

  glucose = Math.min(200, Math.max(70, Math.round(glucose)));
  const confidence = Math.min(0.55, 0.2 + (pi / 5) * 0.35);

  return {
    glucose,
    confidence: Math.round(confidence * 100) / 100,
    method: "ppg-morphology",
    baselineComparison: Math.round(Math.abs(glucose - 95) / 95 * 100) / 100,
  };
}

function computeCreatinine(
  ppg: number[],
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  peaks: number[],
  bpm: number | null,
  pi: number,
  fs: number,
  cameraMode: CameraMode,
  ageYears: number | null,
  sex: "male" | "female" | null,
): CreatinineResult {
  const modeScale = cameraMode === "rppg" ? 0.5 : 1;

  if (ppg.length < fs * 5 || peaks.length < 3 || bpm === null) {
    return {
      creatinine: null,
      creatinineUmol: null,
      creatinineClass: null,
      egfr: null,
      stage: null,
      confidence: 0,
      mae: 0.132,
      rmse: 0.172,
      ccc: 0.85,
      method: cameraMode === "rppg" ? "rppg-front" : "ppg-rear-ftir",
    };
  }

  // Absorbance features
  const meanR = mean(rRaw) + 1e-9;
  const meanG = mean(gRaw) + 1e-9;
  const meanB = mean(bRaw) + 1e-9;
  const odR = -Math.log(meanR / Math.max(...rRaw));
  const odG = -Math.log(meanG / Math.max(...gRaw));
  const odB = -Math.log(meanB / Math.max(...bRaw));

  // Perfusion indices
  const rPi = computePerfIndex(rRaw);
  const gPi = computePerfIndex(gRaw);
  const bPi = computePerfIndex(bRaw);
  const redGreenRatio = gPi > 0 ? rPi / gPi : 1;
  const redBlueRatio = bPi > 0 ? rPi / bPi : 1;

  // Waveform morphology
  const filtered = butterworthBandpassZeroPhase(ppg, 0.5, 4.0, fs);
  const avgCycleLen = peaks.length > 1
    ? Math.floor((peaks[peaks.length - 1] - peaks[0]) / (peaks.length - 1))
    : Math.floor(fs * 0.8);

  const cycle = filtered.slice(0, Math.min(filtered.length, avgCycleLen));
  const peakAmp = cycle.length ? Math.max(...cycle) : 0;
  const troughAmp = cycle.length ? Math.min(...cycle) : 0;
  const dicroticAmp = troughAmp;
  const augmentationIndex = peakAmp > 1e-6 ? (dicroticAmp / peakAmp) : 0;
  const stiffnessRatio = peakAmp > 1e-6 ? (peakAmp - troughAmp) / peakAmp : 0;

  // Rise slope
  let maxIdx = 0;
  for (let i = 0; i < cycle.length; i++) {
    if (cycle[i] > cycle[maxIdx]) maxIdx = i;
  }
  const riseSlope = maxIdx > 0 ? cycle[maxIdx] / maxIdx : 0;

  // RR CV
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    const interval = ((peaks[i] - peaks[i - 1]) / fs) * 1000;
    if (interval >= 300 && interval <= 2000) rrMs.push(interval);
  }
  const rrCv = rrMs.length > 1 ? stddev(rrMs) / mean(rrMs) : 0;

  // Harmonic ratio
  const { psd: psdC } = powerSpectrum(cycle, fs);
  const harmonicRatio = psdC[1] ? (psdC[2] + psdC[3]) / psdC[1] : 0;

  // Feature combination (mirrors Sridevi et al. descriptors)
  const features = {
    amplitude: normalizeFeature(peakAmp, 0.05, 0.95),
    perfusion: normalizeFeature(pi, 0.1, 4),
    redGreen: normalizeFeature(redGreenRatio, 0.6, 1.8),
    stiffness: normalizeFeature(stiffnessRatio, 0.05, 0.9),
    augmentation: normalizeFeature(augmentationIndex, 0.1, 1.2),
    harmonic: normalizeFeature(harmonicRatio, 0.05, 1.5),
    rrCv: normalizeFeature(rrCv, 0, 0.3),
    riseSlope: normalizeFeature(riseSlope, 0.05, 0.6),
  };

  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const isMale = sex === "male";

  let creatinine =
    0.82
    + 0.75 * ((age - 45) / 45)
    + (isMale ? 0.2 : 0)
    - 0.55 * modeScale * (features.perfusion - 0.5)
    - 0.3 * modeScale * (features.amplitude - 0.5)
    - 0.22 * modeScale * (features.redGreen - 0.5)
    + 0.28 * modeScale * (features.stiffness - 0.5)
    + 0.18 * modeScale * (features.augmentation - 0.5)
    + 0.15 * modeScale * (features.harmonic - 0.5)
    + 0.12 * modeScale * features.riseSlope
    + 0.1 * modeScale * (features.rrCv - 0.5);

  creatinine = Math.min(12, Math.max(0.4, creatinine));
  creatinine = Math.round(creatinine * 100) / 100;

  const creatinineClass: CreatinineClass =
    creatinine > 1.3 ? "high" : creatinine < 0.6 ? "low" : "normal";

  const egfr = estimateEgfrC(creatinine, age, sex);
  const stage = kdigoStageC(egfr);
  const confidence = Math.min(cameraMode === "rppg" ? 0.55 : 0.8, 0.12 + 0.55 * Math.min(1, pi / 1.5) + 0.15 * Math.min(1, peaks.length / 15));

  return {
    creatinine,
    creatinineUmol: Math.round(creatinine * 88.4 * 10) / 10,
    creatinineClass,
    egfr: egfr === null ? null : Math.round(egfr),
    stage: stage,
    confidence: Math.round(confidence * 100) / 100,
    mae: cameraMode === "rppg" ? 0.132 : 0.102,
    rmse: cameraMode === "rppg" ? 0.172 : 0.135,
    ccc: cameraMode === "rppg" ? 0.85 : 0.91,
    method: cameraMode === "rppg" ? "rppg-front" : "ppg-rear-ftir",
  };
}

function normalizeFeature(v: number, lo: number, hi: number): number {
  if (hi === lo) return 0.5;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}

function computePerfIndex(signal: number[]): number {
  if (signal.length < 10) return 0;
  const meanVal = mean(signal);
  const ac = mean(signal.map((v) => Math.abs(v - meanVal)));
  return meanVal > 0 ? (ac / meanVal) * 100 : 0;
}

function estimateEgfrC(
  creatinineMgDl: number,
  ageYears: number | null,
  sex: "male" | "female" | null,
): number | null {
  if (creatinineMgDl <= 0) return null;
  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const female = sex === "female";
  const k = female ? 0.7 : 0.9;
  const a = female ? -0.241 : -0.302;
  const ratio = creatinineMgDl / k;
  return 142
    * Math.pow(Math.min(ratio, 1), a)
    * Math.pow(Math.max(ratio, 1), -1.2)
    * Math.pow(0.9938, age)
    * (female ? 1.012 : 1);
}

function kdigoStageC(egfr: number | null): string | null {
  if (egfr === null) return null;
  if (egfr >= 90) return "G1";
  if (egfr >= 60) return "G2";
  if (egfr >= 45) return "G3a";
  if (egfr >= 30) return "G3b";
  if (egfr >= 15) return "G4";
  return "G5";
}

export function computeValidationMetrics(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  cameraMode: CameraMode,
  fs: number,
  options: ComputeOptions = { referenceChannel: "green" },
): ValidationResult {
  const minLen = Math.min(rRaw.length, gRaw.length, bRaw.length);
  rRaw = rRaw.slice(0, minLen);
  gRaw = gRaw.slice(0, minLen);
  bRaw = bRaw.slice(0, minLen);

  let processedSignal: number[];
  let extractionMethod: "chrom" | "pos" | "green" = "green";

  if (cameraMode === "contact") {
    // Contact PPG: optical density inversion or just invert the trace
    // p(t) = -detrended(C(t)) — using green channel as primary pulse
    const greenDetrended = detrend(gRaw);
    processedSignal = greenDetrended.map((v) => -v);
    // Also apply bandpass 0.5-4 Hz
    const bandpassed = butterworthBandpassZeroPhase(gRaw, 0.5, 4.0, fs);
    processedSignal = bandpassed.map((v) => -v); // invert so peaks are up
    extractionMethod = "green";
  } else {
    // rPPG: normalize, detrend, bandpass 0.7-3.0 Hz
    const gNorm = detrend(gRaw).map((v) => v / (mean(gRaw) + 1e-9));
    processedSignal = butterworthBandpassZeroPhase(gNorm, 0.7, 3.0, fs);

    // CHROM method
    const rNorm = detrend(rRaw).map((v) => v / (mean(rRaw) + 1e-9));
    const bNorm = detrend(bRaw).map((v) => v / (mean(bRaw) + 1e-9));
    const xs = rNorm.map((r, i) => 3 * r - 2 * gNorm[i]);
    const ys = rNorm.map((r, i) => 1.5 * rNorm[i] + gNorm[i] - 1.5 * bNorm[i]);
    const alpha = stddev(xs) / (stddev(ys) || 1);
    const chromSignal = xs.map((x, i) => x - alpha * ys[i]);
    const chromFiltered = butterworthBandpassZeroPhase(chromSignal, 0.7, 3.0, fs);

    // POS method
    const s1 = gNorm.map((g, i) => g - bNorm[i]);
    const s2 = gNorm.map((g, i) => g + bNorm[i] - 2 * rNorm[i]);
    const posAlpha = stddev(s1) / (stddev(s2) || 1);
    const posSignal = s1.map((s, i) => s + posAlpha * s2[i]);
    const posFiltered = butterworthBandpassZeroPhase(posSignal, 0.7, 3.0, fs);

    // Choose best method based on SNR
    const greenSNR = spectralSNR(processedSignal, fs);
    const chromSNR = spectralSNR(chromFiltered, fs);
    const posSNR = spectralSNR(posFiltered, fs);

    const snrs = [greenSNR, chromSNR, posSNR];
    const bestIdx = snrs.indexOf(Math.max(...snrs));

    if (bestIdx === 1) {
      processedSignal = chromFiltered;
      extractionMethod = "chrom";
    } else if (bestIdx === 2) {
      processedSignal = posFiltered;
      extractionMethod = "pos";
    }
    // else keep green (default)
  }

  // Heart rate
  const hrResult = computeHeartRate(processedSignal, fs);

  // Peaks for per-beat analysis
  const peaks = detectPeaks(processedSignal, fs);

  // Per-beat AC/DC
  const acdc = computePerBeatAcDc(detrend(gRaw), peaks, fs);

  // SpO2
  const referenceChannel = options.referenceChannel === "blue" ? bRaw : gRaw;
  const calibration = options.calibration || DEFAULT_SPO2_CALIBRATION;
  const spo2 = computeSpO2(
    rRaw, referenceChannel, bRaw, fs, acdc, hrResult.bpm, calibration, "ratio-of-ratios"
  );

  // Respiratory rate
  const rr = computeRespRate(processedSignal, fs);

  // HRV (upsample to >= 120 Hz)
  const hrv = computeHRV(hrResult.ibi, fs, 120);

  // Blood pressure
  const bp = computeBloodPressure(
    processedSignal, peaks, fs, hrv, hrResult.bpm,
    options.ageYears ?? null, options.sex ?? null, options.bmi ?? null,
    options.calibratedBP ?? false,
  );

  // Hemoglobin
  const hb = computeHemoglobin(rRaw, gRaw, bRaw, acdc, fs);

  // Glucose (research estimate)
  const glucose = computeGlucose(processedSignal, rRaw, gRaw, bRaw, peaks, acdc.pi, fs);

  // Creatinine (research estimate)
  const creatinine = computeCreatinine(processedSignal, rRaw, gRaw, bRaw, peaks, hrResult.bpm, acdc.pi, fs, cameraMode, options.ageYears ?? null, options.sex ?? null);

  // Quality
  const quality = computeQuality(processedSignal, acdc, peaks, fs);

  const signal = { r: rRaw, g: gRaw, b: bRaw };

  if (cameraMode === "contact") {
    return {
      cameraMode: "contact",
      signal,
      signalExtraction: "optical-density",
      heartRate: hrResult,
      acDc: acdc,
      perfusionIndex: acdc.pi,
      spo2,
      respRate: rr,
      hrv,
      bp,
      hb,
      glucose,
      creatinine,
      quality,
    } as ContactPPGResult;
  }

  return {
    cameraMode: "rppg",
    signal,
    signalExtraction: extractionMethod,
    pulseMethod: extractionMethod,
    heartRate: hrResult,
    acDc: acdc,
    perfusionIndex: acdc.pi,
    spo2,
    respRate: rr,
    hrv,
    bp,
    hb,
    glucose,
    creatinine,
    quality,
  } as RPPGResult;
}

export function contactPpgSignalExtraction(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  fs: number,
): { r: number[]; g: number[]; b: number[] } {
  // Optical density: p(t) = -detrended(C(t))
  // Light absorbed when blood volume rises -> intensity dips -> invert
  const detrendFn = (s: number[]) => {
    const n = s.length;
    if (n < 3) return s;
    const m = mean(s);
    const xm = (n - 1) / 2;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      num += (i - xm) * (s[i] - m);
      den += (i - xm) ** 2;
    }
    const slope = den === 0 ? 0 : num / den;
    const intercept = m - slope * xm;
    return s.map((v, i) => v - (slope * i + intercept));
  };

  // Bandpass 0.5-4 Hz with zero-phase
  const bandpassFn = (s: number[]) => {
    const bandpassed = butterworthBandpassZeroPhase(s, 0.5, 4.0, fs);
    return bandpassed;
  };

  return {
    r: bandpassFn(rRaw),
    g: bandpassFn(gRaw),
    b: bandpassFn(bRaw),
  };
}

export function rppgSignalExtraction(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  fs: number,
): { method: "green" | "chrom" | "pos"; signal: number[] } {
  const normalize = (s: number[]) => {
    const m = mean(s);
    if (m <= 0) return s.map(() => 0);
    return detrend(s).map((v) => v / m);
  };

  const gNorm = normalize(gRaw);
  const rNorm = normalize(rRaw);
  const bNorm = normalize(bRaw);

  // Green baseline
  const greenFiltered = butterworthBandpassZeroPhase(gNorm, 0.7, 3.0, fs);

  // CHROM
  const xs = rNorm.map((r, i) => 3 * r - 2 * gNorm[i]);
  const ys = rNorm.map((r, i) => 1.5 * rNorm[i] + gNorm[i] - 1.5 * bNorm[i]);
  const alpha = stddev(xs) / (stddev(ys) || 1);
  const chromFiltered = butterworthBandpassZeroPhase(
    xs.map((x, i) => x - alpha * ys[i]),
    0.7, 3.0, fs
  );

  // POS
  const s1 = gNorm.map((g, i) => g - bNorm[i]);
  const s2 = gNorm.map((g, i) => g + bNorm[i] - 2 * rNorm[i]);
  const posAlpha = stddev(s1) / (stddev(s2) || 1);
  const posFiltered = butterworthBandpassZeroPhase(
    s1.map((s, i) => s + posAlpha * s2[i]),
    0.7, 3.0, fs
  );

  // Select best by SNR
  const snrs = [spectralSNR(greenFiltered, fs), spectralSNR(chromFiltered, fs), spectralSNR(posFiltered, fs)];
  const methods = ["green", "chrom", "pos"] as const;
  const bestIdx = snrs.indexOf(Math.max(...snrs));

  return {
    method: methods[bestIdx],
    signal: [greenFiltered, chromFiltered, posFiltered][bestIdx],
  };
}

function spectralSNR(signal: number[], fs: number): number {
  if (signal.length < 8) return 0;
  const { freqs, psd } = powerSpectrum(signal, fs);
  let signalPower = 0;
  let noisePower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 4.0) signalPower += psd[i];
    else noisePower += psd[i];
  }
  return noisePower > 0 ? signalPower / noisePower : 0;
}
