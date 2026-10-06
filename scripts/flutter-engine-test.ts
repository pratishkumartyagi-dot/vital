// Port of vital_monitoring_app/lib/ppg_signal_engine.dart (new engine methods)
// Verifies the advanced signal-processing port: spectralSnr,
// cardiacSpectralProminence, computeSignalQuality, analyzeHeartRate
// (dcRef + ectopic RR rejection), posPulse, perfusionIndexACDC,
// detectPeaks, estimateCreatinine / estimateEgfr.

const fs = 30;
const cardiacLo = 0.5;
const cardiacHi = 3.5;
const noiseBandWidth = 6.0;

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b, 0) / (arr.length - 1);
  return Math.sqrt(v);
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return [...signal];
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = mean(x);
  const yMean = mean(signal);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - xMean) * (signal[i] - yMean);
    den += (x[i] - xMean) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = yMean - slope * xMean;
  return signal.map((y, i) => y - (slope * i + intercept));
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

function normalizeRange(v: number, lo: number, hi: number): number {
  if (hi === lo) return 0;
  return clamp01((v - lo) / (hi - lo));
}

function lowpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cw) / 2 / a0, b1: (1 - cw) / a0, b2: (1 - cw) / 2 / a0,
    a1: (-2 * cw) / a0, a2: (1 - alpha) / a0,
  };
}

function highpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cw) / 2 / a0, b1: -(1 + cw) / a0, b2: (1 + cw) / 2 / a0,
    a1: (-2 * cw) / a0, a2: (1 - alpha) / a0,
  };
}

function applyBiquad(signal: number[], c: any, seedX = 0, seedY = 0): number[] {
  const out = new Array(signal.length).fill(0);
  let x1 = seedX, x2 = seedX, y1 = seedY, y2 = seedY;
  for (let i = 0; i < signal.length; i++) {
    const y = c.b0 * signal[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y;
    x2 = x1; x1 = signal[i];
    y2 = y1; y1 = y;
  }
  return out;
}

function applyBandpassFilterCustom(input: number[], low: number, high: number, fs: number): number[] {
  if (input.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
    return [...input];
  }
  const hp = highpassBiquad(low, fs);
  const lp = lowpassBiquad(high, fs);
  const dc = input.reduce((a, b) => a + b, 0) / input.length;
  let y = applyBiquad(input, hp, dc, 0);
  y = applyBiquad(y, lp, 0, 0);
  y = [...y].reverse();
  y = applyBiquad(y, hp, 0, 0);
  y = applyBiquad(y, lp, 0, 0);
  y = [...y].reverse();
  return y;
}

function rms(signal: number[]): number {
  if (signal.length === 0) return 0;
  let sq = 0;
  for (const v of signal) sq += v * v;
  return Math.sqrt(sq / signal.length);
}

function dcLevel(signal: number[]): number {
  if (signal.length === 0) return 0;
  let sum = 0;
  for (const v of signal) sum += Math.abs(v);
  return sum / signal.length;
}

function powerSpectrum(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  const freqs: number[] = [], psd: number[] = [];
  if (n < 8) return { freqs, psd };
  const detrended = detrend(signal);
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
    const angle = (2 * Math.PI * k) / n;
    for (let i = 0; i < n; i++) {
      const phi = angle * i;
      real += detrended[i] * win[i] * Math.cos(phi);
      imag -= detrended[i] * win[i] * Math.sin(phi);
    }
    freqs.push((k * fs) / n);
    psd.push((real * real + imag * imag) / n / wNorm);
  }
  return { freqs, psd };
}

function spectralSnr(signal: number[], fs: number) {
  const ps = powerSpectrum(signal, fs);
  if (ps.freqs.length === 0) return { snr: 0, signalPower: 0, noiseDensity: 0, sigBins: 0, noiseBins: 0, hasReference: false };
  const noiseHi = Math.min((fs / 2) * 0.9, cardiacHi + noiseBandWidth);
  let signalPower = 0, sigBins = 0, noisePower = 0, noiseBins = 0;
  for (let i = 0; i < ps.freqs.length; i++) {
    const f = ps.freqs[i];
    if (f >= cardiacLo && f <= cardiacHi) { signalPower += ps.psd[i]; sigBins++; }
    else if (f > cardiacHi && f <= noiseHi) { noisePower += ps.psd[i]; noiseBins++; }
  }
  if (sigBins === 0) return { snr: 0, signalPower: 0, noiseDensity: 0, sigBins: 0, noiseBins: 0, hasReference: false };
  const noiseDensity = noiseBins > 0 ? noisePower / noiseBins : 0;
  const inBandNoise = noiseDensity * sigBins;
  if (inBandNoise <= 0) {
    return { snr: Infinity, signalPower, noiseDensity, sigBins, noiseBins, hasReference: false };
  }
  return { snr: signalPower / inBandNoise, signalPower, noiseDensity, sigBins, noiseBins, hasReference: true };
}

function cardiacSpectralProminence(signal: number[], fs: number): number {
  const ps = powerSpectrum(signal, fs);
  const band: number[] = [];
  for (let i = 0; i < ps.freqs.length; i++) {
    if (ps.freqs[i] >= cardiacLo && ps.freqs[i] <= cardiacHi) band.push(ps.psd[i]);
  }
  if (band.length < 8) return 0;
  const sorted = [...band].sort((a, b) => a - b);
  const med = sorted[sorted.length >> 1];
  const peak = sorted[sorted.length - 1];
  return med > 0 ? peak / med : 0;
}

function channelAcDc(signal: number[]): number {
  if (signal.length < 3) return 0;
  const n = signal.length;
  const dc = signal.reduce((a, b) => a + b, 0) / n;
  if (dc <= 1e-9) return 0;
  let sq = 0;
  for (const v of signal) sq += (v - dc) * (v - dc);
  const ac = Math.sqrt(sq / n);
  return ac / dc;
}

function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): { index: number; value: number }[] {
  const filtered = applyBandpassFilterCustom(detrend(signal), cardiacLo, cardiacHi, fs);
  const peaks: { index: number; value: number }[] = [];
  const minDist = Math.floor(Math.max(1, (60 / maxBpm) * fs));
  const sorted = [...filtered].sort((a, b) => a - b);
  const n = sorted.length;
  const p10 = sorted[Math.floor(n * 0.10)];
  const p90 = sorted[Math.floor(n * 0.90)];
  const p97 = sorted[Math.floor(n * 0.97)];
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
  return peaks;
}

function computeHRV(rrMs: number[]) {
  if (rrMs.length < 2) return { meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0 };
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
  return { meanRR, sdnn, rmssd, pnn50: pnn };
}

function classifyRhythm(rrMs: number[]): string {
  if (rrMs.length < 5) return 'unknown';
  const sd = stddev(rrMs);
  const m = mean(rrMs);
  if (m === 0) return 'unknown';
  const cv = (sd / m) * 100;
  if (cv < 20) return 'regular';
  return 'irregular';
}

function computeSignalQuality(filtered: number[], raw: number[], peaks: { index: number; value: number }[], fs: number, dcRef: number): number {
  if (peaks.length < 2) return 0;
  const dc = dcRef > 1e-9 ? dcRef : dcLevel(raw);
  const ac = rms(filtered);
  const pi = dc > 1e-9 ? (ac / dc) * 100 : 0;
  const piScore = clamp01(Math.sqrt(pi / 0.5));
  const snrResult = spectralSnr(raw, fs);
  const snrScore = snrResult.snr > 0 ? clamp01((Math.log(snrResult.snr) / Math.log(10) + 0.3) / 1.2) : 0;
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) rrMs.push(((peaks[i].index - peaks[i - 1].index) / fs) * 1000);
  const med = median(rrMs);
  const filteredRr = med > 0 ? rrMs.filter((d) => Math.abs(d - med) / med <= 0.3) : [...rrMs];
  const rrMsClean = filteredRr.length >= 2 ? filteredRr : rrMs;
  const cv = rrMsClean.length > 1 ? (stddev(rrMsClean) / mean(rrMsClean)) * 100 : 0;
  const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30);
  const prominence = cardiacSpectralProminence(raw, fs);
  const pulseGate = clamp01((prominence - 4) / 12);
  const base = 0.4 * snrScore + 0.35 * piScore + 0.25 * regScore;
  return Math.round(clamp01(base * pulseGate) * 100);
}

function analyzeHeartRate(ppg: number[], fs = 30, dcRef?: number) {
  if (ppg.length < fs * 5) return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0 };
  const trimmed = ppg.slice(Math.min(fs, ppg.length));
  const peaks = detectPeaks(trimmed, fs);
  if (peaks.length < 3) return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0 };
  const valid: { index: number; value: number }[] = [];
  for (const p of peaks) {
    if (valid.length === 0) { valid.push(p); continue; }
    const dtMs = ((p.index - valid[valid.length - 1].index) / fs) * 1000;
    if (dtMs >= 300) valid.push(p);
  }
  if (valid.length < 3) return { bpm: null, peaks: [], rrIntervals: [], hrv: null, rhythm: 'unknown', signalQuality: 0 };
  const rrAll: number[] = [];
  for (let i = 1; i < valid.length; i++) rrAll.push(((valid[i].index - valid[i - 1].index) / fs) * 1000);
  const med = median(rrAll);
  const filteredRr = med > 0 ? rrAll.filter((d) => Math.abs(d - med) / med <= 0.3) : [...rrAll];
  const rrMs = filteredRr.length >= 2 ? filteredRr : rrAll;
  const meanRR = mean(rrMs);
  const bpm = meanRR > 0 ? 60000 / meanRR : 0;
  const hrv = computeHRV(rrMs);
  const rhythm = classifyRhythm(rrMs);
  const filtered = applyBandpassFilterCustom(detrend(trimmed), cardiacLo, cardiacHi, fs);
  const quality = computeSignalQuality(filtered, trimmed, valid, fs, dcRef ?? dcLevel(ppg));
  return {
    bpm: bpm > 30 && bpm < 220 ? bpm : null,
    peaks: valid.map((p) => p.index),
    rrIntervals: rrMs,
    hrv,
    rhythm,
    signalQuality: quality,
  };
}

function perfusionIndexACDC(signal: number[], fs: number, dcRef?: number): number {
  if (signal.length < 10) return 0;
  const dc = dcRef !== undefined && dcRef > 1e-9 ? dcRef : dcLevel(signal);
  if (dc <= 1e-9) return 0;
  const ac = rms(applyBandpassFilterCustom(signal, cardiacLo, cardiacHi, fs));
  return (ac / dc) * 100;
}

function posPulse(red: number[], green: number[], blue: number[], winLen = 32): number[] {
  const n = Math.min(red.length, green.length, blue.length);
  if (n < winLen || winLen < 8) return [];
  const hop = Math.max(1, Math.floor(winLen / 2));
  const sqrt6 = Math.sqrt(6);
  const acc = new Array(n).fill(0);
  const wsum = new Array(n).fill(0);
  for (let start = 0; start + winLen <= n; start += hop) {
    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < winLen; i++) { mr += red[start + i]; mg += green[start + i]; mb += blue[start + i]; }
    mr /= winLen; mg /= winLen; mb /= winLen;
    if (mr <= 1 || mg <= 1 || mb <= 1) continue;
    const cnR = new Array(winLen), cnG = new Array(winLen), cnB = new Array(winLen);
    for (let i = 0; i < winLen; i++) {
      cnR[i] = red[start + i] / mr;
      cnG[i] = green[start + i] / mg;
      cnB[i] = blue[start + i] / mb;
    }
    const s1 = new Array(winLen), s2 = new Array(winLen);
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
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = wsum[i] > 1e-9 ? acc[i] / wsum[i] : 0;
  let gSum = 0;
  for (let i = 0; i < n; i++) gSum += green[i];
  const unitScale = Math.sqrt(3) * (gSum / n);
  for (let i = 0; i < n; i++) out[i] *= unitScale;
  return out;
}

function filteredPerfusionIndex(signal: number[], fs: number): number {
  if (signal.length < fs * 2) return 0;
  const filtered = applyBandpassFilterCustom(signal, cardiacLo, cardiacHi, fs);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  if (dc <= 1e-9) return 0;
  return (rms(filtered) / dc) * 100;
}

function estimateEgfr(creatinineMgDl: number | null, ageYears: number | null, sex: string | null): number | null {
  if (creatinineMgDl === null || creatinineMgDl <= 0 || !Number.isFinite(creatinineMgDl)) return null;
  const age = ageYears && ageYears > 0 ? ageYears : 45;
  const female = sex === 'female';
  const k = female ? 0.7 : 0.9;
  const a = female ? -0.241 : -0.302;
  const ratio = creatinineMgDl / k;
  const egfr =
    142 *
    Math.pow(Math.min(ratio, 1), a) *
    Math.pow(Math.max(ratio, 1), -1.2) *
    Math.pow(0.9938, age) *
    (female ? 1.012 : 1.0);
  return egfr;
}

function kdigoStage(egfr: number | null): string | null {
  if (egfr === null) return null;
  if (egfr >= 90) return 'G1';
  if (egfr >= 60) return 'G2';
  if (egfr >= 45) return 'G3a';
  if (egfr >= 30) return 'G3b';
  if (egfr >= 15) return 'G4';
  return 'G5';
}

// ---- synthetic signals ----
function fingerPPG(seconds: number, hrBpm: number, respBpm: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * (hrBpm / 60) * t) + 3 * Math.sin(2 * Math.PI * 2 * (hrBpm / 60) * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * (respBpm / 60) * t);
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

function facialRPPG(seconds: number, hrBpm: number, respBpm: number, pulsatility = 0.002, seed = 7) {
  const n = Math.floor(seconds * fs);
  const red: number[] = [], green: number[] = [], blue: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const pulse = Math.sin(2 * Math.PI * (hrBpm / 60) * t) + 0.3 * Math.sin(2 * Math.PI * 2 * (hrBpm / 60) * t + 0.4);
    const resp = 0.15 * Math.sin(2 * Math.PI * (respBpm / 60) * t);
    const noise = rnd() * 0.01;
    // skin-tone DC levels with weak pulsatile chrominance modulation.
    // Realistic rPPG: green carries the strongest pulse (haemoglobin
    // absorbs green), red moderate, blue weakest. Avoid the POS null
    // space (equal pulse amplitude in all channels), which would make
    // the S1/S2 projections cancel the pulse entirely.
    const r = 120 * (1 + pulsatility * 0.7 * pulse + resp + noise);
    const g = 140 * (1 + pulsatility * 1.0 * pulse + resp + noise);
    const b = 100 * (1 + pulsatility * 0.5 * pulse + resp + noise);
    red.push(r); green.push(g); blue.push(b);
  }
  return { red, green, blue };
}

console.log('=== PPGSignalEngine port verification ===\n');

// 1. Heart rate accuracy across a range
console.log('--- analyzeHeartRate (rear contact PPG, true HR) ---');
for (const hr of [50, 60, 72, 90, 110, 130]) {
  const g = fingerPPG(15, hr, 15);
  const r = analyzeHeartRate(g, fs);
  console.log(`  true=${hr}  bpm=${r.bpm?.toFixed(1)}  err=${r.bpm ? Math.abs(r.bpm - hr).toFixed(2) : 'null'}  SQI=${r.signalQuality}`);
}

// 2. Signal quality on clean vs noisy
console.log('\n--- computeSignalQuality / spectralSnr (noise sweep) ---');
for (const noiseAmp of [0.02, 0.5, 1.0, 2.0, 4.0]) {
  const n = Math.floor(15 * fs);
  const g: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * 1.2 * t) + 3 * Math.sin(2 * Math.PI * 2.4 * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * 0.25 * t);
    g.push(110 + cardiac + resp + (Math.random() - 0.5) * 2 * noiseAmp);
  }
  const snr = spectralSnr(g, fs);
  const peaks = detectPeaks(g, fs);
  const filtered = applyBandpassFilterCustom(detrend(g), cardiacLo, cardiacHi, fs);
  const sqi = computeSignalQuality(filtered, g, peaks, fs, dcLevel(g));
  const hr = analyzeHeartRate(g, fs);
  console.log(`  noise=${noiseAmp.toFixed(2)}  SNR=${snr.snr.toFixed(2)}  prom=${cardiacSpectralProminence(g, fs).toFixed(1)}  SQI=${sqi}  HR=${hr.bpm?.toFixed(1)}`);
}

// 3. POS pulse on facial rPPG
console.log('\n--- posPulse (facial rPPG, true HR) ---');
for (const puls of [0.001, 0.002, 0.005]) {
  const { red, green, blue } = facialRPPG(15, 72, 15, puls);
  const pulse = posPulse(red, green, blue);
  const dcRef = dcLevel(green);
  const hr = analyzeHeartRate(pulse, fs, dcRef);
  const pi = perfusionIndexACDC(pulse, fs, dcRef);
  console.log(`  pulsatility=${puls}  HR=${hr.bpm?.toFixed(1)} (true 72)  PI=${pi.toFixed(3)}%  SQI=${hr.signalQuality}`);
}

// 4. Flat signal rejection
console.log('\n--- flat / noise-only rejection ---');
const flat = new Array(300).fill(110).map(() => 110 + (Math.random() - 0.5) * 0.5);
const flatHr = analyzeHeartRate(flat, fs);
const flatSnr = spectralSnr(flat, fs);
console.log(`  flat: HR=${flatHr.bpm}  SQI=${flatHr.signalQuality}  SNR=${flatSnr.snr.toFixed(2)}`);
const pureNoise = new Array(900).fill(0).map(() => 110 + (Math.random() - 0.5) * 40);
const noiseHr = analyzeHeartRate(pureNoise, fs);
console.log(`  pure noise: HR=${noiseHr.bpm}  SQI=${noiseHr.signalQuality}`);

// 5. Ectopic RR rejection / HRV sanity
console.log('\n--- HRV (SDNN/RMSSD) on clean 72 bpm ---');
const g72 = fingerPPG(30, 72, 15);
const hr72 = analyzeHeartRate(g72, fs);
console.log(`  bpm=${hr72.bpm?.toFixed(1)}  SDNN=${hr72.hrv?.sdnn.toFixed(1)}ms  RMSSD=${hr72.hrv?.rmssd.toFixed(1)}ms  rhythm=${hr72.rhythm}`);

// 6. eGFR / KDIGO staging
console.log('\n--- CKD-EPI 2021 eGFR + KDIGO ---');
for (const cr of [0.7, 1.0, 1.3, 2.0, 4.0]) {
  const egfrM = estimateEgfr(cr, 45, 'male');
  const egfrF = estimateEgfr(cr, 45, 'female');
  console.log(`  Cr=${cr} mg/dL  eGFR(male,45y)=${egfrM?.toFixed(0)} [${kdigoStage(egfrM)}]  eGFR(female,45y)=${egfrF?.toFixed(0)} [${kdigoStage(egfrF)}]`);
}

console.log('\n=== done ===');
