import { powerSpectrum, butterworthBandpass, analyzeHeartRate } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;
function clamp01(x: number): number { return Math.max(0, Math.min(1, x)); }
function rawPI(signal: number[]): number {
  if (signal.length < 10) return 0;
  let m = 0, absMean = 0;
  for (const v of signal) { m += v; absMean += Math.abs(v); }
  m /= signal.length; absMean /= signal.length;
  const denom = Math.abs(m) > 1e-9 ? m : absMean;
  if (denom <= 1e-9) return 0;
  let ac = 0;
  for (const v of signal) ac += Math.abs(v - m);
  ac /= signal.length;
  return (ac / denom) * 100;
}
function fPI(signal: number[], fs: number): number {
  const filtered = butterworthBandpass(signal, 0.5, 3.5, fs, true);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  if (dc <= 1e-9) return 0;
  let sq = 0;
  for (const v of filtered) sq += v * v;
  const ac = Math.sqrt(sq / filtered.length);
  return (ac / dc) * 100;
}

// Pure noise (no cardiac signal)
const n = fs * 30;
const rndSeed = 42;
let s = rndSeed;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
const pureNoise = Array.from({length: n}, () => 120 + rnd() * 45);

// Pure noise with low amplitude (just DC + noise, no cardiac)
const pureNoiseLow = Array.from({length: n}, () => 100 + rnd() * 5);

const heartNoise = analyzeHeartRate(pureNoise, fs);
const heartNoiseLow = analyzeHeartRate(pureNoiseLow, fs);

const rawPI_N = rawPI(pureNoise);
const fPI_N = fPI(pureNoise, fs);
const rawPI_NL = rawPI(pureNoiseLow);
const fPI_NL = fPI(pureNoiseLow, fs);

const { freqs, psd } = powerSpectrum(pureNoise, fs);
let sigR = 0, noiseR = 0;
for (let i = 0; i < freqs.length; i++) {
  if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigR += psd[i];
  else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseR += psd[i];
}
const snrR_N = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);

console.log("Pure noise (std=22.5):");
console.log(`  rawPI=${rawPI_N.toFixed(2)} fPI=${fPI_N.toFixed(2)} snrR=${snrR_N.toFixed(3)} SQI=${heartNoise.signalQuality} peaks=${heartNoise.peaks.length}`);

const { psd: psdNL } = powerSpectrum(pureNoiseLow, fs);
let sigR2 = 0, noiseR2 = 0;
for (let i = 0; i < freqs.length; i++) {
  if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigR2 += psdNL[i];
  else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseR2 += psdNL[i];
}
const snrR_NL = noiseR2 > 1e-12 ? sigR2 / noiseR2 : (sigR2 > 1e-12 ? 8 : 0);

console.log("Pure noise low (std=2.5):");
console.log(`  rawPI=${rawPI_NL.toFixed(2)} fPI=${fPI_NL.toFixed(2)} snrR=${snrR_NL.toFixed(3)} SQI=${heartNoiseLow.signalQuality} peaks=${heartNoiseLow.peaks.length}`);
