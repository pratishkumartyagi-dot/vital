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
  for (const v of signal) ac += Math.abs(v - m); ac /= signal.length;
  return (ac / denom) * 100;
}
function fPI(signal: number[], fs: number): number {
  const filtered = butterworthBandpass(signal, 0.5, 3.5, fs, true);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  if (dc <= 1e-9) return 0;
  let sq = 0;
  for (const v of filtered) sq += v * v;
  return (Math.sqrt(sq / filtered.length) / dc) * 100;
}

function synth(fs: number, seconds: number, hrBpm: number, respHz = 0.25, noise = 4, seed = 13): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const ph = 2 * Math.PI * (hrBpm / 60) * t;
    let pulse = Math.sin(ph) + 0.5 * Math.sin(2 * ph + 0.5) + 0.25 * Math.sin(3 * ph);
    pulse = (pulse + 1.75) / 3.5;
    const baseline = 0.5 + 0.18 * Math.sin(2 * Math.PI * respHz * t) + 0.12 * Math.sin(2 * Math.PI * 0.06 * t);
    out.push(120 + 70 * (pulse * baseline - 0.5) + rnd() * noise);
  }
  return out;
}

function analyzeSNR(ppg: number[], fs: number) {
  const { freqs, psd: psdRaw } = powerSpectrum(ppg, fs);
  const { psd: psdFiltered } = powerSpectrum(butterworthBandpass(ppg, 0.5, 3.5, fs, true), fs);
  
  let sigR = 0, noiseR = 0;
  let sigF = 0, noiseF = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) { sigR += psdRaw[i]; sigF += psdFiltered[i]; }
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) { noiseR += psdRaw[i]; noiseF += psdFiltered[i]; }
  }
  const snrR = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);
  const snrF = noiseF > 1e-12 ? sigF / noiseF : (sigF > 1e-12 ? 8 : 0);
  
  // Dominance: peak/median in cardiac band on raw PSD
  const cardiacBin: number[] = [];
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) cardiacBin.push(psdRaw[i]);
  }
  const sorted = cardiacBin.slice().sort((a, b) => a - b);
  const med = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 1e-12;
  const max = Math.max(...cardiacBin);
  const dominance = med > 1e-12 ? max / med : 0;
  
  return { snrR, snrF, dominance, rawPI: rawPI(ppg), fPI: fPI(ppg, fs) };
}

const tests = [
  { name: "Clean", sig: synth(fs, 30, 75) },
  { name: "Noisy", sig: synth(fs, 30, 75, 0.25, 45, 5) },
  { name: "Med noise", sig: synth(fs, 30, 75, 0.25, 15, 5) },
  { name: "Weak", sig: synth(fs, 30, 75).map(v => 120 + (v - 120) * 0.15) },
];

let s = 42;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
const noiseOnly = Array.from({length: fs * 30}, () => 120 + rnd() * 45);
tests.push({ name: "Pure noise", sig: noiseOnly });

for (const { name, sig } of tests) {
  const h = analyzeHeartRate(sig, fs);
  const { snrR, snrF, dominance, rawPI: rpi, fPI: fpi } = analyzeSNR(sig, fs);
  console.log(`${name}: snrR=${snrR.toFixed(3)} snrF=${snrF.toFixed(1)} dom=${dominance.toFixed(1)} rawPI=${rpi.toFixed(2)} fPI=${fpi.toFixed(2)} SQI=${h.signalQuality} peaks=${h.peaks.length}`);
}
