import { powerSpectrum, butterworthBandpass, analyzeHeartRate, computePerfusionIndex as rawPI, filteredPerfusionIndex } from "../src/app/vital-monitoring/ppg-analysis";

// filteredPerfusionIndex is not exported, let me reimplement it inline
function fPI(signal: number[], fs: number): number {
  const filtered = butterworthBandpass(signal, 0.5, 3.5, fs, true);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  if (dc <= 1e-9) return 0;
  let sq = 0;
  for (const v of filtered) sq += v * v;
  const ac = Math.sqrt(sq / filtered.length);
  return (ac / dc) * 100;
}

function rPI(signal: number[]): number {
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

function synth(fs: number, seconds: number, hrBpm: number, respHz = 0.25, noise = 4, seed = 13): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
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

const fs = 30;

const clean = synth(fs, 30, 75);
const noisy = synth(fs, 30, 75, 0.25, 45, 5);
const med = synth(fs, 30, 75, 0.25, 15, 5);
const weak = clean.map(v => 120 + (v - 120) * 0.15);

console.log("Clean:  rawPI=" + rPI(clean).toFixed(2) + " fPI=" + fPI(clean, fs).toFixed(2) + " SQI=" + analyzeHeartRate(clean, fs).signalQuality);
console.log("Noisy:  rawPI=" + rPI(noisy).toFixed(2) + " fPI=" + fPI(noisy, fs).toFixed(2) + " SQI=" + analyzeHeartRate(noisy, fs).signalQuality);
console.log("Med:    rawPI=" + rPI(med).toFixed(2) + " fPI=" + fPI(med, fs).toFixed(2) + " SQI=" + analyzeHeartRate(med, fs).signalQuality);
console.log("Weak:   rawPI=" + rPI(weak).toFixed(2) + " fPI=" + fPI(weak, fs).toFixed(2) + " SQI=" + analyzeHeartRate(weak, fs).signalQuality);
