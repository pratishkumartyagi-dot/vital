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
  const { freqs, psd } = powerSpectrum(ppg, fs);
  let sigR = 0, noiseR = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigR += psd[i];
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseR += psd[i];
  }
  const snrR = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);
  return { snrR, rawPI: rawPI(ppg) };
}

// Weak perfusion with low noise (the "valid but weak signal" case)
console.log("=== Weak perfusion, low noise ===");
for (const gain of [0.5, 0.3, 0.15, 0.08]) {
  const sig = synth(fs, 30, 75, 0.25, 4, 5).map(v => 120 + (v - 120) * gain);
  const { snrR, rawPI: rpi } = analyzeSNR(sig, fs);
  const h = analyzeHeartRate(sig, fs);
  console.log(`gain=${gain}: snrR=${snrR.toFixed(3)} rawPI=${rpi.toFixed(2)} SQI=${h.signalQuality} peaks=${h.peaks.length}`);
}

// What SQI would these get with different peakGate thresholds?
console.log("\n=== SQI with different gates (gain=0.15, snrR=0.664) ===");
const sig = synth(fs, 30, 75).map(v => 120 + (v - 120) * 0.15);
const h = analyzeHeartRate(sig, fs);
const { snrR, rawPI: rpi } = analyzeSNR(sig, fs);
const piScore = Math.min(1, rpi / 1.5);

// Compute SNR score
const { freqs, psd } = powerSpectrum(sig, fs);
let sigR = 0, noiseR = 0;
for (let i = 0; i < freqs.length; i++) {
  if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigR += psd[i];
  else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseR += psd[i];
}
const snrScore = Math.min(1, snrR / 4);

// Compute regScore
const rrMs: number[] = [];
for (let i = 1; i < h.peaks.length; i++) rrMs.push(((h.peaks[i] - h.peaks[i-1]) / fs) * 1000);
const meanRR = rrMs.reduce((a,b) => a+b, 0)/rrMs.length || 1;
const varRR = rrMs.reduce((a,b) => a + (b-meanRR)*(b-meanRR), 0) / (rrMs.length || 1);
const cv = meanRR > 0 ? (Math.sqrt(varRR) / meanRR) * 100 : 100;
const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30);

const rawScore = 0.45 * piScore + 0.35 * snrScore + 0.2 * regScore;
console.log(`  piScore=${piScore.toFixed(2)} snrScore=${snrScore.toFixed(3)} regScore=${regScore.toFixed(2)} cv=${cv.toFixed(0)}`);
console.log(`  rawScore=${(rawScore*100).toFixed(0)}`);

// Current gate (0.5 threshold)
const gate05 = clamp01((snrR - 0.5) / 1.0);
console.log(`  gate(0.5): ${gate05.toFixed(2)} → SQI=${Math.round(rawScore*gate05*100)}`);

// Proposed gates
const gate02 = clamp01((snrR - 0.2) / 1.0);
console.log(`  gate(0.2): ${gate02.toFixed(2)} → SQI=${Math.round(rawScore*gate02*100)}`);

const gate01 = clamp01((snrR - 0.1) / 1.0);
console.log(`  gate(0.1): ${gate01.toFixed(2)} → SQI=${Math.round(rawScore*gate01*100)}`);
