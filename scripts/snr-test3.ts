import { powerSpectrum, butterworthBandpass, analyzeHeartRate } from "../src/app/vital-monitoring/ppg-analysis";

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

function analyzeSQIComponents(ppg: number[], fs: number) {
  const filtered = butterworthBandpass(ppg, 0.5, 3.5, fs, true);
  const heart = analyzeHeartRate(ppg, fs);
  
  const pi = rawPI(ppg);
  const piScore = Math.min(1, pi / 1.5);
  
  const { freqs, psd } = powerSpectrum(ppg, fs);
  let sigR = 0, noiseR = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigR += psd[i];
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noiseR += psd[i];
  }
  const snrR = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);
  const snrScore = Math.min(1, snrR / 4);
  
  const peaks = heart.peaks;
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) rrMs.push(((peaks[i] - peaks[i-1]) / fs) * 1000);
  const meanRR = rrMs.reduce((a,b) => a+b, 0)/rrMs.length || 1;
  const varRR = rrMs.reduce((a,b) => a + (b-meanRR)*(b-meanRR), 0) / (rrMs.length || 1);
  const cv = meanRR > 0 ? (Math.sqrt(varRR) / meanRR) * 100 : 100;
  const regScore = cv < 10 ? 1 : Math.max(0, 1 - (cv - 10) / 30);
  
  const oldGate = clamp01((snrR - 0.5) / 1.0);
  
  const cardiacPSD = freqs.map((f,i) => f >= 0.5 && f <= 3.5 ? psd[i] : 0);
  const cardiacMax = Math.max(...cardiacPSD);
  const cardiacSorted = cardiacPSD.filter(x => x > 0).sort((a,b) => a - b);
  const cardiacMed = cardiacSorted.length > 0 ? cardiacSorted[Math.floor(cardiacSorted.length/2)] : 1e-12;
  const dominance = cardiacMed > 1e-12 ? cardiacMax / cardiacMed : 0;
  const newGate = clamp01(dominance / 5);
  
  const rawScore = 0.45 * piScore + 0.35 * snrScore + 0.2 * regScore;
  
  console.log(`  PI=${pi.toFixed(2)} snrR=${snrR.toFixed(3)} cv=${cv.toFixed(0)} peaks=${peaks.length} dom=${dominance.toFixed(1)}`);
  console.log(`  rawScore=${(rawScore*100).toFixed(0)} oldGate=${oldGate.toFixed(2)} oldSQI=${Math.round(rawScore*oldGate*100)} newGate=${newGate.toFixed(2)} newSQI=${Math.round(rawScore*newGate*100)}`);
  
  return { oldSQI: Math.round(rawScore * oldGate * 100), newSQI: Math.round(rawScore * newGate * 100) };
}

console.log("Clean:"); analyzeSQIComponents(synth(fs, 30, 75), fs);
console.log("Noisy:"); analyzeSQIComponents(synth(fs, 30, 75, 0.25, 45, 5), fs);
console.log("Med:"); analyzeSQIComponents(synth(fs, 30, 75, 0.25, 15, 5), fs);
console.log("Weak:"); analyzeSQIComponents(synth(fs, 30, 75).map(v => 120 + (v-120)*0.15), fs);
