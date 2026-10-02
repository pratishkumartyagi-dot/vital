import { powerSpectrum, butterworthBandpass, analyzeHeartRate } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;
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
  const filtered = butterworthBandpass(ppg, 0.5, 3.5, fs, true);
  const { freqs, psd: psdRaw } = powerSpectrum(ppg, fs);
  const { psd: psdFiltered } = powerSpectrum(filtered, fs);
  
  let sigR = 0, noiseR = 0;
  let sigF = 0, noiseF = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) { sigR += psdRaw[i]; sigF += psdFiltered[i]; }
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) { noiseR += psdRaw[i]; noiseF += psdFiltered[i]; }
  }
  const snrR = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);
  const snrF = noiseF > 1e-12 ? sigF / noiseF : (sigF > 1e-12 ? 8 : 0);
  return { snrR, snrF };
}

console.log("=== Noise levels ===");
for (const noise of [4, 10, 15, 20, 25, 30, 35, 40, 45]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const { snrR, snrF } = analyzeSNR(sig, fs);
  const h = analyzeHeartRate(sig, fs);
  console.log(`noise=${noise}: snrR=${snrR.toFixed(3)} snrF=${snrF.toFixed(1)} SQI=${h.signalQuality}`);
}

console.log("\n=== Pure noise (no cardiac) ===");
let s = 42;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
const n = fs * 30;
for (const noise of [5, 10, 20, 45]) {
  const pureNoise = Array.from({length: n}, () => 120 + rnd() * noise);
  const { snrR, snrF } = analyzeSNR(pureNoise, fs);
  const h = analyzeHeartRate(pureNoise, fs);
  console.log(`noise=${noise}: snrR=${snrR.toFixed(3)} snrF=${snrF.toFixed(1)} SQI=${h.signalQuality} peaks=${h.peaks.length}`);
}
