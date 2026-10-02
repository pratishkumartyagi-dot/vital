import { powerSpectrum, butterworthBandpass, analyzeHeartRate } from "../src/app/vital-monitoring/ppg-analysis";

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

function getSNR(ppg: number[], fs: number) {
  const filtered = butterworthBandpass(ppg, 0.5, 3.5, fs, true);
  const { freqs, psd: psdFiltered } = powerSpectrum(filtered, fs);
  const { psd: psdRaw } = powerSpectrum(ppg, fs);
  
  let sigF = 0, noiseF = 0;
  let sigR = 0, noiseR = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) { sigF += psdFiltered[i]; sigR += psdRaw[i]; }
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) { noiseF += psdFiltered[i]; noiseR += psdRaw[i]; }
  }
  const snrF = noiseF > 1e-12 ? sigF / noiseF : 0;
  const snrR = noiseR > 1e-12 ? sigR / noiseR : (sigR > 1e-12 ? 8 : 0);
  return { snrF, snrR };
}

const clean = synth(fs, 30, 75);
const cleanHR = analyzeHeartRate(clean, fs);
console.log("Clean:", getSNR(clean, fs), "SQI=", cleanHR.signalQuality);

const noisy = synth(fs, 30, 75, 0.25, 45, 5);
const noisyHR = analyzeHeartRate(noisy, fs);
console.log("Noisy:", getSNR(noisy, fs), "SQI=", noisyHR.signalQuality);

const med = synth(fs, 30, 75, 0.25, 15, 5);
const medHR = analyzeHeartRate(med, fs);
console.log("Med noise:", getSNR(med, fs), "SQI=", medHR.signalQuality);

const weak = clean.map(v => 120 + (v - 120) * 0.15);
const weakHR = analyzeHeartRate(weak, fs);
console.log("Weak:", getSNR(weak, fs), "SQI=", weakHR.signalQuality);
