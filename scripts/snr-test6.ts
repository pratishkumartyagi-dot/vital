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

// Find noise level that gives snrR ≈ 1.0
for (const noise of [20, 25, 30, 35, 40, 45]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const { snrR, rawPI: rpi } = analyzeSNR(sig, fs);
  const h = analyzeHeartRate(sig, fs);
  console.log(`noise=${noise}: snrR=${snrR.toFixed(3)} rawPI=${rpi.toFixed(2)} SQI=${h.signalQuality}`);
}

// Also test with lower gain (weaker signal, same noise → lower SNR)
for (const gain of [0.5, 0.3, 0.1]) {
  const sig = synth(fs, 30, 75).map(v => 120 + (v - 120) * gain + (Math.random() - 0.5) * 6);
  const { snrR, rawPI: rpi } = analyzeSNR(sig, fs);
  const h = analyzeHeartRate(sig, fs);
  console.log(`gain=${gain}: snrR=${snrR.toFixed(3)} rawPI=${rpi.toFixed(2)} SQI=${h.signalQuality}`);
}
