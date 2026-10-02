import { analyzeHeartRate } from "../src/app/vital-monitoring/ppg-analysis";

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

console.log("=== SNR Blueprint Verification ===");
console.log("SNR > 5.0 (Clean): should be high SQI");
for (const noise of [4, 10, 15, 20]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const h = analyzeHeartRate(sig, fs);
  console.log(`  noise=${noise}: SQI=${h.signalQuality} bpm=${h.bpm?.toFixed(1)}`);
}

console.log("\nSNR 3.0-5.0 (Minor noise): recovery zone, SQI near clean");
for (const noise of [20, 25]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const h = analyzeHeartRate(sig, fs);
  console.log(`  noise=${noise}: SQI=${h.signalQuality} bpm=${h.bpm?.toFixed(1)}`);
}

console.log("\nSNR 1.0-3.0 (Highly unstable): should be capped below 50");
for (const noise of [30, 35]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const h = analyzeHeartRate(sig, fs);
  console.log(`  noise=${noise}: SQI=${h.signalQuality} bpm=${h.bpm?.toFixed(1)}`);
}

console.log("\nSNR < 1.0 (Pure artifacts): should be clamped to 0");
for (const noise of [40, 45]) {
  const sig = synth(fs, 30, 75, 0.25, noise, 5);
  const h = analyzeHeartRate(sig, fs);
  console.log(`  noise=${noise}: SQI=${h.signalQuality} bpm=${h.bpm?.toFixed(1)}`);
}

console.log("\n=== Pure noise (no cardiac) ===");
let s = 42;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
const n = fs * 30;
const pureNoise = Array.from({length: n}, () => 120 + rnd() * 45);
const h = analyzeHeartRate(pureNoise, fs);
console.log(`  pure noise: SQI=${h.signalQuality} peaks=${h.peaks.length}`);

console.log("\n=== Clean reference ===");
const clean = synth(fs, 30, 75);
const hc = analyzeHeartRate(clean, fs);
console.log(`  clean: SQI=${hc.signalQuality} bpm=${hc.bpm?.toFixed(1)}`);
