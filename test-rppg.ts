import { analyzeVitals, posPulse, butterworthBandpass, computeSignalQuality, detectPeaks, analyzeHeartRate } from "./src/app/vital-monitoring/ppg-analysis";

// Test POS pulse with synthetic data similar to self-check
const fs = 30;
const hrTrue = 78;
const seconds = 30;
const n = seconds * fs;
const red: number[] = [];
const green: number[] = [];
const blue: number[] = [];
let s = 91;
const rnd = () => {
  s = (s * 1103515245 + 12345) & 0x7fffffff;
  return s / 0x7fffffff - 0.5;
};
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const ph = 2 * Math.PI * (hrTrue / 60) * t;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  const motion = 1 + 0.35 * Math.sin(2 * Math.PI * 1.0 * t) + 0.002 * rnd();
  red.push(135 * (1 + 0.06 * pulse) * motion);
  green.push(120 * (1 + 0.18 * pulse) * motion);
  blue.push(110 * (1 + 0.10 * pulse) * motion);
}

const pos = posPulse(red, green, blue, 32);
console.log("POS length:", pos.length);
console.log("POS range:", Math.min(...pos).toFixed(3), "to", Math.max(...pos).toFixed(3));

// Test analyzeHeartRate on POS
const hr = analyzeHeartRate(pos, fs);
console.log("\nHeart rate on POS:", hr.bpm, "SQI:", hr.signalQuality, "peaks:", hr.peaks.length);

// Test analyzeVitals front mode
const vitals = analyzeVitals(pos, fs, red, green, blue, { cameraMode: "front" });
console.log("\nVitals front mode:");
console.log("  HR:", vitals.heart.bpm, "SQI:", vitals.heart.signalQuality);
console.log("  Resp:", vitals.resp.respRate, "method:", vitals.resp.method, "conf:", vitals.resp.confidence);

// Check raw green
const hrGreen = analyzeHeartRate(green, fs);
console.log("\nHeart rate on raw green:", hrGreen.bpm, "SQI:", hrGreen.signalQuality, "peaks:", hrGreen.peaks.length);

process.exit(hr.signalQuality < 50 ? 1 : 0);
