import { analyzeVitals, posPulse, analyzeHeartRate, computePerfusionIndex, powerSpectrum } from "./src/app/vital-monitoring/ppg-analysis";

// Test with more realistic noise - lower SNR
const fs = 30;
const hrTrue = 72;
const seconds = 30;
const n = seconds * fs;

// Test 1: Good quality (like self-check)
console.log("=== Test 1: Good quality ===");
const red1: number[] = [];
const green1: number[] = [];
const blue1: number[] = [];
let s = 91;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const ph = 2 * Math.PI * (hrTrue / 60) * t;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  const motion = 1 + 0.35 * Math.sin(2 * Math.PI * 1.0 * t) + 0.002 * rnd();
  red1.push(135 * (1 + 0.06 * pulse) * motion);
  green1.push(120 * (1 + 0.18 * pulse) * motion);
  blue1.push(110 * (1 + 0.10 * pulse) * motion);
}
const pos1 = posPulse(red1, green1, blue1, 32);
const hr1 = analyzeHeartRate(pos1, fs);
console.log("POS SQI:", hr1.signalQuality, "BPM:", hr1.bpm?.toFixed(1));
const pi1 = computePerfusionIndex(pos1);
console.log("POS PI:", pi1.toFixed(3) + "%");
const { freqs, psd } = powerSpectrum(pos1, fs);
let sigPower = 0, noisePower = 0;
for (let i = 0; i < freqs.length; i++) {
  if (freqs[i] >= 0.5 && freqs[i] <= 3.5) sigPower += psd[i];
  else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noisePower += psd[i];
}
console.log("POS SNR:", (sigPower / noisePower).toFixed(2));

// Test 2: Lower pulsatility (more realistic for rPPG)
console.log("\n=== Test 2: Lower pulsatility (realistic rPPG) ===");
const red2: number[] = [];
const green2: number[] = [];
const blue2: number[] = [];
s = 91;
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const ph = 2 * Math.PI * (hrTrue / 60) * t;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  // Lower pulsatility: R=2%, G=6%, B=3% (typical for facial rPPG)
  const motion = 1 + 0.1 * Math.sin(2 * Math.PI * 0.5 * t) + 0.005 * rnd(); // slower motion
  red2.push(135 * (1 + 0.02 * pulse) * motion);
  green2.push(120 * (1 + 0.06 * pulse) * motion);
  blue2.push(110 * (1 + 0.03 * pulse) * motion);
}
const pos2 = posPulse(red2, green2, blue2, 32);
const hr2 = analyzeHeartRate(pos2, fs);
console.log("POS SQI:", hr2.signalQuality, "BPM:", hr2.bpm?.toFixed(1));
const pi2 = computePerfusionIndex(pos2);
console.log("POS PI:", pi2.toFixed(3) + "%");
const { freqs: f2, psd: p2 } = powerSpectrum(pos2, fs);
let sigPower2 = 0, noisePower2 = 0;
for (let i = 0; i < f2.length; i++) {
  if (f2[i] >= 0.5 && f2[i] <= 3.5) sigPower2 += p2[i];
  else if (f2[i] > 3.5 && f2[i] <= fs * 0.4) noisePower2 += p2[i];
}
console.log("POS SNR:", (sigPower2 / noisePower2).toFixed(2));

// Test 3: With breathing artifact (respiratory sinus arrhythmia)
console.log("\n=== Test 3: With respiratory variation ===");
const red3: number[] = [];
const green3: number[] = [];
const blue3: number[] = [];
s = 91;
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const respMod = 1 + 0.05 * Math.sin(2 * Math.PI * 0.25 * t); // RSA
  const ph = 2 * Math.PI * (hrTrue / 60) * t * respMod;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  const motion = 1 + 0.1 * Math.sin(2 * Math.PI * 0.5 * t) + 0.005 * rnd();
  red3.push(135 * (1 + 0.02 * pulse) * motion);
  green3.push(120 * (1 + 0.06 * pulse) * motion);
  blue3.push(110 * (1 + 0.03 * pulse) * motion);
}
const pos3 = posPulse(red3, green3, blue3, 32);
const hr3 = analyzeHeartRate(pos3, fs);
console.log("POS SQI:", hr3.signalQuality, "BPM:", hr3.bpm?.toFixed(1));
const pi3 = computePerfusionIndex(pos3);
console.log("POS PI:", pi3.toFixed(3) + "%");
const { freqs: f3, psd: p3 } = powerSpectrum(pos3, fs);
let sigPower3 = 0, noisePower3 = 0;
for (let i = 0; i < f3.length; i++) {
  if (f3[i] >= 0.5 && f3[i] <= 3.5) sigPower3 += p3[i];
  else if (f3[i] > 3.5 && f3[i] <= fs * 0.4) noisePower3 += p3[i];
}
console.log("POS SNR:", (sigPower3 / noisePower3).toFixed(2));

// Test 4: Very low pulsatility + noise
console.log("\n=== Test 4: Very low pulsatility + more noise ===");
const red4: number[] = [];
const green4: number[] = [];
const blue4: number[] = [];
s = 91;
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const ph = 2 * Math.PI * (hrTrue / 60) * t;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  const motion = 1 + 0.2 * Math.sin(2 * Math.PI * 0.5 * t) + 0.02 * rnd(); // more motion + noise
  red4.push(135 * (1 + 0.01 * pulse) * motion);
  green4.push(120 * (1 + 0.04 * pulse) * motion);
  blue4.push(110 * (1 + 0.02 * pulse) * motion);
}
const pos4 = posPulse(red4, green4, blue4, 32);
const hr4 = analyzeHeartRate(pos4, fs);
console.log("POS SQI:", hr4.signalQuality, "BPM:", hr4.bpm?.toFixed(1));
const pi4 = computePerfusionIndex(pos4);
console.log("POS PI:", pi4.toFixed(3) + "%");
const { freqs: f4, psd: p4 } = powerSpectrum(pos4, fs);
let sigPower4 = 0, noisePower4 = 0;
for (let i = 0; i < f4.length; i++) {
  if (f4[i] >= 0.5 && f4[i] <= 3.5) sigPower4 += p4[i];
  else if (f4[i] > 3.5 && f4[i] <= fs * 0.4) noisePower4 += p4[i];
}
console.log("POS SNR:", (sigPower4 / noisePower4).toFixed(2));

// Test 5: Skin detection failure simulation - full frame with background
console.log("\n=== Test 5: Background contamination (poor skin detection) ===");
const red5: number[] = [];
const green5: number[] = [];
const blue5: number[] = [];
s = 91;
for (let i = 0; i < n; i++) {
  const t = i / fs;
  const ph = 2 * Math.PI * (hrTrue / 60) * t;
  const pulse = Math.sin(ph) + 0.3 * Math.sin(2 * ph + 0.7);
  // Face is only 30% of frame, rest is static background
  const faceRatio = 0.3;
  const bgR = 100, bgG = 80, bgB = 60; // background color
  const faceR = 135 * (1 + 0.02 * pulse);
  const faceG = 120 * (1 + 0.06 * pulse);
  const faceB = 110 * (1 + 0.03 * pulse);
  const motion = 1 + 0.1 * Math.sin(2 * Math.PI * 0.5 * t);
  red5.push((faceRatio * faceR + (1 - faceRatio) * bgR) * motion);
  green5.push((faceRatio * faceG + (1 - faceRatio) * bgG) * motion);
  blue5.push((faceRatio * faceB + (1 - faceRatio) * bgB) * motion);
}
const pos5 = posPulse(red5, green5, blue5, 32);
const hr5 = analyzeHeartRate(pos5, fs);
console.log("POS SQI:", hr5.signalQuality, "BPM:", hr5.bpm?.toFixed(1));
const pi5 = computePerfusionIndex(pos5);
console.log("POS PI:", pi5.toFixed(3) + "%");
const { freqs: f5, psd: p5 } = powerSpectrum(pos5, fs);
let sigPower5 = 0, noisePower5 = 0;
for (let i = 0; i < f5.length; i++) {
  if (f5[i] >= 0.5 && f5[i] <= 3.5) sigPower5 += p5[i];
  else if (f5[i] > 3.5 && f5[i] <= fs * 0.4) noisePower5 += p5[i];
}
console.log("POS SNR:", (sigPower5 / noisePower5).toFixed(2));

process.exit(0);
