import { butterworthBandpass } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;

// Reuse the CORRECT web filter (RBJ biquad HP0.5+LP3.5, zero-phase) to validate the fix.
function findPeaks(signal: number[], minDistance = 1, threshold = 0.0): number[] {
  const peaks: number[] = [];
  if (signal.length < 3) return peaks;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const stdDev = Math.sqrt(signal.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / signal.length);
  const dt = threshold * stdDev;
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > dt) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) peaks.push(i);
    }
  }
  return peaks;
}

// Dart-style PI: peak-to-trough/mean DC *100, but with CORRECT bandpass
function computePerfIndexCorrect(rawSignal: number[]): number {
  if (rawSignal.length < 10) return 0;
  const filtered = butterworthBandpass(rawSignal, 0.5, 3.5, fs, true);
  const peaks: number[] = [], troughs: number[] = [];
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1]) peaks.push(i);
    if (filtered[i] < filtered[i - 1] && filtered[i] < filtered[i + 1]) troughs.push(i);
  }
  if (peaks.length === 0 || troughs.length === 0) return 0;
  let total = 0;
  for (const p of peaks) {
    const t = troughs.reduce((a, b) => Math.abs(a - p) < Math.abs(b - p) ? a : b);
    total += Math.abs(filtered[p] - filtered[t]);
  }
  const meanAc = total / peaks.length;
  const dc = rawSignal.reduce((a, b) => a + b, 0) / rawSignal.length;
  return dc === 0 ? 0 : (meanAc / dc) * 100;
}

// Dart-style HR with CORRECT bandpass
function dartHQCorrect(signal: number[]): number | null {
  const filtered = butterworthBandpass(signal, 0.5, 3.5, fs, true);
  const peaks = findPeaks(filtered, 8, 0.3);
  if (peaks.length < 3) return null;
  const intervals = [];
  for (let i = 1; i < peaks.length; i++) intervals.push((peaks[i] - peaks[i - 1]) / fs);
  const rr = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  return 60 / rr;
}

// Two-stage respiratory rate: Stage1 cardiac 0.5-4.0 Hz, Stage2 baseline-wander lowpass 0.06-0.7 Hz + autocorr/FFT fusion
function respTwoStage(signal: number[]): number | null {
  if (signal.length < fs * 10) return null;
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  // Stage 2: lowpass 0.06-0.7 Hz isolates baseline wander (respiration)
  const bw = butterworthBandpass(signal.map((v) => v - dc), 0.06, 0.7, fs, true);
  // autocorr
  const m = bw.reduce((a, b) => a + b, 0) / bw.length;
  const centered = bw.map((v) => v - m);
  const norm = centered.reduce((a, b) => a + b * b, 0) || 1;
  const ac: number[] = [];
  for (let lag = 0; lag < bw.length; lag++) {
    let s = 0;
    for (let i = 0; i < bw.length - lag; i++) s += centered[i] * centered[i + lag];
    ac.push(s / norm);
  }
  const minLag = Math.floor(fs / 0.7), maxLag = Math.floor(fs / 0.06);
  let bestLag = -1, bestVal = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, ac.length - 1); lag++) {
    if (ac[lag] > bestVal) { bestVal = ac[lag]; bestLag = lag; }
  }
  const fft = bw; // crude FFT fallback: use autocorr dominant lag
  let bpm = 0;
  if (bestLag > 0) bpm = (60 * fs) / bestLag;
  if (bpm < 5 || bpm > 50) {
    // fallback: crude DFT dominant freq in 0.06-0.7
    let bestF = 0, bestP = -Infinity;
    for (let k = 1; k < Math.floor(bw.length / 2); k++) {
      const freq = (k * fs) / bw.length;
      if (freq < 0.06 || freq > 0.7) continue;
      let real = 0, imag = 0;
      for (let i = 0; i < bw.length; i++) {
        const a = (2 * Math.PI * k * i) / bw.length;
        real += bw[i] * Math.cos(a); imag -= bw[i] * Math.sin(a);
      }
      const p = real * real + imag * imag;
      if (p > bestP) { bestP = p; bestF = freq; }
    }
    if (bestF > 0) bpm = bestF * 60;
  }
  return bpm >= 5 && bpm <= 50 ? bpm : null;
}

function fingerPPG(seconds: number, hrBpm: number, respBpm: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * (hrBpm / 60) * t) + 3 * Math.sin(2 * Math.PI * 2 * (hrBpm / 60) * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * (respBpm / 60) * t);
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

console.log("=== With CORRECT bandpass (RBJ biquad) ===");
const green = fingerPPG(15, 72, 15);
console.log("PI (good finger):", computePerfIndexCorrect(green).toFixed(1), "%  rear gate<0.2 reject?", computePerfIndexCorrect(green) < 0.2);
console.log("HR:", dartHQCorrect(green)?.toFixed(1), "(true 72)");

const flat = new Array(300).fill(110).map(() => 110 + (Math.random() - 0.5) * 0.5);
console.log("PI (flat):", computePerfIndexCorrect(flat).toFixed(1), "%  rear gate<0.2 reject?", computePerfIndexCorrect(flat) < 0.2, "  demo<0.5?", computePerfIndexCorrect(flat) < 0.5);

console.log("\n=== Two-stage respiratory rate (baseline wander, 0.06-0.7 Hz) ===");
for (const { hr, resp } of [{ hr: 72, resp: 15 }, { hr: 60, resp: 12 }, { hr: 90, resp: 18 }, { hr: 110, resp: 20 }]) {
  const g = fingerPPG(30, hr, resp);
  const r = respTwoStage(g);
  console.log(`  HR=${hr} trueResp=${resp} -> RR=${r?.toFixed(1)} err=${r ? Math.abs(r - resp).toFixed(1) : "null"}`);
}
