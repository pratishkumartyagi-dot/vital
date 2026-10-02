// Port of vital_monitoring_app Dart PPG pipeline to verify Flutter bugs.
const fs = 30;

function applyBandpassSOS(input: number[]): number[] {
  const sos = [
    [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
    [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
  ];
  let stage = input.slice();
  for (const section of sos) {
    const [b0, b1, b2, a0, a1, a2] = section;
    const out = new Array(input.length).fill(0);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < input.length; i++) {
      const x0 = stage[i];
      const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      out[i] = y0; x2 = x1; x1 = x0; y2 = y1; y1 = y0;
    }
    stage = out;
  }
  return stage;
}

function findPeaks(signal: number[], minDistance = 1, threshold = 0.0): number[] {
  const peaks: number[] = [];
  if (signal.length < 3) return peaks;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const stdDev = Math.sqrt(signal.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / signal.length);
  const dynamicThreshold = threshold * stdDev;
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > dynamicThreshold) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) peaks.push(i);
    }
  }
  return peaks;
}

// Dart _calculateHeartRate: bandpass then peak-detect (threshold 0.3, minDist 8)
function dartHR(signal: number[]): number | null {
  const filtered = applyBandpassSOS(signal);
  const peaks = findPeaks(filtered, 8, 0.3);
  if (peaks.length < 3) return null;
  // average RR interval
  const intervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) intervals.push((peaks[i] - peaks[i - 1]) / fs);
  const rr = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  return 60 / rr;
}

function computePerfIndex(rawSignal: number[]): number {
  if (rawSignal.length < 10) return 0;
  const filtered = applyBandpassSOS(rawSignal);
  const peaks: number[] = [], troughs: number[] = [];
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i-1] && filtered[i] > filtered[i+1]) peaks.push(i);
    if (filtered[i] < filtered[i-1] && filtered[i] < filtered[i+1]) troughs.push(i);
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

function fingerPPG(seconds: number, hrBpm: number, respBpm: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * (hrBpm/60) * t) + 3 * Math.sin(2 * Math.PI * 2 * (hrBpm/60) * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * (respBpm/60) * t);
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

console.log("=== Flutter pipeline (broken SOS bandpass) ===");
const green = fingerPPG(15, 72, 15);
console.log("HR (Dart _calculateHeartRate):", dartHR(green)?.toFixed(1), "(true 72)");
console.log("PI (finger gate):", computePerfIndex(green).toFixed(1), "%  rear threshold<0.2 reject, demo<0.5 reject");

console.log("\n=== No-finger (flat) signal ===");
const flat = new Array(300).fill(110).map((v) => v + (Math.random() - 0.5) * 0.5);
console.log("PI (flat):", computePerfIndex(flat).toFixed(1), "% -> rear gate rejects?", computePerfIndex(flat) < 0.2);

console.log("\n=== With motion artifact (front-ish) ===");
function noisyPPG(seconds: number, hrBpm: number) {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = 99;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 6 * Math.sin(2 * Math.PI * (hrBpm/60) * t);
    const motion = 25 * Math.sin(2 * Math.PI * 1.0 * t) * Math.sin(2 * Math.PI * (hrBpm/60) * t * 0); // low freq
    out.push(110 + cardiac + motion + rnd() * 3);
  }
  return out;
}
const nm = noisyPPG(15, 72);
console.log("HR on noisy:", dartHR(nm)?.toFixed(1), "(true 72)");
console.log("PI on noisy:", computePerfIndex(nm).toFixed(1), "%");
