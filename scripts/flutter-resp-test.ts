// Port of vital_monitoring_app/lib/vital_sign_controller.dart _estimateRespiratoryRate + _findPeaks

function findPeaks(signal: number[], minDistance = 1, threshold = 0.0): number[] {
  const peaks: number[] = [];
  if (signal.length < 3) return peaks;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const stdDev = Math.sqrt(signal.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / signal.length);
  const dynamicThreshold = threshold * stdDev;
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > dynamicThreshold) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

function estimateRespiratoryRate(signal: number[]): number {
  const fs = 30;
  if (signal.length < 180) return 15.0;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const detrended = signal.map((v) => v - mean);
  const alpha = 0.15;
  const smoothed: number[] = [];
  let prev = detrended[0];
  smoothed.push(prev);
  for (let i = 1; i < detrended.length; i++) {
    const val = alpha * detrended[i] + (1 - alpha) * prev;
    smoothed.push(val);
    prev = val;
  }
  const peaks = findPeaks(smoothed, 60, 0.0);
  console.log("  resp peaks found:", peaks.length, "first few:", peaks.slice(0, 6));
  if (peaks.length < 2) return 15.0;
  const intervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    intervals.push((peaks[i] - peaks[i - 1]) / 30.0);
  }
  const valid = intervals.filter((t) => t >= 1.0 && t <= 6.0);
  if (valid.length === 0) return 15.0;
  const avgInterval = valid.reduce((a, b) => a + b, 0) / valid.length;
  const rate = 60.0 / avgInterval;
  return Math.min(35.0, Math.max(8.0, rate));
}

const fs = 30;
function realisticPPG(seconds: number, hrBpm: number, respBpm: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * (hrBpm / 60) * t)
      + 3 * Math.sin(2 * Math.PI * 2 * (hrBpm / 60) * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * (respBpm / 60) * t);
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

console.log("=== Flutter _estimateRespiratoryRate on realistic finger PPG ===");
for (const { hr, resp } of [{ hr: 72, resp: 15 }, { hr: 60, resp: 12 }, { hr: 90, resp: 18 }, { hr: 110, resp: 20 }]) {
  const green = realisticPPG(30, hr, resp);
  // mimic live buffer (120 samples = 4s) up to 300 samples
  const rate = estimateRespiratoryRate(green);
  console.log(`  HR=${hr} Resp=${resp} -> Flutter RR=${rate.toFixed(1)} (err=${Math.abs(rate-resp).toFixed(1)})`);
}

console.log("\n=== Flutter demo synthetic PPG (100 + 20*sin(1.2Hz)) ===");
const demo: number[] = [];
for (let i = 0; i < 300; i++) {
  const t = i * 0.033;
  demo.push(100 + 20 * Math.sin(2 * Math.PI * 1.2 * t) + 5 * Math.sin(2 * Math.PI * 2.5 * t) + (Math.random() - 0.5) * 2);
}
console.log("  Flutter RR on demo:", estimateRespiratoryRate(demo));
