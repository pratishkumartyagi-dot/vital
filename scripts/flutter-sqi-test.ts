// Port of vital_monitoring_app/lib/ppg_signal_engine.dart
// applyBandpassFilter (SOS) + computePerfusionIndex, to check for correctness

function applyBandpassFilter(input: number[]): number[] {
  const length = input.length;
  const sos = [
    [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
    [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
  ];
  let stage = input.slice();
  for (const section of sos) {
    const [b0, b1, b2, a0, a1, a2] = section;
    const out = new Array(length).fill(0);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < length; i++) {
      const x0 = stage[i];
      const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      out[i] = y0;
      x2 = x1; x1 = x0;
      y2 = y1; y1 = y0;
    }
    stage = out;
  }
  return stage;
}

function computePerfusionIndex(rawSignal: number[]): number {
  if (rawSignal.length < 10) return 0.0;
  const filtered = applyBandpassFilter(rawSignal);
  const peaks: number[] = [];
  const troughs: number[] = [];
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1]) peaks.push(i);
    if (filtered[i] < filtered[i - 1] && filtered[i] < filtered[i + 1]) troughs.push(i);
  }
  if (peaks.length === 0 || troughs.length === 0) return 0.0;
  let totalAc = 0, count = 0;
  for (const peakIdx of peaks) {
    const closestTrough = troughs.reduce((a, b) => Math.abs(a - peakIdx) < Math.abs(b - peakIdx) ? a : b);
    totalAc += Math.abs(filtered[peakIdx] - filtered[closestTrough]);
    count++;
  }
  if (count === 0) return 0.0;
  const meanAc = totalAc / count;
  const dc = rawSignal.reduce((a, b) => a + b, 0) / rawSignal.length;
  return dc === 0 ? 0.0 : (meanAc / dc) * 100.0;
}

const fs = 30;
// Realistic finger PPG: DC=110, cardiac 1.2Hz amp 10, respiratory 0.25Hz amp 4
function fingerPPG(seconds: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * 1.2 * t) + 3 * Math.sin(2 * Math.PI * 2.4 * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * 0.25 * t);
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

const ppg = fingerPPG(15); // 15 seconds = 450 samples
const pi = computePerfusionIndex(ppg);
console.log("Dart ported PI on realistic finger PPG:", pi.toFixed(3), "%");
console.log("  rear gate (pi<0.2 reject):", pi < 0.2 ? "REJECTED" : "accepted");
console.log("  demo gate (pi<0.5 reject):", pi < 0.5 ? "REJECTED" : "accepted");

// Check what the bandpass does to a pure 1.2 Hz (cardiac) tone vs 0.25 Hz (resp)
function tone(f: number, seconds: number): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(110 + 15 * Math.sin(2 * Math.PI * f * (i / fs)));
  return out;
}
const cardiac = tone(1.2, 15);
const respTone = tone(0.25, 15);
const bpCardiac = applyBandpassFilter(cardiac);
const bpResp = applyBandpassFilter(respTone);
const ac = (sig: number[]) => { let sq = 0; for (const v of sig) sq += v * v; return Math.sqrt(sq / sig.length); };
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
console.log("\nBandpass gain (RMS in-band / RMS out-of-band):");
console.log("  1.2 Hz (cardiac) RMS in:", ac(cardiac).toFixed(2), "out:", ac(bpCardiac).toFixed(2), "ratio:", (ac(bpCardiac)/ac(cardiac)).toFixed(2));
console.log("  0.25 Hz (resp)    RMS in:", ac(respTone).toFixed(2), "out:", ac(bpResp).toFixed(2), "ratio:", (ac(bpResp)/ac(respTone)).toFixed(2));
