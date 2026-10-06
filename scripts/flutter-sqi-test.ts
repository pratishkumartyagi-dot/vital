// Port of vital_monitoring_app/lib/ppg_signal_engine.dart
// applyBandpassFilter (RBJ biquad HP0.5+LP3.5, zero-phase) +
// computePerfusionIndex (RMS AC / mean DC), to check correctness.

function lowpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cw) / 2 / a0, b1: (1 - cw) / a0, b2: (1 - cw) / 2 / a0,
    a1: (-2 * cw) / a0, a2: (1 - alpha) / a0,
  };
}

function highpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cw) / 2 / a0, b1: -(1 + cw) / a0, b2: (1 + cw) / 2 / a0,
    a1: (-2 * cw) / a0, a2: (1 - alpha) / a0,
  };
}

function applyBiquad(signal: number[], c: any, seedX = 0, seedY = 0): number[] {
  const out = new Array(signal.length).fill(0);
  let x1 = seedX, x2 = seedX, y1 = seedY, y2 = seedY;
  for (let i = 0; i < signal.length; i++) {
    const y = c.b0 * signal[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y;
    x2 = x1; x1 = signal[i];
    y2 = y1; y1 = y;
  }
  return out;
}

function applyBandpassFilter(input: number[]): number[] {
  const fs = 30, low = 0.5, high = 3.5;
  if (input.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
    return [...input];
  }
  const hp = highpassBiquad(low, fs);
  const lp = lowpassBiquad(high, fs);
  const dc = input.reduce((a, b) => a + b, 0) / input.length;
  let y = applyBiquad(input, hp, dc, 0);
  y = applyBiquad(y, lp, 0, 0);
  y = [...y].reverse();
  y = applyBiquad(y, hp, 0, 0);
  y = applyBiquad(y, lp, 0, 0);
  y = [...y].reverse();
  return y;
}

function computePerfusionIndex(rawSignal: number[]): number {
  if (rawSignal.length < 10) return 0.0;
  const filtered = applyBandpassFilter(rawSignal);
  const dc = rawSignal.reduce((a, b) => a + b, 0) / rawSignal.length;
  if (dc <= 1e-9) return 0.0;
  let sq = 0;
  for (const v of filtered) sq += v * v;
  const ac = Math.sqrt(sq / filtered.length);
  return (ac / dc) * 100.0;
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

// Flat-signal rejection (PI should collapse to ~0)
const flat = new Array(300).fill(110).map(() => 110 + (Math.random() - 0.5) * 0.5);
console.log("\nFlat signal PI:", computePerfusionIndex(flat).toFixed(3), "% (should be ~0)");
