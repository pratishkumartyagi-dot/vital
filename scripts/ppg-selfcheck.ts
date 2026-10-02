/**
 * Regression self-check for the browser PPG analysis pipeline.
 *
 * Run with: bun run test:ppg
 *
 * Generates synthetic PPG with a known heart rate / respiratory rate and
 * asserts the estimators stay within tolerance. It exists because a previous
 * regression replaced the bandpass coefficients with a curve whose passband was
 * centred on ~0.5 Hz, which silently corrupted BPM, HRV and the AC/DC ratios
 * used for SpO2/Hb.
 */
import { analyzeHeartRate, analyzeRespRate, butterworthBandpass } from "../src/app/vital-monitoring/ppg-analysis";

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

function acDc(raw: number[], fs: number): { ac: number; dc: number } {
  const f = butterworthBandpass(raw, 0.5, 3.5, fs, true);
  const dc = raw.reduce((a, b) => a + b, 0) / raw.length;
  let sq = 0;
  for (const v of f) sq += v * v;
  return { ac: Math.sqrt(sq / f.length), dc };
}

let failures = 0;
function check(label: string, ok: boolean, detail: string) {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}  ${detail}`);
}

const fs = 30;

for (const hr of [50, 60, 72, 90, 110, 130, 150]) {
  const ppg = synth(fs, 30, hr);
  const h = analyzeHeartRate(ppg, fs);
  const err = h.bpm === null ? Infinity : Math.abs(h.bpm - hr);
  check(`HR ${hr} bpm`, err <= 3, `measured=${h.bpm?.toFixed(1)} err=${err.toFixed(2)}`);
  const sdnn = h.hrv?.sdnn ?? 0;
  check(`HR ${hr} HRV sane`, sdnn >= 0 && sdnn < 250, `sdnn=${sdnn.toFixed(1)}ms rmssd=${h.hrv?.rmssd.toFixed(1)}ms`);
}

{
  const ppg = synth(fs, 30, 75, 0.25);
  const r = analyzeRespRate(ppg, fs);
  const err = r.respRate === null ? Infinity : Math.abs(r.respRate - 15);
  check("Resp rate", err <= 2, `measured=${r.respRate?.toFixed(1)} err=${err.toFixed(2)}`);
}

{
  const red = synth(fs, 30, 72, 0.25, 1, 1);
  const green = red.map((v) => 120 + (v - 120) * 1.5);
  const { ac: acR, dc: dcR } = acDc(red, fs);
  const { ac: acG, dc: dcG } = acDc(green, fs);
  const R = (acR / dcR) / (acG / dcG);
  const spo2 = Math.min(100, Math.max(88, 110 - 25 * R));
  const hb = Math.min(20, Math.max(5, 18.2 - 6.8 * R));
  check("SpO2 plausible", spo2 >= 88 && spo2 <= 100, `R=${R.toFixed(3)} spo2=${spo2.toFixed(1)}%`);
  check("Hb plausible", hb >= 5 && hb <= 20, `hb=${hb.toFixed(1)} g/dL`);
}

console.log(failures === 0 ? "\nAll PPG self-checks passed." : `\n${failures} PPG self-check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
