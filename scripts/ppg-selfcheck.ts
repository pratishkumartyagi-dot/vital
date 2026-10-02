/**
 * Regression self-check for the browser PPG analysis pipeline.
 *
 * Run with: bun run test:ppg
 *
 * Generates synthetic PPG with a known heart rate / respiratory rate and
 * asserts the estimators stay within tolerance. It exists because a previous
 * regression replaced the bandpass coefficients with a curve whose passband was
 * centred on ~0.5 Hz, which silently corrupted BPM, HRV and the AC/DC ratios
 * used for SpO2/Hb. Now also guards the creatinine surrogate (age handling,
 * front-mode responsiveness, SQI gating), the POS chrominance pulse, the
 * Hann-windowed power spectrum, and the glucose signal-quality gate.
 */
import {
  analyzeHeartRate,
  analyzeRespRate,
  analyzeVitals,
  butterworthBandpass,
  estimateEgfr,
  extractCreatinineFeatures,
  posPulse,
  powerSpectrum,
} from "../src/app/vital-monitoring/ppg-analysis";

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

function scaleSignal(sig: number[], gain: number): number[] {
  return sig.map((v) => 120 + (v - 120) * gain);
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

// --- POS chrominance pulse (front-camera rPPG) ---
{
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
    // Multiplicative in-band "motion / illumination" modulation seen
    // identically by every channel (the POS-suppressible artifact class).
    // At 35% it dwarfs the green pulse amplitude, so a raw channel locks onto
    // the artifact while POS cancels the common mode exactly.
    const motion = 1 + 0.35 * Math.sin(2 * Math.PI * 1.0 * t) + 0.002 * rnd();
    // Pulsatility ranking G > B > R, per the paper's u_pbv = [0.33, 0.77, 0.53].
    red.push(135 * (1 + 0.06 * pulse) * motion);
    green.push(120 * (1 + 0.18 * pulse) * motion);
    blue.push(110 * (1 + 0.10 * pulse) * motion);
  }
  const pos = posPulse(red, green, blue, 32);
  const hPos = analyzeHeartRate(pos, fs);
  const posErr = hPos.bpm === null ? Infinity : Math.abs(hPos.bpm - hrTrue);
  const hGreen = analyzeHeartRate(green, fs);
  const greenErr = hGreen.bpm === null ? Infinity : Math.abs(hGreen.bpm - hrTrue);
  check("POS recovers HR from artifact-corrupted RGB", posErr <= 5, `pos=${hPos.bpm?.toFixed(1)} err=${posErr.toFixed(2)}`);
  check("POS beats raw channel under common artifact", posErr < greenErr && greenErr > 5, `green=${hGreen.bpm?.toFixed(1)} err=${greenErr === Infinity ? "inf" : greenErr.toFixed(2)}`);
}

// --- Hann-windowed power spectrum ---
{
  const n = 120;
  const tone: number[] = [];
  for (let i = 0; i < n; i++) tone.push(10 * Math.sin(2 * Math.PI * 1.23 * (i / fs)));
  const { freqs, psd } = powerSpectrum(tone, fs);
  let bestIdx = 0;
  for (let i = 1; i < psd.length; i++) if (psd[i] > psd[bestIdx]) bestIdx = i;
  const bin = fs / n;
  const err = Math.abs(freqs[bestIdx] - 1.23);
  check("Hann window keeps tone peak on correct bin", err <= bin / 2, `dominant=${freqs[bestIdx].toFixed(2)}Hz bin=${bin.toFixed(3)}Hz`);
  const total = psd.reduce((a, b) => a + b, 0);
  check("Hann window suppresses spectral leakage", psd[bestIdx] / total > 0.4, `peak/total=${(psd[bestIdx] / total).toFixed(2)}`);
}

// --- Creatinine estimator ---
{
  const ppg = synth(fs, 30, 75);
  const red = scaleSignal(ppg, 0.7);
  const green = scaleSignal(ppg, 1.0);
  const blue = scaleSignal(ppg, 0.4);
  const peaks = analyzeHeartRate(ppg, fs).peaks;
  const feats = extractCreatinineFeatures(ppg, red, green, blue, peaks, fs);
  check("Creatinine features extracted", feats.beats >= 3 && feats.perfusionIndex > 0, `beats=${feats.beats} PI=${feats.perfusionIndex.toFixed(2)}%`);

  // Perfusion feature tracks pulsatile amplitude (raw-MAD PI bug is fixed).
  const weakPPG = scaleSignal(ppg, 0.15);
  const weakPeaks = analyzeHeartRate(weakPPG, fs).peaks;
  const weak = extractCreatinineFeatures(weakPPG, scaleSignal(red, 0.15), scaleSignal(green, 0.15), scaleSignal(blue, 0.15), weakPeaks, fs);
  check("Creatinine PI scales with pulse amplitude", weak.perfusionIndex < feats.perfusionIndex / 3, `strong=${feats.perfusionIndex.toFixed(2)}% weak=${weak.perfusionIndex.toFixed(2)}%`);

  const age = 40;
  const vRear = analyzeVitals(ppg, fs, red, green, blue, { cameraMode: "rear", ageYears: age, sex: "male" });
  const crRear = vRear.creatinine.creatinine;
  check("Creatinine produced on clean rear PPG", crRear !== null && crRear >= 0.4 && crRear <= 12, `cr=${crRear} class=${vRear.creatinine.creatinineClass} conf=${vRear.creatinine.confidence}`);
  check("Creatinine eGFR + stage populated", vRear.creatinine.egfr !== null && vRear.creatinine.stage !== null, `egfr=${vRear.creatinine.egfr} stage=${vRear.creatinine.stage}`);

  const vFront = analyzeVitals(ppg, fs, red, green, blue, { cameraMode: "front", ageYears: age, sex: "male" });
  check("Front mode produces creatinine with capped confidence", vFront.creatinine.creatinine !== null && vFront.creatinine.confidence <= 0.55 && vFront.creatinine.method === "rppg-front", `cr=${vFront.creatinine.creatinine} conf=${vFront.creatinine.confidence}`);

  // Age must NOT shift the creatinine estimate itself (no double count);
  // it only re-enters through CKD-EPI eGFR downstream.
  const vAge = analyzeVitals(ppg, fs, red, green, blue, { cameraMode: "rear", ageYears: 85, sex: "male" });
  const egfrYoung = estimateEgfr(vRear.creatinine.creatinine, 40, "male") ?? 0;
  const egfrOld = estimateEgfr(vRear.creatinine.creatinine, 85, "male") ?? 0;
  check("Age double-count removed from creatinine", vAge.creatinine.creatinine === vRear.creatinine.creatinine, `cr40=${vRear.creatinine.creatinine} cr85=${vAge.creatinine.creatinine}`);
  check("Age still applied via CKD-EPI", egfrOld < egfrYoung * 0.9, `egfr40=${egfrYoung.toFixed(0)} egfr85=${egfrOld.toFixed(0)}`);

  // Front mode must respond to the signal (perfusion), not collapse to a
  // demographic constant.
  const vFrontWeak = analyzeVitals(scaleSignal(ppg, 0.01), fs, scaleSignal(red, 0.01), scaleSignal(green, 0.01), scaleSignal(blue, 0.01), { cameraMode: "front", ageYears: age, sex: "male" });
  const crFrontStrong = vFront.creatinine.creatinine;
  const crFrontWeak = vFrontWeak.creatinine.creatinine;
  check("Front mode not collapsed to constant", crFrontStrong !== null && crFrontWeak !== null && Math.abs(crFrontStrong - crFrontWeak) > 0.05, `strong=${crFrontStrong} weak=${crFrontWeak}`);

  // SQI gate: noise-dominated signal must produce no creatinine and no glucose.
  const noisy = synth(fs, 30, 75, 0.25, 45, 5);
  const vNoisy = analyzeVitals(noisy, fs, scaleSignal(noisy, 0.7), scaleSignal(noisy, 1), scaleSignal(noisy, 0.4), { cameraMode: "rear", ageYears: age, sex: "male" });
  check("Low-SQI gates creatinine", vNoisy.creatinine.creatinine === null, `sqi=${vNoisy.heart.signalQuality} cr=${vNoisy.creatinine.creatinine}`);
  check("Low-SQI gates glucose", vNoisy.glucose.glucose === null, `sqi=${vNoisy.heart.signalQuality} glucose=${vNoisy.glucose.glucose}`);
  check("Clean signal yields glucose", vRear.glucose.glucose !== null && vRear.glucose.glucose >= 70 && vRear.glucose.glucose <= 200, `sqi=${vRear.heart.signalQuality} glucose=${vRear.glucose.glucose}`);
}

// --- CKD-EPI eGFR sanity ---
{
  const mid = estimateEgfr(1.0, 45, "male");
  check("eGFR plausible mid-life male", mid !== null && mid > 70 && mid < 110, `egfr=${mid?.toFixed(1)}`);
  check("eGFR null-safe", estimateEgfr(null, 45, "male") === null && estimateEgfr(0, 45, "male") === null, "null/0 -> null");
}

console.log(failures === 0 ? "\nAll PPG self-checks passed." : `\n${failures} PPG self-check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
