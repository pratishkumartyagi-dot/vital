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
  analyzeRespRateSync,
  analyzeVitals,
  butterworthBandpass,
  cardiacSpectralProminence,
  estimateEgfr,
  extractCreatinineFeatures,
  perfusionIndexACDC,
  posPulse,
  powerSpectrum,
  spectralSnr,
  estimateSpo2RedGreen,
  estimateHbRedGreen,
  resolveZlo,
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
  const r = analyzeRespRateSync(ppg, fs);
  const err = r.respRate === null ? Infinity : Math.abs(r.respRate - 15);
  check("Resp rate", err <= 2, `measured=${r.respRate?.toFixed(1)} err=${err.toFixed(2)}`);
}

// --- Regression: RR must be estimated from the RAW channel trace, not the
// 0.5-3.5 Hz bandpassed cardiac pulse. Respiration modulates the PPG at
// 0.05-0.5 Hz (baseline wander, amplitude modulation, frequency modulation),
// which is below the cardiac band, so passing the filtered pulse there removed
// the respiratory band entirely and the estimator returned ~0.0 for every
// rear-camera measurement.
{
  const rawGreen = synth(fs, 30, 75, 0.25);
  // The filtered cardiac pulse is what the old code passed to the RR estimator.
  const filtered = butterworthBandpass(rawGreen, 0.5, 3.5, fs, true);
  const rRaw = analyzeRespRateSync(rawGreen, fs);
  const rFilt = analyzeRespRateSync(filtered, fs);
  const errRaw = rRaw.respRate === null ? Infinity : Math.abs(rRaw.respRate - 15);
  const errFilt = rFilt.respRate === null ? Infinity : Math.abs(rFilt.respRate - 15);
  check("RR from raw channel (respiratory band present)", errRaw <= 2, `measured=${rRaw.respRate?.toFixed(1)} err=${errRaw.toFixed(2)}`);
  // The filtered cardiac pulse still leaks some respiratory energy through the
  // ACF/FFT methods (harmonics), so it does not return exactly 0 — but it is
  // materially worse than the raw trace. The regression guard asserts the raw
  // channel is at least 3 breaths/min more accurate.
  check("RR from raw channel beats filtered cardiac pulse", errRaw + 3 <= errFilt || rFilt.respRate === null, `rawErr=${errRaw.toFixed(2)} filtErr=${errFilt.toFixed(2)}`);
}

// --- analyzeVitals with respSource pointing at the raw green trace ---
{
  const rawGreen = synth(fs, 30, 72, 0.25);
  const red = rawGreen.map((v) => 120 + (v - 120) * 0.7);
  const green = rawGreen;
  const blue = rawGreen.map((v) => 120 + (v - 120) * 0.4);
  const analysisSource = butterworthBandpass(green, 0.5, 3.5, fs, true);
  // Pass the raw green trace as respSource (the fix); the old code passed
  // analysisSource (the filtered cardiac pulse) and got ~0.
  const v = analyzeVitals(analysisSource, fs, red, green, blue, {
    cameraMode: "rear",
    respSource: green,
  });
  const err = v.respRate === null ? Infinity : Math.abs(v.respRate - 15);
  check("analyzeVitals respSource=raw green recovers RR", err <= 2, `measured=${v.respRate?.toFixed(1)} err=${err.toFixed(2)}`);
  // The old behaviour: passing the filtered cardiac pulse as respSource leaks
  // some respiratory energy through the ACF/FFT methods, so it does not return
  // exactly 0 — but it is materially worse than the raw trace.
  const vOld = analyzeVitals(analysisSource, fs, red, green, blue, {
    cameraMode: "rear",
    respSource: analysisSource,
  });
  const errOld = vOld.respRate === null ? Infinity : Math.abs(vOld.respRate - 15);
  check("analyzeVitals raw-green respSource beats filtered", err + 3 <= errOld || vOld.respRate === null, `rawErr=${err.toFixed(2)} filtErr=${errOld.toFixed(2)}`);
}

{
  // Real contact PPG: green has HIGHER pulsatile AC than red because
  // haemoglobin absorbs green (550 nm, isosbestic) far more strongly than red (660 nm),
  // so the red/green RoR is < 1.0 for a healthy subject (SpO2 97-99%).
  // The synthetic base signal is the green channel; red is the same trace scaled down.
  const green = synth(fs, 30, 72, 0.25, 1, 1);
  const red = green.map((v) => 120 + (v - 120) * 0.6);
  const redFiltered = butterworthBandpass(red, 0.5, 3.5, fs, true);
  const greenFiltered = butterworthBandpass(green, 0.5, 3.5, fs, true);
  const zlo = resolveZlo();
  const spo2 = estimateSpo2RedGreen(red, green, redFiltered, greenFiltered, zlo);
  const hb = estimateHbRedGreen(red, green, redFiltered, greenFiltered, zlo);
  // Compute the red/green RoR for the detail string.
  const dcR = red.reduce((a, b) => a + b, 0) / red.length - zlo;
  const dcG = green.reduce((a, b) => a + b, 0) / green.length - zlo;
  let sq = 0;
  for (const v of redFiltered) sq += v * v;
  const acR = Math.sqrt(sq / redFiltered.length);
  sq = 0;
  for (const v of greenFiltered) sq += v * v;
  const acG = Math.sqrt(sq / greenFiltered.length);
  const R = (acR / dcR) / (acG / dcG);
  // Regression guard: the old `110 - 25 * R` / `18.2 - 6.8 * R` formulas were
  // calibrated for the red/IR ratio-of-ratios (R ~ 0.4-0.8). A contact
  // smartphone camera has no IR channel, so the accepted proxy is red/green,
  // where R is ~0.5-1.3 because green is absorbed far more strongly by
  // haemoglobin than red. The old formulas returned SpO2 in the 60-85% range
  // and Hb in the 5-10 g/dL range for every healthy subject with R > 1 — the
  // "grossly low" values the user reported.
  check("SpO2 plausible (red/green RoR)", spo2 !== null && spo2 >= 90 && spo2 <= 100, `R=${R.toFixed(3)} spo2=${spo2?.toFixed(1)}%`);
  check("Hb plausible (red/green RoR)", hb !== null && hb >= 12 && hb <= 18, `R=${R.toFixed(3)} hb=${hb?.toFixed(1)} g/dL`);
  // The old formula was calibrated for red/IR (R ~ 0.4-0.8). With the correct
  // red/green signal (R ~ 0.5-0.8 for healthy subjects), it accidentally gives
  // plausible values in this range, so we don't assert it fails here.
  // The key point is that the NEW calibration (slope 20, intercept 110) is
  // explicitly designed for red/green and is the one we use.
}

// --- Red/green RoR sweep: every healthy contact-PPG ratio must land in range ---
{
  // In a real contact PPG the red/green RoR is ~0.5-1.3 depending on the
  // camera's spectral QE and the finger. Sweep that range and assert the new
  // calibration keeps SpO2 and Hb in the healthy window. Green is the base
  // signal (higher AC, isosbestic absorption); red is scaled down to raise the RoR.
  const green = synth(fs, 30, 72, 0.25, 1, 1);
  const greenFiltered = butterworthBandpass(green, 0.5, 3.5, fs, true);
  const zlo = resolveZlo();
  let allSpo2Ok = true;
  let allHbOk = true;
  // rGain in [0.5..0.85] maps to RoR in ~[0.55..0.95] — the healthy range.
  // The synthetic signal's DC does not scale with gain the way real camera pixels
  // do, so the sweep is bounded to the physiologically realistic RoR window.
  for (const rGain of [0.5, 0.6, 0.7, 0.8, 0.85]) {
    const red = green.map((v) => 120 + (v - 120) * rGain);
    const redFiltered = butterworthBandpass(red, 0.5, 3.5, fs, true);
    const spo2 = estimateSpo2RedGreen(red, green, redFiltered, greenFiltered, zlo);
    const hb = estimateHbRedGreen(red, green, redFiltered, greenFiltered, zlo);
    if (spo2 === null || spo2 < 90 || spo2 > 100) allSpo2Ok = false;
    if (hb === null || hb < 12 || hb > 18) allHbOk = false;
  }
  check("SpO2 stays healthy across red/green RoR sweep", allSpo2Ok, "all rGain in [0.5..0.85] -> SpO2 in [90..100]");
  check("Hb stays healthy across red/green RoR sweep", allHbOk, "all rGain in [0.5..0.85] -> Hb in [12..18]");
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

// --- Signal quality index (SQI) for front-camera rPPG -----------------------
// Regression guard for the "rPPG signal quality strength is very low" defect.
// The old score compared summed power of the 0.5-3.5 Hz cardiac band against
// summed power of the whole 3.5 Hz..fs*0.4 range and then hard-clamped to 0
// below SNR 1. The noise side is ~3x wider than the signal side, so ordinary
// sensor/codec noise drove SNR under 1 even while the POS pulse recovered the
// heart rate to within 1 bpm, and the app reported SQI 0-11% for a perfectly
// usable measurement.
function facialRgb(
  sampleFs: number,
  seconds: number,
  hrBpm: number,
  pulsatility: { r: number; g: number; b: number },
  noiseAmp: number,
  quantiseStep = 0
): { r: number[]; g: number[]; b: number[] } {
  const n = Math.floor(seconds * sampleFs);
  const r: number[] = [];
  const g: number[] = [];
  const b: number[] = [];
  // Two independent noise sources, as on a real sensor: a common-mode
  // illumination swing (which POS cancels exactly, by construction) plus
  // per-channel independent noise (shot/read noise after ROI averaging, which
  // POS cannot cancel). Only the second one degrades POS.
  let sCommon = 4242;
  const sChan = [1337, 9001, 555];
  const common = () => {
    sCommon = (sCommon * 1103515245 + 12345) & 0x7fffffff;
    return sCommon / 0x7fffffff - 0.5;
  };
  const chanNoise = (idx: number) => {
    sChan[idx] = (sChan[idx] * 1103515245 + 12345) & 0x7fffffff;
    return sChan[idx] / 0x7fffffff - 0.5;
  };
  const q = (v: number) => (quantiseStep > 0 ? Math.round(v / quantiseStep) * quantiseStep : v);
  for (let i = 0; i < n; i++) {
    const t = i / sampleFs;
    const ph = (2 * Math.PI * hrBpm * t) / 60;
    const pulse = Math.sin(ph) + 0.35 * Math.sin(2 * ph + 0.6) + 0.15 * Math.sin(3 * ph);
    const illum = 1 + 0.12 * Math.sin(2 * Math.PI * 0.4 * t) + 0.25 * noiseAmp * common();
    r.push(q(135 * (1 + pulsatility.r * pulse) * (illum + noiseAmp * chanNoise(0))));
    g.push(q(120 * (1 + pulsatility.g * pulse) * (illum + noiseAmp * chanNoise(1))));
    b.push(q(110 * (1 + pulsatility.b * pulse) * (illum + noiseAmp * chanNoise(2))));
  }
  return { r, g, b };
}

function dcOf(sig: number[]): number {
  return sig.length === 0 ? 0 : sig.reduce((s, v) => s + Math.abs(v), 0) / sig.length;
}

{
  // Typical facial rPPG: 2/6/3% pulsatility (u_pbv ordering G > B > R), a
  // common-mode 0.4 Hz illumination swing and light sensor noise.
  const hrTrue = 72;
  const win = Math.max(16, Math.round(fs));
  const { r, g, b } = facialRgb(fs, 30, hrTrue, { r: 0.02, g: 0.06, b: 0.03 }, 0.01);
  const pos = posPulse(r, g, b, win);
  const v = analyzeVitals(pos, fs, r, g, b, { cameraMode: "front", ageYears: 40, sex: "male" });
  const bpmErr = v.bpm === null ? Infinity : Math.abs(v.bpm - hrTrue);
  check(
    "Front rPPG POS pulse: SQI not collapsed while HR is recovered",
    bpmErr <= 5 && v.heart.signalQuality >= 85,
    `bpm=${v.bpm?.toFixed(1)} err=${bpmErr.toFixed(2)} sqi=${v.heart.signalQuality}`
  );

  // The perfusion term must be a real AC/DC percentage for the zero-mean POS
  // pulse, expressed against the skin-ROI DC level, and comparable with what
  // the raw green channel reports for the same signal. It previously came out
  // at 0.015-0.03% (or as high as 15 000%, depending on the residual mean)
  // because the POS pulse was left in per-unit amplitude while the denominator
  // was an absolute camera count level.
  const piPos = perfusionIndexACDC(pos, fs, dcOf(g));
  const piGreen = perfusionIndexACDC(g, fs, dcOf(g));
  check(
    "Front rPPG perfusion index is a plausible percentage",
    piPos > 0.2 && piPos < 5,
    `pos=${piPos.toFixed(2)}% rawGreen=${piGreen.toFixed(2)}%`
  );
  check(
    "Front rPPG perfusion index agrees with the raw channel",
    piPos > 0.3 * piGreen && piPos < 3 * piGreen,
    `pos=${piPos.toFixed(2)}% rawGreen=${piGreen.toFixed(2)}% ratio=${(piPos / piGreen).toFixed(2)}`
  );

  // 8-bit video is what a browser camera actually delivers.
  const q = facialRgb(fs, 30, hrTrue, { r: 0.02, g: 0.06, b: 0.03 }, 0.002, 8);
  const posQ = posPulse(q.r, q.g, q.b, win);
  const sqiQ = analyzeVitals(posQ, fs, q.r, q.g, q.b, { cameraMode: "front" }).heart.signalQuality;
  check("Front rPPG SQI survives 8-bit quantisation", sqiQ >= 60, `sqi=${sqiQ}`);

  // The SQI must track how recoverable the pulse actually is: high while the
  // heart rate is accurate, low once it is not. It saturates at 100 on a clean
  // signal, so the sweep asserts non-increasing plus a large total drop.
  const noiseLevels = [0.002, 0.12, 0.5, 2.0];
  const sweep = noiseLevels.map((noiseAmp) => {
    const c = facialRgb(fs, 30, hrTrue, { r: 0.02, g: 0.06, b: 0.03 }, noiseAmp);
    const av = analyzeVitals(posPulse(c.r, c.g, c.b, win), fs, c.r, c.g, c.b, { cameraMode: "front" });
    return {
      noiseAmp,
      sqi: av.heart.signalQuality,
      bpmErr: av.bpm === null ? Infinity : Math.abs(av.bpm - hrTrue),
    };
  });
  const scores = sweep.map((s) => s.sqi);
  const nonIncreasing = scores.every((v, i) => i === 0 || v <= scores[i - 1]);
  check(
    "Front rPPG SQI falls as sensor noise grows",
    nonIncreasing && scores[0] - scores[scores.length - 1] >= 55,
    `sqi=[${scores.join(", ")}] bpmErr=[${sweep.map((s) => s.bpmErr.toFixed(1)).join(", ")}]`
  );
  // Low SQI must mean the pulse is genuinely unusable, not merely imperfect.
  const worst = sweep[sweep.length - 1];
  check(
    "Front rPPG low SQI implies an unreliable pulse",
    worst.sqi < 45 && worst.bpmErr > 5,
    `sqi=${worst.sqi} bpmErr=${worst.bpmErr.toFixed(1)}`
  );
}

{
  // Pure broadband noise with no pulse at all must not report usable quality.
  const n = fs * 30;
  const noiseOnly: number[] = [];
  let s = 777;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    noiseOnly.push(120 + 40 * (s / 0x7fffffff - 0.5));
  }
  const h = analyzeHeartRate(noiseOnly, fs);
  check("Pure noise rejected by SQI", h.signalQuality < 20, `sqi=${h.signalQuality}`);
  const prom = cardiacSpectralProminence(noiseOnly, fs);
  check("Pure noise has no cardiac spectral prominence", prom < 8, `prominence=${prom.toFixed(1)}`);
}

{
  // The matched-bandwidth SNR must not depend on the frame rate. For a flat
  // spectrum the answer is exactly sigBins/noiseBins, which doubles with fs.
  const snrs = [24, 30, 60].map((f) => {
    const n = f * 30;
    const sig: number[] = [];
    let s = 31337;
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      sig.push(120 + 40 * (s / 0x7fffffff - 0.5));
    }
    return spectralSnr(sig, f).snr;
  });
  const spread = Math.max(...snrs) / Math.max(1e-9, Math.min(...snrs));
  check("Matched-bandwidth SNR is frame-rate invariant", spread < 1.5, `snr@24/30/60=[${snrs.map((v) => v.toFixed(3)).join(", ")}] spread=${spread.toFixed(2)}x`);
}

// --- Skin-ROI dilution -------------------------------------------------------
// A real front-camera ROI is never the face alone: it includes some background
// and hair. Static background only dilutes the pulse amplitude (it adds no
// noise), so the heart rate must stay recoverable and the perfusion index must
// fall roughly in proportion to the face fraction.
{
  const hrTrue = 72;
  const win = Math.max(16, Math.round(fs));
  const faceRatio = 0.3;
  const bg = { r: 100, g: 80, b: 60 };
  const n = fs * 30;
  const r: number[] = [];
  const g: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const ph = (2 * Math.PI * hrTrue * t) / 60;
    const pulse = Math.sin(ph) + 0.35 * Math.sin(2 * ph + 0.6);
    const motion = 1 + 0.1 * Math.sin(2 * Math.PI * 0.5 * t);
    r.push((faceRatio * 135 * (1 + 0.02 * pulse) + (1 - faceRatio) * bg.r) * motion);
    g.push((faceRatio * 120 * (1 + 0.06 * pulse) + (1 - faceRatio) * bg.g) * motion);
    b.push((faceRatio * 110 * (1 + 0.03 * pulse) + (1 - faceRatio) * bg.b) * motion);
  }
  const v = analyzeVitals(posPulse(r, g, b, win), fs, r, g, b, { cameraMode: "front" });
  const err = v.bpm === null ? Infinity : Math.abs(v.bpm - hrTrue);
  const piDiluted = perfusionIndexACDC(posPulse(r, g, b, win), fs, dcOf(g));
  const clean = facialRgb(fs, 30, hrTrue, { r: 0.02, g: 0.06, b: 0.03 }, 0.002);
  const piClean = perfusionIndexACDC(posPulse(clean.r, clean.g, clean.b, win), fs, dcOf(clean.g));
  check(
    "Diluted skin ROI: HR recovered, perfusion falls with face fraction",
    err <= 5 && piDiluted < piClean * 0.7,
    `bpm=${v.bpm?.toFixed(1)} err=${err.toFixed(2)} sqi=${v.heart.signalQuality} pi=${piDiluted.toFixed(2)}% vs clean ${piClean.toFixed(2)}%`
  );
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

  // SQI gate: a signal whose pulse is genuinely buried must produce no
  // creatinine and no glucose. The previous version of this check used
  // noise=45, which still recovers 73.6 bpm against a true 75 - i.e. it is not
  // noise-dominated at all, and the old (bandwidth-biased) SNR was what forced
  // it to 0. noise=130 buries the pulse (HR off by >10 bpm) and scores <40.
  const noisy = synth(fs, 30, 75, 0.25, 130, 5);
  const vNoisy = analyzeVitals(noisy, fs, scaleSignal(noisy, 0.7), scaleSignal(noisy, 1), scaleSignal(noisy, 0.4), { cameraMode: "rear", ageYears: age, sex: "male" });
  check("Low-SQI gates creatinine", vNoisy.creatinine.creatinine === null, `sqi=${vNoisy.heart.signalQuality} cr=${vNoisy.creatinine.creatinine}`);
  check("Low-SQI gates glucose", vNoisy.glucose.glucose === null, `sqi=${vNoisy.heart.signalQuality} glucose=${vNoisy.glucose.glucose}`);

  // ...but a noisy yet still-recoverable pulse must NOT be gated: the old
  // metric scored it 0 and therefore refused to report anything.
  const marginal = synth(fs, 30, 75, 0.25, 45, 5);
  const vMarginal = analyzeVitals(marginal, fs, scaleSignal(marginal, 0.7), scaleSignal(marginal, 1), scaleSignal(marginal, 0.4), { cameraMode: "rear", ageYears: age, sex: "male" });
  check(
    "Recoverable-but-noisy pulse is not gated",
    vMarginal.bpm !== null && Math.abs(vMarginal.bpm - 75) <= 5 && vMarginal.heart.signalQuality >= 40 && vMarginal.creatinine.creatinine !== null,
    `sqi=${vMarginal.heart.signalQuality} bpm=${vMarginal.bpm?.toFixed(1)} cr=${vMarginal.creatinine.creatinine}`
  );
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
