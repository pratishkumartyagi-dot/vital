import {
  analyzeHeartRate,
  analyzeRespRate,
  analyzeVitals,
  butterworthBandpass,
  powerSpectrum,
} from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;

// Realistic finger PPG: DC + cardiac + RESPIRATORY BASELINE WANDER (additive DC shift)
function realisticPPG(seconds: number, hrBpm: number, respBpm: number, seed = 7): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const cardiac = 10 * Math.sin(2 * Math.PI * (hrBpm / 60) * t)
      + 3 * Math.sin(2 * Math.PI * 2 * (hrBpm / 60) * t + 0.4);
    const resp = 4 * Math.sin(2 * Math.PI * (respBpm / 60) * t); // baseline wander as DC shift
    out.push(110 + cardiac + resp + rnd() * 2);
  }
  return out;
}

const tests = [
  { hr: 72, resp: 15 },
  { hr: 60, resp: 12 },
  { hr: 90, resp: 18 },
  { hr: 110, resp: 20 },
];

for (const { hr, resp } of tests) {
  const ppg = realisticPPG(30, hr, resp);
  const h = analyzeHeartRate(ppg, fs);
  const r = analyzeRespRate(ppg, fs);
  const hrErr = h.bpm === null ? Infinity : Math.abs(h.bpm - hr);
  const respErr = r.respRate === null ? Infinity : Math.abs(r.respRate - resp);
  console.log(`HR=${hr} (${h.bpm?.toFixed(1)}) err=${hrErr.toFixed(2)} | SQI=${h.signalQuality} | Resp=${resp} (${r.respRate?.toFixed(1)}) err=${respErr.toFixed(2)} method=${r.method} conf=${r.confidence.toFixed(2)}`);
}

// Also test analyzeVitals path (what page.tsx uses) with green-only
console.log("\n--- analyzeVitals (web app path) ---");
const ppg = realisticPPG(30, 72, 15);
const v = analyzeVitals(ppg, fs);
console.log(`bpm=${v.bpm?.toFixed(1)} resp=${v.respRate?.toFixed(1)} sqi=${v.heart.signalQuality}`);

// Test the bandpass alone on respiratory extraction
console.log("\n--- bandpass 0.1-0.5 Hz on baseline wander ---");
const detrended = ppg.map((v, i) => { const m = ppg.reduce((a,b)=>a+b,0)/ppg.length; return v - m; });
const bp = butterworthBandpass(detrended, 0.1, 0.5, fs, true);
const { freqs, psd } = powerSpectrum(bp, fs);
let bestIdx = 0;
for (let i = 1; i < psd.length; i++) if (psd[i] > psd[bestIdx]) bestIdx = i;
console.log(`dominant freq in 0.1-0.5 band: ${freqs[bestIdx].toFixed(3)} Hz = ${freqs[bestIdx]*60} bpm`);
