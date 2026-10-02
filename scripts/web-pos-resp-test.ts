import { posPulse, analyzeRespRate, analyzeVitals, butterworthBandpass } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;
// Build a realistic face/rPPG RGB signal: DC + cardiac + respiratory baseline wander
function faceRGB(seconds: number, hrBpm: number, respBpm: number, seed = 31): {r:number[];g:number[];b:number[]} {
  const n = Math.floor(seconds * fs);
  const r: number[] = []; const g: number[] = []; const b: number[] = [];
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const pulse = Math.sin(2 * Math.PI * (hrBpm/60) * t) + 0.3 * Math.sin(2 * Math.PI * 2 * (hrBpm/60) * t + 0.7);
    const motion = 1 + 0.35 * Math.sin(2 * Math.PI * 1.0 * t) + 0.002 * rnd();
    const resp = 0.12 * Math.sin(2 * Math.PI * (respBpm/60) * t);
    r.push(135 * (1 + 0.06 * pulse + resp) * motion);
    g.push(120 * (1 + 0.18 * pulse + resp) * motion);
    b.push(110 * (1 + 0.10 * pulse + resp) * motion);
  }
  return { r, g, b };
}

console.log("=== Front camera (POS) respiratory rate ===");
for (const { hr, resp } of [{ hr: 72, resp: 15 }, { hr: 90, resp: 18 }]) {
  const rgb = faceRGB(30, hr, resp);
  const pos = posPulse(rgb.r, rgb.g, rgb.b, 32);
  const r = analyzeRespRate(pos, fs);
  console.log(`  HR=${hr} Resp=${resp} -> POS resp=${r.respRate?.toFixed(1)} method=${r.method} conf=${r.confidence.toFixed(2)}`);
  // Also: analyzeVitals path (front mode)
  const v = analyzeVitals(pos, fs, rgb.r, rgb.g, rgb.b, { cameraMode: "front" });
  console.log(`    analyzeVitals: bpm=${v.bpm?.toFixed(1)} resp=${v.respRate?.toFixed(1)} sqi=${v.heart.signalQuality}`);
}

console.log("\n=== Rear camera (green) respiratory rate (already verified, re-check) ===");
// reuse: just use green channel directly
const rgb2 = faceRGB(30, 72, 15);
const v = analyzeVitals(rgb2.g, fs, rgb2.r, rgb2.g, rgb2.b, { cameraMode: "rear" });
console.log(`  rear: bpm=${v.bpm?.toFixed(1)} resp=${v.respRate?.toFixed(1)} sqi=${v.heart.signalQuality}`);
