import { posPulse, analyzeRespRate, butterworthBandpass } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;
// Face/rPPG RGB signal with motion artifact + respiratory baseline wander
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

console.log("Resp rate source comparison (true front-mode, face with motion):");
for (const { hr, resp } of [{ hr: 72, resp: 15 }, { hr: 90, resp: 18 }, { hr: 60, resp: 12 }]) {
  const { r, g, b } = faceRGB(30, hr, resp);
  const pos = posPulse(r, g, b, 32);

  // Current web app path: resp from POS pulse
  const rFromPos = analyzeRespRate(pos, fs);

  // Two-stage idea: extract baseline wander from RAW green (Stage1 bandpass cardiac away, Stage2 lowpass resp)
  const greenRaw = g;
  const rFromGreen = analyzeRespRate(greenRaw, fs);

  // Two-stage explicitly: Stage 2 = lowpass 0.06-0.7 Hz on detrended green
  const detrended = greenRaw.map((v) => v - greenRaw.reduce((a, b) => a + b, 0) / greenRaw.length);
  const bw = butterworthBandpass(detrended, 0.06, 0.7, fs, true);
  const rFromBW = analyzeRespRate(bw, fs);

  console.log(`  HR=${hr} trueResp=${resp}: POS=${rFromPos.respRate?.toFixed(1)} | greenRaw=${rFromGreen.respRate?.toFixed(1)} | BW(0.06-0.7)=${rFromBW.respRate?.toFixed(1)}`);
}
