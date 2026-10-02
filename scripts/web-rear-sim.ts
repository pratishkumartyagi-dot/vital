import { analyzeVitals } from "../src/app/vital-monitoring/ppg-analysis";

const fs = 30;

// Simulate the web app rear-camera flow EXACTLY as page.tsx does it:
// isFingerFrame -> collect avgG during "measuring" -> every 2s run analyzeVitals(analysisSource=ppgWin, fs, redWin, greenWin, blueWin)
function simFrame() {
  // a frame: finger present -> red dominant, brightness in range
  return { avgR: 150, avgG: 120, avgB: 100 };
}
function isFingerFrame(avgR: number, avgG: number, avgB: number): boolean {
  return avgR > 90 && avgR < 245 && avgR > avgG + 15 && avgR > avgB + 15 && Math.abs(avgR - avgG) > 8;
}

const ppgBuffer: number[] = [];
const redBuf: number[] = [], greenBuf: number[] = [], blueBuf: number[] = [];
const samples: { sqi: number; resp: number | null }[] = [];

const finger = isFingerFrame(150, 120, 100);
console.log("isFingerFrame for finger frame:", finger);

// simulate 30s of 30fps rear-camera with a real finger (cardiac 1.2Hz + resp 0.25Hz baseline wander)
let s = 7;
const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
const nFrames = 30 * fs; // 30s
let lastUpdate = -3000;
for (let i = 0; i < nFrames; i++) {
  const t = i / fs;
  const cardiac = 8 * Math.sin(2 * Math.PI * 1.2 * t);
  const resp = 4 * Math.sin(2 * Math.PI * 0.25 * t);
  const avgG = 110 + cardiac + resp + rnd() * 1.5;
  const avgR = 155 + cardiac * 0.6 + resp * 0.4 + rnd() * 2;
  const avgB = 100 + cardiac * 0.4 + resp * 0.3 + rnd() * 2;

  const fi = isFingerFrame(avgR, avgG, avgB);
  const collecting = true && fi; // measuring phase assumed active
  if (collecting) {
    ppgBuffer.push(avgG); if (ppgBuffer.length > 1800) ppgBuffer.shift();
    redBuf.push(avgR); if (redBuf.length > 1800) redBuf.shift();
    greenBuf.push(avgG); if (greenBuf.length > 1800) greenBuf.shift();
    blueBuf.push(avgB); if (blueBuf.length > 1800) blueBuf.shift();
  }
  // every 2s
  if (i * 1000 / fs - lastUpdate > 2000) {
    lastUpdate = i * 1000 / fs;
    const windowLen = Math.floor(fs * 30);
    const ppgWin = ppgBuffer.slice(-Math.min(ppgBuffer.length, windowLen));
    const redWin = redBuf.slice(-Math.min(redBuf.length, windowLen));
    const greenWin = greenBuf.slice(-Math.min(greenBuf.length, windowLen));
    const blueWin = blueBuf.slice(-Math.min(blueBuf.length, windowLen));
    const analysisSource = ppgWin; // rear mode
    const a = analyzeVitals(analysisSource, fs, redWin, greenWin, blueWin, { cameraMode: "rear", ageYears: 40, sex: "male" });
    samples.push({ sqi: a.heart.signalQuality, resp: a.respRate });
    if (i % (5 * fs) === 0) console.log(`  t=${(i/fs).toFixed(0)}s SQI=${a.heart.signalQuality} resp=${a.respRate?.toFixed(1) ?? "null"} bpm=${a.bpm?.toFixed(1)}`);
  }
}
const last = samples[samples.length - 1];
console.log(`\nFinal: SQI=${last.sqi} Resp=${last.resp?.toFixed(1) ?? "null"}`);
console.log("Finger was detected throughout:", finger);
