function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.reduce((a, b) => a + (b - m) * (b - m), 0) / (arr.length - 1);
  return Math.sqrt(v);
}
interface Peak { index: number; value: number; }
const SOS = [
  [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
  [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
];
function applySosFilter(signal: number[], sos: number[][]): number[] {
  let stage = signal.slice();
  for (const section of sos) {
    const [b0, b1, b2, a0, a1, a2] = section;
    const out: number[] = new Array(stage.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < stage.length; i++) { const x0 = stage[i]; const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0; out[i] = y0; x2 = x1; x1 = x0; y2 = y1; y1 = y0; }
    stage = out;
  }
  return stage;
}
function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return signal.slice();
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = mean(x); const yMean = mean(signal);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - xMean) * (signal[i] - yMean); den += (x[i] - xMean) ** 2; }
  const slope = den === 0 ? 0 : num / den;
  const intercept = yMean - slope * xMean;
  return signal.map((y, i) => y - (slope * i + intercept));
}
function butterworthBandpass(signal: number[], low: number, high: number, fs: number): number[] {
  if (fs === 30 && low === 0.5 && high === 3.5) return applySosFilter(signal, SOS);
  return signal.slice();
}
function detectPeaks(signal: number[], fs: number, minBpm = 40, maxBpm = 180): { peaks: Peak[]; filtered: number[] } {
  const filtered = butterworthBandpass(detrend(signal), 0.5, 3.5, fs);
  const peaks: Peak[] = [];
  const minDist = Math.max(1, Math.floor((60 / maxBpm) * fs));
  const sorted = filtered.slice().sort((a, b) => a - b);
  const loIdx = Math.floor(sorted.length * 0.10);
  const hiIdx = Math.floor(sorted.length * 0.90);
  const baseline = sorted[loIdx];
  const peakCeil = sorted[hiIdx];
  const minHeight = baseline + 0.20 * (peakCeil - baseline);
  for (let i = 1; i < filtered.length - 1; i++) {
    if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1]) {
      if (filtered[i] > minHeight) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1].index >= minDist) peaks.push({ index: i, value: filtered[i] });
        else { if (filtered[i] > peaks[peaks.length - 1].value) peaks[peaks.length - 1] = { index: i, value: filtered[i] }; }
      }
    }
  }
  return { peaks, filtered };
}
function computePerfusionIndex(signal: number[]): number {
  if (signal.length < 10) return 0;
  const m = mean(signal);
  let ac = 0;
  for (const v of signal) ac += Math.abs(v - m);
  ac /= signal.length;
  return m > 0 ? (ac / m) * 100 : 0;
}
function powerSpectrum(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  const freqs: number[] = []; const psd: number[] = [];
  if (n < 8) return { freqs, psd };
  const detrended = detrend(signal);
  for (let k = 0; k < Math.floor(n / 2); k++) {
    let real = 0, imag = 0;
    for (let i = 0; i < n; i++) { const angle = (2 * Math.PI * k * i) / n; real += detrended[i] * Math.cos(angle); imag -= detrended[i] * Math.sin(angle); }
    freqs.push((k * fs) / n);
    psd.push((real * real + imag * imag) / n);
  }
  return { freqs, psd };
}
function computeSignalQuality(filtered: number[], raw: number[], peaks: Peak[], fs: number): number {
  if (peaks.length < 2) return 0;
  const pi = computePerfusionIndex(raw);
  const piScore = Math.min(1, pi / 2);
  const { freqs, psd } = powerSpectrum(filtered, fs);
  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.5 && freqs[i] <= 3.5) signalPower += psd[i];
    else if (freqs[i] > 3.5 && freqs[i] <= fs * 0.4) noisePower += psd[i];
  }
  const snr = noisePower > 0 ? signalPower / noisePower : 0;
  const snrScore = Math.min(1, snr / 4);
  const rrMs: number[] = [];
  for (let i = 1; i < peaks.length; i++) rrMs.push(((peaks[i].index - peaks[i - 1].index) / fs) * 1000);
  const cv = rrMs.length > 1 ? (stddev(rrMs) / mean(rrMs)) * 100 : 100;
  const regScore = Math.max(0, 1 - cv / 20);
  return Math.round((0.45 * piScore + 0.35 * snrScore + 0.2 * regScore) * 100);
}
const fs = 30;
function genRealistic(ac: number, dc: number, noise: number, secs: number) {
  const n = Math.floor(secs * fs);
  const out: number[] = [];
  const heartHz = 72 / 60;
  const respHz = 0.25;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const beat = Math.sin(2 * Math.PI * heartHz * t);
    const dicrotic = 0.35 * Math.sin(2 * Math.PI * heartHz * t - 0.6);
    const pulse = Math.max(0, beat + dicrotic);
    const baseline = 0.5 + 0.2 * Math.sin(2 * Math.PI * respHz * t);
    const nz = (Math.random() - 0.5) * noise;
    out.push(dc + ac * pulse * baseline + nz);
  }
  return out;
}
console.log("=== CURRENT (pi/2 threshold) ===");
for (const ac of [5, 8, 10, 12, 15]) {
  const src = genRealistic(ac, 128, 0.3, 30);
  const { peaks, filtered } = detectPeaks(src, fs);
  const valid = peaks.filter((_, i) => { if (i === 0) return true; const dtMs = ((peaks[i].index - peaks[i - 1].index) / fs) * 1000; return dtMs >= 300 && dtMs <= 2000; });
  if (valid.length < 2) { console.log("AC", ac, "peaks", valid.length, "SKIP"); continue; }
  const pi = computePerfusionIndex(src);
  console.log("AC amp", ac, "PI:", pi.toFixed(3), "SQ:", computeSignalQuality(filtered, src, valid, fs));
}
