export interface RRMethodResult {
  respRate: number | null;
  confidence: number;
  method: string;
}

export interface RRResult extends RRMethodResult {
  waveform: number[];
}

function detrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 3) return signal.slice();
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const yMean = signal.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - xMean) * (signal[i] - yMean);
    den += (x[i] - xMean) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = yMean - slope * xMean;
  return signal.map((y, i) => y - (slope * i + intercept));
}

function butterworthBandpass(signal: number[], low: number, high: number, fs: number): number[] {
  if (signal.length < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
    return signal.slice();
  }
  return applyButterworthCascade(signal, low, high, fs, true);
}

function applyButterworthCascade(signal: number[], low: number, high: number, fs: number, zeroPhase: boolean): number[] {
  const hp = highpassBiquad(low, fs);
  const lp = lowpassBiquad(high, fs);
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;

  let y = applyBiquad(signal, hp, dc, 0);
  y = applyBiquad(y, lp, 0, 0);

  if (zeroPhase) {
    y = y.reverse();
    y = applyBiquad(y, hp, 0, 0);
    y = applyBiquad(y, lp, 0, 0);
    y = y.reverse();
  }
  return y;
}

function lowpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cw) / 2 / a0,
    b1: (1 - cw) / a0,
    b2: (1 - cw) / 2 / a0,
    a1: (-2 * cw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function highpassBiquad(fc: number, fs: number, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cw = Math.cos(w0);
  const sw = Math.sin(w0);
  const alpha = sw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cw) / 2 / a0,
    b1: -(1 + cw) / a0,
    b2: (1 + cw) / 2 / a0,
    a1: (-2 * cw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function applyBiquad(signal: number[], c: { b0: number; b1: number; b2: number; a1: number; a2: number }, seedX: number, seedY: number): number[] {
  const out: number[] = new Array(signal.length);
  let x1 = seedX, x2 = seedX, y1 = seedY, y2 = seedY;
  for (let i = 0; i < signal.length; i++) {
    const y = c.b0 * signal[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y;
    x2 = x1; x1 = signal[i];
    y2 = y1; y1 = y;
  }
  return out;
}

function firFilter(signal: number[], cutoff: number, fs: number, type: 'low' | 'high', order = 100): number[] {
  if (signal.length < order) return signal.slice();
  
  const nyq = fs / 2;
  const normCutoff = cutoff / nyq;
  const halfOrder = Math.floor(order / 2);
  const taps: number[] = new Array(order + 1);
  
  if (type === 'low') {
    for (let i = 0; i <= order; i++) {
      const n = i - halfOrder;
      if (n === 0) {
        taps[i] = 2 * normCutoff;
      } else {
        taps[i] = Math.sin(2 * Math.PI * normCutoff * n) / (Math.PI * n);
      }
    }
  } else {
    for (let i = 0; i <= order; i++) {
      const n = i - halfOrder;
      if (n === 0) {
        taps[i] = 0;
      } else {
        taps[i] = (n % 2 === 0 ? 1 : -1) * Math.sin(2 * Math.PI * normCutoff * n) / (Math.PI * n);
      }
    }
  }
  
  const window: number[] = new Array(order + 1);
  for (let i = 0; i <= order; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / order);
  }
  for (let i = 0; i <= order; i++) {
    taps[i] *= window[i];
  }
  
  const sum = taps.reduce((a, b) => a + b, 0);
  for (let i = 0; i <= order; i++) taps[i] /= sum;
  
  const out: number[] = new Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    let acc = 0;
    for (let j = 0; j <= order; j++) {
      const idx = i - j;
      if (idx >= 0) acc += taps[j] * signal[idx];
    }
    out[i] = acc;
  }
  
  for (let i = 0; i < halfOrder; i++) {
    out[i] = 0;
  }
  
  return out;
}

function autocorrelate(signal: number[]): { values: number[]; lags: number[] } {
  const n = signal.length;
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  const centered = signal.map(v => v - mean);
  const c0 = centered.reduce((a, b) => a + b * b, 0);
  
  const maxLag = Math.min(n - 1, n);
  const values: number[] = [];
  const lags: number[] = [];
  
  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) {
      sum += centered[i] * centered[i + lag];
    }
    values.push(sum / c0);
    lags.push(lag);
  }
  
  return { values, lags };
}

function detectPeaks3Pt(signal: number[], minDist: number): number[] {
  const peaks: number[] = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

function arburg(signal: number[], order: number): { a: number[]; v: number } {
  const n = signal.length;
  const a: number[] = [1.0];
  let refSignal = signal.slice();
  let v = refSignal.reduce((s, x) => s + x * x, 0) / n;
  
  for (let p = 1; p <= order; p++) {
    let num = 0;
    let den = 0;
    for (let i = p; i < n; i++) {
      num += refSignal[i] * refSignal[i - p];
      den += refSignal[i - p] * refSignal[i - p] + refSignal[i] * refSignal[i];
    }
    const k = -2 * num / (den || 1);
    
    for (let i = 1; i < p; i++) {
      a[i] = a[i] + k * a[p - i];
    }
    a[p] = k;
    
    v = v * (1 - k * k);
  }
  
  return { a, v };
}

function rootsPolynomial(coeffs: number[]): { real: number; imag: number }[] {
  const n = coeffs.length;
  if (n <= 1) return [];
  
  const roots: { real: number; imag: number }[] = [];
  const a = coeffs.slice();
  let count = n;
  
  for (let iter = 0; iter < n - 1; iter++) {
    if (count <= 1) break;
    
    let x = 0, y = 0;
    for (let i = 1; i < count; i++) {
      x += a[i];
    }
    
    let maxIter = 100;
    let converged = false;
    
    for (let it = 0; it < maxIter; it++) {
      let zr = a[count - 1], zi = 0;
      let dr = a[count - 1], di = 0;
      
      for (let i = count - 2; i >= 0; i--) {
        const tr = zr * x - zi * y + a[i];
        zi = zr * y + zi * x;
        zr = tr;
        
        const tri = dr * x - di * y;
        di = dr * y + di * x + a[i];
        dr = tri;
      }
      
      const denom = dr * dr + di * di;
      if (denom < 1e-20) break;
      
      const dx = (zr * dr + zi * di) / denom;
      const dy = (zi * dr - zr * di) / denom;
      
      x -= dx;
      y -= dy;
      
      if (Math.abs(dx) < 1e-10 && Math.abs(dy) < 1e-10) {
        converged = true;
        break;
      }
    }
    
    if (converged) {
      roots.push({ real: x, imag: y });
      
      const b: number[] = new Array(count - 1);
      b[0] = a[0];
      for (let i = 1; i < count - 1; i++) {
        b[i] = a[i] + x * b[i - 1] / (b[0] !== 0 ? b[0] : 1);
      }
      for (let i = 0; i < count - 1; i++) {
        a[i] = b[i];
      }
      count--;
    }
  }
  
  if (count > 0 && a.length > 1 && a[1] !== 0) {
    const root = -a[0] / a[1];
    roots.push({ real: root, imag: 0 });
  }
  
  return roots;
}

function estimateRespRatePPGSync(
  ppg: number[],
  fs: number,
  options: {
    rrRange?: [number, number];
    downsampleFreq?: number;
    arModelOrder?: number;
  } = {}
): RRResult {
  const rrRange = options.rrRange ?? [8, 35];
  const downsampleFreq = options.downsampleFreq ?? 10;
  const arModelOrder = options.arModelOrder ?? 16;
  
  if (ppg.length < fs * 10) {
    return { respRate: null, confidence: 0, method: 'insufficient-data', waveform: [] };
  }

  const detrended = detrend(ppg);
  const filtered = butterworthBandpass(detrended, 0.05, 0.7, fs);

  const downsampleRatio = fs / downsampleFreq;
  const downsampled: number[] = [];
  for (let i = 0; i < filtered.length; i += Math.floor(downsampleRatio)) {
    downsampled.push(filtered[i]);
  }
  
  const detrendedDown = detrend(downsampled);

  const acfResult = estimateByACF(detrendedDown, downsampleFreq, rrRange, fs);
  const pksResult = estimateByPKS(filtered, downsampleFreq, rrRange, fs);
  const arpResult = estimateByARP(detrendedDown, downsampleFreq, arModelOrder, rrRange);
  
  const methods = [acfResult, pksResult, arpResult].filter(m => m.respRate !== null);
  
  if (methods.length === 0) {
    return { respRate: null, confidence: 0, method: 'all-methods-failed', waveform: filtered };
  }

  let fusedRate = 0;
  let totalWeight = 0;
  let methodNames: string[] = [];
  
  for (const m of methods) {
    const weight = m.confidence;
    fusedRate += m.respRate! * weight;
    totalWeight += weight;
    methodNames.push(`${m.method}:${m.respRate!.toFixed(1)}`);
  }
  
  if (totalWeight > 0) {
    fusedRate /= totalWeight;
  } else {
    fusedRate = methods[0].respRate!;
  }
  
  const confidence = totalWeight / methods.length;

  if (fusedRate < rrRange[0] || fusedRate > rrRange[1]) {
    return { respRate: null, confidence: 0, method: 'out-of-range', waveform: filtered };
  }

  return {
    respRate: Math.round(fusedRate * 10) / 10,
    confidence: Math.round(confidence * 100) / 100,
    method: `RR-${methodNames.join('+')}`,
    waveform: filtered
  };
}

export function estimateRespRatePPG(
  ppg: number[],
  fs: number,
  options: {
    rrRange?: [number, number];
    downsampleFreq?: number;
    arModelOrder?: number;
  } = {}
): Promise<RRResult> {
  return Promise.resolve(estimateRespRatePPGSync(ppg, fs, options));
}

export { estimateRespRatePPGSync as estimateRespRateRearPPG };

function estimateByACF(signal: number[], fs: number, rrRange: [number, number], originalFs: number): RRMethodResult {
  const acf = autocorrelate(signal);
  
  const minLag = Math.floor(fs / (rrRange[1] / 60));
  const maxLag = Math.floor(fs / (rrRange[0] / 60));
  
  let bestLag = -1;
  let bestVal = -Infinity;
  
  for (let lag = minLag; lag <= Math.min(maxLag, acf.values.length - 1); lag++) {
    if (acf.values[lag] > bestVal) {
      bestVal = acf.values[lag];
      bestLag = lag;
    }
  }
  
  if (bestLag <= 0) {
    return { respRate: null, confidence: 0, method: 'ACF' };
  }
  
  const rate = 60 / (bestLag / fs);
  const confidence = Math.max(0, Math.min(1, bestVal));
  
  return {
    respRate: Math.round(rate * 10) / 10,
    confidence,
    method: 'ACF'
  };
}

function estimateByPKS(signal: number[], fs: number, rrRange: [number, number], originalFs: number): RRMethodResult {
  const minDist = Math.floor(fs / (rrRange[1] / 60));
  const peaks = detectPeaks3Pt(signal, minDist);
  
  if (peaks.length < 2) {
    return { respRate: null, confidence: 0, method: 'PKS' };
  }
  
  const duration = (peaks[peaks.length - 1] - peaks[0]) / fs;
  if (duration <= 0) {
    return { respRate: null, confidence: 0, method: 'PKS' };
  }
  
  const rate = 60 * (peaks.length - 1) / duration;
  const confidence = Math.min(1, peaks.length / 5);
  
  return {
    respRate: Math.round(rate * 10) / 10,
    confidence,
    method: 'PKS'
  };
}

function estimateByARP(signal: number[], fs: number, modelOrder: number, rrRange: [number, number]): RRMethodResult {
  if (signal.length < modelOrder + 1) {
    return { respRate: null, confidence: 0, method: 'ARP' };
  }
  
  const { a } = arburg(signal, modelOrder);
  
  if (a.length <= 1 || a.every(x => x === 0)) {
    return { respRate: null, confidence: 0, method: 'ARP' };
  }
  
  const coeffForRoots = a.slice(1).reverse();
  coeffForRoots.push(a[0]);
  
  const poles = rootsPolynomial(a);
  
  let maxMag = 0;
  let bestAngle = 0;
  
  for (const pole of poles) {
    const angle = Math.atan2(pole.imag, pole.real);
    const freq = (angle * fs) / (2 * Math.PI);
    const bpm = freq * 60;
    
    if (bpm >= rrRange[0] && bpm <= rrRange[1]) {
      const mag = Math.sqrt(pole.real * pole.real + pole.imag * pole.imag);
      if (mag > maxMag) {
        maxMag = mag;
        bestAngle = bpm;
      }
    }
  }
  
  if (maxMag === 0) {
    return { respRate: null, confidence: 0, method: 'ARP' };
  }
  
  return {
    respRate: Math.round(bestAngle * 10) / 10,
    confidence: Math.min(1, maxMag * 2),
    method: 'ARP'
  };
}