import * as tf from '@tensorflow/tfjs';

export interface CorrencoderConfig {
  inputLength: number;
  nKernels: number[];
  kernelSizes: number[];
  dropout: number;
}

export interface RespWaveformResult {
  waveform: number[];
  respRate: number;
  confidence: number;
  method: string;
}

const DEFAULT_CONFIG: CorrencoderConfig = {
  inputLength: 1024,
  nKernels: [8, 8, 8],
  kernelSizes: [150, 75, 50],
  dropout: 0.5
};

class CorrencoderInference {
  private model: tf.LayersModel | null = null;
  private config: CorrencoderConfig;
  private isLoaded = false;

  constructor(config: Partial<CorrencoderConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async load(modelUrl: string): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(modelUrl);
      this.isLoaded = true;
      console.log('Correncoder model loaded successfully');
    } catch (error) {
      console.error('Failed to load Correncoder model:', error);
      throw error;
    }
  }

  async loadFromIndexedDB(modelName: string = 'correncoder-model'): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`indexeddb://${modelName}`);
      this.isLoaded = true;
      console.log('Correncoder model loaded from IndexedDB');
    } catch (error) {
      console.warn('No cached Correncoder model found:', error);
      throw error;
    }
  }

  async saveToIndexedDB(modelName: string = 'correncoder-model'): Promise<void> {
    if (!this.model) throw new Error('No model to save');
    await this.model.save(`indexeddb://${modelName}`);
    console.log('Correncoder model saved to IndexedDB');
  }

  preprocessPPG(ppg: number[]): tf.Tensor {
    const targetLen = this.config.inputLength;
    let signal = new Float32Array(ppg);

    if (signal.length > targetLen) {
      signal = signal.slice(0, targetLen);
    } else if (signal.length < targetLen) {
      const padded = new Float32Array(targetLen);
      padded.set(signal);
      signal = padded;
    }

    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const std = Math.sqrt(signal.reduce((a, b) => a + (b - mean) ** 2, 0) / signal.length);
    const normalized = Array.from(signal).map(v => (v - mean) / (std + 1e-8));

    const tensorData = new Float32Array(normalized);
    return tf.tensor(tensorData, [1, targetLen, 1]);
  }

  estimateRespRateFromWaveform(waveform: number[], fs: number): { rate: number; confidence: number } {
    if (waveform.length < fs * 2) return { rate: 0, confidence: 0 };

    let peaks = 0;
    for (let i = 1; i < waveform.length - 1; i++) {
      if (waveform[i] > waveform[i - 1] && waveform[i] > waveform[i + 1]) {
        if (waveform[i] > 0.5) peaks++;
      }
    }

    const durationSec = waveform.length / fs;
    const rate = (peaks / durationSec) * 60;
    const confidence = Math.min(1, peaks / (durationSec * 0.3));

    if (rate < 5 || rate > 50) return { rate: 0, confidence: 0 };
    return { rate, confidence };
  }

  async predict(ppg: number[], fs: number): Promise<RespWaveformResult> {
    if (!this.isLoaded || !this.model) {
      return this.fallbackEstimate(ppg, fs);
    }

    const input = this.preprocessPPG(ppg);
    try {
      const output = this.model.predict(input) as tf.Tensor;
      const waveformData = await output.data();
      const waveform = Array.from(waveformData);

      input.dispose();
      output.dispose();

      const { rate, confidence } = this.estimateRespRateFromWaveform(waveform, fs);

      return {
        waveform,
        respRate: Math.round(rate * 10) / 10,
        confidence,
        method: 'correncoder-dl'
      };
    } catch (error) {
      console.error('Correncoder inference failed, using fallback:', error);
      input.dispose();
      return this.fallbackEstimate(ppg, fs);
    }
  }

  private fallbackEstimate(ppg: number[], fs: number): RespWaveformResult {
    const detrended = this.detrend(ppg);
    const lowpass = this.butterworthLowpass(detrended, 0.5, fs);
    const ac = this.autocorrelate(lowpass);
    const minLag = Math.floor(fs / 0.5);
    const maxLag = Math.floor(fs / 0.1);
    let bestLag = -1;
    let bestVal = -Infinity;
    for (let lag = minLag; lag <= Math.min(maxLag, ac.length - 1); lag++) {
      if (ac[lag] > bestVal) {
        bestVal = ac[lag];
        bestLag = lag;
      }
    }
    let rate = 0;
    if (bestLag > 0) rate = (60 * fs) / bestLag;
    const confidence = Math.max(0, Math.min(1, bestVal));
    if (rate < 5 || rate > 50) return { waveform: [], respRate: 0, confidence: 0, method: 'fallback-autocorr' };
    return { waveform: [], respRate: Math.round(rate * 10) / 10, confidence, method: 'fallback-autocorr' };
  }

  private detrend(signal: number[]): number[] {
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

  private butterworthLowpass(signal: number[], fc: number, fs: number): number[] {
    const w0 = (2 * Math.PI * fc) / fs;
    const cw = Math.cos(w0);
    const sw = Math.sin(w0);
    const alpha = sw / (2 * Math.SQRT1_2);
    const a0 = 1 + alpha;
    const b0 = (1 - cw) / 2 / a0;
    const b1 = (1 - cw) / a0;
    const b2 = (1 - cw) / 2 / a0;
    const a1 = (-2 * cw) / a0;
    const a2 = (1 - alpha) / a0;

    const out: number[] = new Array(signal.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < signal.length; i++) {
      const y = b0 * signal[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      out[i] = y;
      x2 = x1; x1 = signal[i];
      y2 = y1; y1 = y;
    }
    return out;
  }

  private autocorrelate(signal: number[]): number[] {
    const n = signal.length;
    const m = signal.reduce((a, b) => a + b, 0) / n;
    const centered = signal.map(v => v - m);
    const norm = centered.reduce((a, b) => a + b * b, 0) || 1;
    const out: number[] = [];
    for (let lag = 0; lag < n; lag++) {
      let s = 0;
      for (let i = 0; i < n - lag; i++) s += centered[i] * centered[i + lag];
      out.push(s / norm);
    }
    return out;
  }

  isReady(): boolean {
    return this.isLoaded;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isLoaded = false;
    }
  }
}

let correncoderInstance: CorrencoderInference | null = null;

export function getCorrencoderInstance(config?: Partial<CorrencoderConfig>): CorrencoderInference {
  if (!correncoderInstance) {
    correncoderInstance = new CorrencoderInference(config);
  }
  return correncoderInstance;
}

export async function estimateRespRateCorrencoder(ppg: number[], fs: number = 125): Promise<RespWaveformResult> {
  const instance = getCorrencoderInstance();
  return instance.predict(ppg, fs);
}

export function resetCorrencoderInstance(): void {
  if (correncoderInstance) {
    correncoderInstance.dispose();
    correncoderInstance = null;
  }
}