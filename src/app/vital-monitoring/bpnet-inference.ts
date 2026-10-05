import * as tf from '@tensorflow/tfjs';

export interface BPNetConfig {
  inputLength: number;
  cnnFilters: number[];
  kernelSize: number;
  poolSize: number;
  lstmHidden: number;
  lstmLayers: number;
  dropout: number;
}

export interface BPResult {
  sbp: number | null;
  dbp: number | null;
  map: number | null;
  pulsePressure: number | null;
  confidence: number;
}

const DEFAULT_CONFIG: BPNetConfig = {
  inputLength: 1024,
  cnnFilters: [32, 64, 128],
  kernelSize: 3,
  poolSize: 2,
  lstmHidden: 128,
  lstmLayers: 2,
  dropout: 0.2
};

class BPNetInference {
  private model: tf.LayersModel | null = null;
  private config: BPNetConfig;
  private isLoaded = false;

  constructor(config: Partial<BPNetConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async load(modelUrl: string): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(modelUrl);
      this.isLoaded = true;
      console.log('BPNet model loaded successfully');
    } catch (error) {
      console.error('Failed to load BPNet model:', error);
      throw error;
    }
  }

  async loadFromIndexedDB(modelName: string = 'bpnet-model'): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`indexeddb://${modelName}`);
      this.isLoaded = true;
      console.log('BPNet model loaded from IndexedDB');
    } catch (error) {
      console.warn('No cached BPNet model found, will use fallback:', error);
      throw error;
    }
  }

  async saveToIndexedDB(modelName: string = 'bpnet-model'): Promise<void> {
    if (!this.model) throw new Error('No model to save');
    await this.model.save(`indexeddb://${modelName}`);
    console.log('BPNet model saved to IndexedDB');
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

    // tf.tensor3d expects [batch, seq, channels] - create proper shape
    const tensorData = new Float32Array(normalized);
    return tf.tensor(tensorData, [1, targetLen, 1]);
  }

  async predict(ppg: number[]): Promise<BPResult> {
    if (!this.isLoaded || !this.model) {
      return this.fallbackEstimate(ppg);
    }

    const input = this.preprocessPPG(ppg);
    try {
      const outputs = this.model.predict(input) as tf.Tensor[];
      const sbpTensor = outputs[0];
      const dbpTensor = outputs[1];

      const sbp = (await sbpTensor.data())[0];
      const dbp = (await dbpTensor.data())[0];

      input.dispose();
      sbpTensor.dispose();
      dbpTensor.dispose();

      const pulsePressure = sbp - dbp;
      const map = dbp + pulsePressure / 3;

      return {
        sbp: Math.round(Math.max(80, Math.min(200, sbp))),
        dbp: Math.round(Math.max(50, Math.min(130, dbp))),
        map: Math.round(Math.max(60, Math.min(150, map))),
        pulsePressure: Math.round(Math.max(20, Math.min(100, pulsePressure))),
        confidence: 0.85
      };
    } catch (error) {
      console.error('BPNet inference failed, using fallback:', error);
      input.dispose();
      return this.fallbackEstimate(ppg);
    }
  }

  private fallbackEstimate(ppg: number[]): BPResult {
    const fs = 30;
    if (ppg.length < fs * 5) {
      return { sbp: null, dbp: null, map: null, pulsePressure: null, confidence: 0 };
    }

    const mean = ppg.reduce((a, b) => a + b, 0) / ppg.length;
    const std = Math.sqrt(ppg.reduce((a, b) => a + (b - mean) ** 2, 0) / ppg.length);
    const cv = std / Math.abs(mean);

    const baseSystolic = 120;
    const baseDiastolic = 80;
    const adjustment = cv * 20;

    return {
      sbp: Math.round(baseSystolic + adjustment),
      dbp: Math.round(baseDiastolic + adjustment * 0.5),
      map: Math.round(baseDiastolic + (baseSystolic - baseDiastolic) / 3),
      pulsePressure: Math.round(baseSystolic - baseDiastolic),
      confidence: 0.3
    };
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

let bpnetInstance: BPNetInference | null = null;

export function getBPNetInstance(config?: Partial<BPNetConfig>): BPNetInference {
  if (!bpnetInstance) {
    bpnetInstance = new BPNetInference(config);
  }
  return bpnetInstance;
}

export async function estimateBPDeep(ppg: number[], fs: number = 30): Promise<BPResult> {
  const instance = getBPNetInstance();
  return instance.predict(ppg);
}

export function resetBPNetInstance(): void {
  if (bpnetInstance) {
    bpnetInstance.dispose();
    bpnetInstance = null;
  }
}