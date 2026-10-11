/**
 * ROSE-Net (Remote Oxygen Saturation Estimation Network) — TensorFlow.js inference module
 *
 * Implements non-contact SpO2 estimation via rPPG from facial video streams
 * using the deep learning architecture from:
 *   Chowdhury, M.H., Reaz, M.B.I., Ali, S.H.Md., Khan, M.S., Chowdhury, M.E.H.
 *   "ROSE-Net: Leveraging remote photoplethysmography to estimate oxygen
 *   saturation using deep learning." Biomedical Signal Processing and Control, 100, 2025.
 *
 * The model takes a Spatial-Temporal Map (STMap) as input — a 2D representation
 * of normalised RGB colour intensities over time — and outputs an SpO2 estimate.
 *
 * Cross-domain training approach:
 *   1. Pre-train on BIDMC contact-PPG-derived STMaps (source domain)
 *   2. Fine-tune on camera-based rPPG video STMaps (target domain)
 */
import * as tf from "@tensorflow/tfjs";

export interface ROSENetConfig {
  inputLength: number;
  inChannels: number;
  baseChannels: number;
  numBlocks: number;
  lstmHidden: number;
  dropout: number;
  useAttention: boolean;
}

const DEFAULT_CONFIG: ROSENetConfig = {
  inputLength: 256,
  inChannels: 6, // RGB + YUV
  baseChannels: 32,
  numBlocks: 4,
  lstmHidden: 64,
  dropout: 0.3,
  useAttention: true,
};

export type SpO2Method = "rosenet" | "empirical";

export interface SpO2Result {
  spo2: number | null;
  rRatio: number | null;
  quality: number;
  method: SpO2Method;
  confidence: number;
  modelUsed: boolean;
}

/**
 * Generate a spatial-temporal map (STMap) from per-frame RGB means.
 *
 * The STMap is a 2-D representation where:
 *   - One axis is time (frame index)
 *   - The other axis is colour channels (R, G, B, or extended with YUV/YCrCb)
 *
 * Each value is the mean-normalised channel intensity (Cn = C / mean(C)).
 */
export function generateSTMap(
  rgbSignals: number[][],
  colorSpaces: ("rgb" | "yuv" | "ycrcb")[] = ["rgb", "yuv"],
  targetLength: number = 256,
): number[][] {
  const n = rgbSignals.length;
  if (n === 0) return [];

  // Convert to float64 arrays per channel
  const r = new Float64Array(n);
  const g = new Float64Array(n);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = rgbSignals[i][0] ?? 0;
    g[i] = rgbSignals[i][1] ?? 0;
    b[i] = rgbSignals[i][2] ?? 0;
  }

  // Temporal mean normalisation: Cn = C / mean(C)
  const meanR = r.reduce((a, b) => a + b, 0) / n;
  const meanG = g.reduce((a, b) => a + b, 0) / n;
  const meanB = b.reduce((a, b) => a + b, 0) / n;

  if (meanR <= 1e-8 || meanG <= 1e-8 || meanB <= 1e-8) {
    return [];
  }

  // Build normalised signals for each colour space
  const channelArrays: Float64Array[] = [];

  for (const cs of colorSpaces) {
    if (cs === "rgb") {
      const rN = new Float64Array(n);
      const gN = new Float64Array(n);
      const bN = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        rN[i] = r[i] / meanR;
        gN[i] = g[i] / meanG;
        bN[i] = b[i] / meanB;
      }
      channelArrays.push(rN, gN, bN);
    } else if (cs === "yuv") {
      // YUV conversion from normalised RGB
      const rN = new Float64Array(n);
      const gN = new Float64Array(n);
      const bN = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        rN[i] = r[i] / meanR;
        gN[i] = g[i] / meanG;
        bN[i] = b[i] / meanB;
      }
      const y = new Float64Array(n);
      const u = new Float64Array(n);
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const ri = rN[i], gi = gN[i], bi = bN[i];
        y[i] = 0.299 * ri + 0.587 * gi + 0.114 * bi;
        u[i] = 0.492 * (bi - y[i]);
        v[i] = 0.877 * (ri - y[i]);
      }
      // Normalise YUV
      const meanY = y.reduce((a, b) => a + b, 0) / n;
      const meanU = u.reduce((a, b) => a + b, 0) / n;
      const meanV = v.reduce((a, b) => a + b, 0) / n;
      if (meanY > 1e-8) {
        for (let i = 0; i < n; i++) y[i] /= meanY;
      }
      if (Math.abs(meanU) > 1e-8) {
        for (let i = 0; i < n; i++) u[i] /= meanU;
      }
      if (Math.abs(meanV) > 1e-8) {
        for (let i = 0; i < n; i++) v[i] /= meanV;
      }
      channelArrays.push(y, u, v);
    } else if (cs === "ycrcb") {
      const rN = new Float64Array(n);
      const gN = new Float64Array(n);
      const bN = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        rN[i] = r[i] / meanR;
        gN[i] = g[i] / meanG;
        bN[i] = b[i] / meanB;
      }
      const y = new Float64Array(n);
      const cr = new Float64Array(n);
      const cb = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const ri = rN[i], gi = gN[i], bi = bN[i];
        y[i] = 0.299 * ri + 0.587 * gi + 0.114 * bi;
        cr[i] = 0.713 * (ri - y[i]) + 0.5;
        cb[i] = 0.713 * (bi - y[i]) + 0.5;
      }
      const meanY = y.reduce((a, b) => a + b, 0) / n;
      if (meanY > 1e-8) {
        for (let i = 0; i < n; i++) y[i] /= meanY;
      }
      channelArrays.push(y, cr, cb);
    }
  }

  // Resample each channel to targetLength
  const nChannels = channelArrays.length;
  const stmap = new Array(targetLength);
  const xOld = n > 1 ? Array.from({ length: n }, (_, i) => i / (n - 1)) : [0, 1];
  const xNew = Array.from({ length: targetLength }, (_, i) =>
    targetLength > 1 ? i / (targetLength - 1) : 0
  );

  for (let t = 0; t < targetLength; t++) {
    const timeIdx = xNew[t];
    stmap[t] = new Array(nChannels);
    for (let c = 0; c < nChannels; c++) {
      stmap[t][c] = _linearInterpolate(channelArrays[c], timeIdx, n);
    }
  }

  return stmap;
}

function _linearInterpolate(arr: Float64Array, t: number, n: number): number {
  const scaled = t * (n - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(i0 + 1, n - 1);
  const frac = scaled - i0;
  return arr[i0] * (1 - frac) + arr[i1] * frac;
}

/**
 * ROSE-Net model wrapper for browser-side inference.
 */
export class ROSENetInference {
  private model: tf.LayersModel | null = null;
  private config: ROSENetConfig;
  private isLoaded = false;

  constructor(config: Partial<ROSENetConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Load the ROSE-Net model from a URL (TF.js format).
   */
  async load(modelUrl: string): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(modelUrl);
      this.isLoaded = true;
      console.log("ROSE-Net model loaded successfully");
    } catch (error) {
      console.warn("Failed to load ROSE-Net model, using fallback:", error);
      this.isLoaded = false;
      this.model = null;
    }
  }

  /**
   * Load model from IndexedDB cache.
   */
  async loadFromCache(modelName: string = "rosenet-model"): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`indexeddb://${modelName}`);
      this.isLoaded = true;
      console.log("ROSE-Net model loaded from IndexedDB");
    } catch (error) {
      console.warn("No cached ROSE-Net model found:", error);
      this.isLoaded = false;
      this.model = null;
    }
  }

  /**
   * Save model to IndexedDB.
   */
  async saveToCache(modelName: string = "rosenet-model"): Promise<void> {
    if (!this.model) throw new Error("No model to save");
    await this.model.save(`indexeddb://${modelName}`);
    console.log("ROSE-Net model saved to IndexedDB");
  }

  /**
   * Preprocess RGB signals into a TF.js tensor for model input.
   *
   * Output shape: [1, channels, 1, time_steps] — matching the
   * PyTorch model's input convention.
   */
  preprocess(rgbSignals: number[][], targetLength?: number): tf.Tensor {
    const len = targetLength ?? this.config.inputLength;
    const stmap = generateSTMap(rgbSignals, ["rgb", "yuv"], len);

    if (stmap.length === 0) {
      return tf.zeros([1, this.config.inChannels, 1, len]);
    }

    // Convert stmap (T, C) to tensor (1, C, 1, T)
    const flatData = new Float32Array(stmap.length * stmap[0].length);
    const nChannels = stmap[0].length;
    for (let t = 0; t < stmap.length; t++) {
      for (let c = 0; c < nChannels; c++) {
        flatData[c * stmap.length + t] = stmap[t][c];
      }
    }

    return tf.tensor(flatData, [1, nChannels, 1, stmap.length]);
  }

  /**
   * Run ROSE-Net SpO2 inference.
   *
   * Args:
   *   rRaw   — raw red channel means per frame
   *   gRaw   — raw green channel means per frame
   *   bRaw   — raw blue channel means per frame
   *   hr     — estimated heart rate (bpm), used for empirical fallback
   *   sqi    — signal quality index (0-1)
   *
   * Returns:
   *   SpO2 estimate with method and confidence metadata.
   */
  async estimateSpO2(
    rRaw: number[],
    gRaw: number[],
    bRaw: number[],
    hr: number | null,
    sqi: number,
  ): Promise<SpO2Result> {
    if (rRaw.length !== gRaw.length || rRaw.length !== bRaw.length) {
      return { spo2: null, rRatio: null, quality: 0, method: "empirical", confidence: 0, modelUsed: false };
    }

    if (rRaw.length < 30) {
      return { spo2: null, rRatio: null, quality: 0, method: "empirical", confidence: 0, modelUsed: false };
    }

    const rgbSignals = rRaw.map((_, i) => [rRaw[i], gRaw[i], bRaw[i]]);

    // Try ROSE-Net model
    if (this.isLoaded && this.model) {
      const inputTensor = this.preprocess(rgbSignals);
      try {
        const output = this.model.predict(inputTensor) as tf.Tensor;
        const data = await output.data();
        const spo2 = data[0];
        const result: number = Math.min(100, Math.max(70, spo2));
        inputTensor.dispose();
        output.dispose();
        return {
          spo2: Math.round(result),
          rRatio: null,
          quality: Math.min(1, sqi / 0.7),
          method: "rosenet",
          confidence: 0.85,
          modelUsed: true,
        };
      } catch (error) {
        console.warn("ROSE-Net inference failed, falling back:", error);
        inputTensor.dispose();
      }
    }

    // Fallback: empirical AC/DC ratio-of-ratios
    const empResult = computeEmpiricalSpO2(rRaw, gRaw, bRaw, hr, sqi);
    return empResult;
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

/**
 * Empirical SpO2 estimation via AC/DC ratio-of-ratios.
 * R = (AC_red/DC_red) / (AC_green/DC_green)
 * SpO2 = 110 - 25 * R
 *
 * This is the traditional method that ROSE-Net improves upon.
 */
export function computeEmpiricalSpO2(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  hr: number | null,
  sqi: number,
  spo2Offset: number = 0,
): SpO2Result {
  if (hr === null || hr <= 0) {
    return { spo2: null, rRatio: null, quality: 0, method: "empirical", confidence: 0, modelUsed: false };
  }

  const acR = spectralAmplitude(rRaw, 30, hr / 60, 0.15);
  const acG = spectralAmplitude(gRaw, 30, hr / 60, 0.15);
  const dcR = mean(rRaw);
  const dcG = mean(gRaw);

  if (dcR <= 1e-9 || dcG <= 1e-9 || acG <= 1e-12) {
    return { spo2: null, rRatio: null, quality: 0, method: "empirical", confidence: 0, modelUsed: false };
  }

  const rRatio = (acR / dcR) / (acG / dcG);
  let spo2 = 110 - 25 * rRatio + spo2Offset;

  // Skin tone correction (Fitzpatrick 1-6, default 3.5)
  spo2 += (3.5 - 3.5) * 0.3; // 0 by default

  spo2 = Math.min(100, Math.max(70, spo2));

  return {
    spo2: Math.round(spo2),
    rRatio: rRatio,
    quality: Math.min(1, sqi / 0.7),
    method: "empirical",
    confidence: 0.5,
    modelUsed: false,
  };
}

function spectralAmplitude(signal: number[], fs: number, targetFreq: number, tolHz: number): number {
  const n = signal.length;
  if (n < 8) return 0;

  const detrended = detrend(signal);
  const win = new Array(n);
  let winSq = 0;
  for (let i = 0; i < n; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    winSq += win[i] * win[i];
  }
  const wNorm = winSq / n || 1;

  const df = fs / n;
  const kTarget = Math.round(targetFreq / df);
  const kTol = Math.max(1, Math.round(tolHz / df));

  let amp = 0;
  for (let k = Math.max(1, kTarget - kTol); k <= Math.min(Math.floor(n / 2), kTarget + kTol); k++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      re += detrended[i] * win[i] * Math.cos(angle);
      im -= detrended[i] * win[i] * Math.sin(angle);
    }
    amp += Math.sqrt(re * re + im * im) / (n * Math.sqrt(wNorm));
  }
  return amp;
}

function mean(x: number[]): number {
  if (x.length === 0) return 0;
  return x.reduce((a, b) => a + b, 0) / x.length;
}

function detrend(x: number[]): number[] {
  const n = x.length;
  if (n < 3) return x.slice();
  const xm = mean(x);
  const tm = (n - 1) / 2;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - tm) * (x[i] - xm);
    den += (i - tm) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = xm - slope * tm;
  return x.map((v, i) => v - (slope * i + intercept));
}

// Singleton instance for app-wide use
let rosenetInstance: ROSENetInference | null = null;

export function getROSENetInstance(config?: Partial<ROSENetConfig>): ROSENetInference {
  if (!rosenetInstance) {
    rosenetInstance = new ROSENetInference(config);
  }
  return rosenetInstance;
}

export async function estimateSpO2(
  rRaw: number[],
  gRaw: number[],
  bRaw: number[],
  hr: number | null,
  sqi: number,
): Promise<SpO2Result> {
  const instance = getROSENetInstance();
  return instance.estimateSpO2(rRaw, gRaw, bRaw, hr, sqi);
}

export function resetROSENetInstance(): void {
  if (rosenetInstance) {
    rosenetInstance.dispose();
    rosenetInstance = null;
  }
}
