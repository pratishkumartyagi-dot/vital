"use client";

import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface FaceLandmark {
  x: number;
  y: number;
  z: number;
}

export interface MediapipeROIResult {
  landmarks: FaceLandmark[];
  roi: number[][];
  coords: { x: number; y: number; w: number; h: number };
}

/**
 * MediaPipe AI Edge FaceMesh-based ROI detector for rPPG.
 *
 * Uses Google's MediaPipe Tasks Vision face_landmarker model (468 3D landmarks)
 * to extract precise facial ROI regions for pulse signal extraction.
 * Model: https://github.com/google-ai-edge/mediapipe
 *
 * The face mesh provides 468 landmarks covering the full face contour, eyes,
 * eyebrows, nose, lips, and forehead — enabling sub-region ROI extraction
 * that targets areas with strong pulse signals (cheeks, forehead, nose bridge).
 */
export class MediapipeFaceROI {
  private faceLandmarker: FaceLandmarker | null = null;
  private vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      const wasmPath =
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm";
      this.vision = await FilesetResolver.forVisionTasks(wasmPath);

      const modelUrl =
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/landmarker/float16/1/face_landmarker.f16.task";

      this.faceLandmarker = await FaceLandmarker.createFromOptions(this.vision, {
        baseOptions: {
          modelAssetPath: modelUrl,
        },
        runningMode: "IMAGE",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      this.initialized = true;
    } catch (err) {
      console.warn("MediaPipe FaceMesh init failed, ROI detection disabled:", err);
      this.initialized = false;
    }
  }

  /**
   * Detect face landmarks in a video frame and extract ROI regions.
   * Returns null if MediaPipe is unavailable or no face detected.
   */
  detect(
    video: HTMLVideoElement,
    canvasWidth: number = 64,
    canvasHeight: number = 64,
  ): MediapipeROIResult | null {
    if (!this.faceLandmarker || !this.initialized) return null;

    try {
      const result = this.faceLandmarker.detect(video);

      if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
        return null;
      }

      const faceLandmarks: NormalizedLandmark[] = result.faceLandmarks[0];
      if (!faceLandmarks || faceLandmarks.length === 0) return null;

      // Convert to array of {x,y,z}
      const landmarks: FaceLandmark[] = faceLandmarks.map((lm) => ({
        x: lm.x * canvasWidth,
        y: lm.y * canvasHeight,
        z: lm.z || 0,
      }));

      // Compute bounding box of all landmarks
      const xs = faceLandmarks.map((lm) => lm.x);
      const ys = faceLandmarks.map((lm) => lm.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      const roi = this.extractRoiRegions(faceLandmarks, canvasWidth, canvasHeight);

      return {
        landmarks,
        roi,
        coords: {
          x: minX * canvasWidth,
          y: minY * canvasHeight,
          w: (maxX - minX) * canvasWidth,
          h: (maxY - minY) * canvasHeight,
        },
      };
    } catch (err) {
      console.warn("MediaPipe face detection failed:", err);
      return null;
    }
  }

  /**
   * Extract ROI regions from face landmarks for robust signal averaging.
   * Uses 5 sub-regions targeting high-pulse-amplitude areas.
   */
  private extractRoiRegions(
    landmarks: NormalizedLandmark[],
    width: number,
    height: number,
  ): number[][] {
    // Region definitions as normalized rectangles (x, y, w, h)
    const regions = [
      { x: 0.15, y: 0.05, w: 0.30, h: 0.20 }, // forehead
      { x: 0.05, y: 0.35, w: 0.25, h: 0.30 }, // left cheek
      { x: 0.70, y: 0.35, w: 0.25, h: 0.30 }, // right cheek
      { x: 0.40, y: 0.35, w: 0.20, h: 0.35 }, // nose bridge
      { x: 0.10, y: 0.60, w: 0.35, h: 0.25 }, // lower face / jawline
      { x: 0.55, y: 0.60, w: 0.35, h: 0.25 }, // lower face / jawline
    ];

    const roi: number[][] = [];

    for (const region of regions) {
      const startX = Math.max(0, Math.floor(width * region.x));
      const endX = Math.min(width, Math.floor(width * (region.x + region.w)));
      const startY = Math.max(0, Math.floor(height * region.y));
      const endY = Math.min(height, Math.floor(height * (region.y + region.h)));

      const regionData: number[] = [];
      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          regionData.push(px, py, endX - startX, endY - startY);
        }
      }
      if (regionData.length > 0) roi.push(regionData);
    }

    return roi;
  }

  /**
   * Given image data from a canvas, extract RGB means from the 5 face regions
   * using MediaPipe face landmarks. Used as a fallback when MediaPipe is
   * not available, returns the static 5-region layout.
   */
  extractRoiFromImageData(
    imageData: ImageData,
    width: number,
    height: number,
    landmarks?: FaceLandmark[],
  ): number[][] {
    const data = imageData.data;

    if (!landmarks || landmarks.length === 0) {
      // Fallback: use static region grid
      return this.extractStaticRoi(imageData, width, height);
    }

    // Use landmark-derived bounding boxes
    const xs = landmarks.map((l) => l.x);
    const ys = landmarks.map((l) => l.y);
    const minX = Math.max(0, Math.min(...xs) - 5);
    const maxX = Math.min(width, Math.max(...xs) + 5);
    const minY = Math.max(0, Math.min(...ys) - 5);
    const maxY = Math.min(height, Math.max(...ys) + 5);
    const w = maxX - minX;
    const h = maxY - minY;

    const regions = [
      { x: 0.15, y: 0.05, rw: 0.30, rh: 0.20 },
      { x: 0.05, y: 0.35, rw: 0.25, rh: 0.30 },
      { x: 0.70, y: 0.35, rw: 0.25, rh: 0.30 },
      { x: 0.40, y: 0.35, rw: 0.20, rh: 0.35 },
      { x: 0.10, y: 0.60, rw: 0.35, rh: 0.25 },
    ];

    const roi: number[][] = [];
    for (const region of regions) {
      const startX = Math.max(minX, Math.floor(minX + w * region.x));
      const endX = Math.min(maxX, Math.floor(minX + w * (region.x + region.rw)));
      const startY = Math.max(minY, Math.floor(minY + h * region.y));
      const endY = Math.min(maxY, Math.floor(minY + h * (region.y + region.rh)));

      const regionData: number[] = [];
      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          const idx = (py * width + px) * 4;
          regionData.push(data[idx], data[idx + 1], data[idx + 2]);
        }
      }
      if (regionData.length > 0) roi.push(regionData);
    }

    return roi;
  }

  private extractStaticRoi(imageData: ImageData, width: number, height: number): number[][] {
    const data = imageData.data;
    // Reuse the existing static 5-region approach from page.tsx
    const regions = [
      { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
      { x: 0.6, y: 0.1, w: 0.3, h: 0.3 },
      { x: 0.35, y: 0.35, w: 0.3, h: 0.3 },
      { x: 0.1, y: 0.6, w: 0.35, h: 0.25 },
      { x: 0.55, y: 0.6, w: 0.35, h: 0.25 },
    ];

    const roi: number[][] = [];
    for (const region of regions) {
      const startX = Math.floor(width * region.x);
      const endX = Math.floor(width * (region.x + region.w));
      const startY = Math.floor(height * region.y);
      const endY = Math.floor(height * (region.y + region.h));

      const regionData: number[] = [];
      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          const idx = (py * width + px) * 4;
          regionData.push(data[idx], data[idx + 1], data[idx + 2]);
        }
      }
      roi.push(regionData);
    }
    return roi;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  dispose(): void {
    if (this.faceLandmarker) {
      this.faceLandmarker.close();
      this.faceLandmarker = null;
    }
    this.initialized = false;
  }
}

// Singleton instance
let mediapipeInstance: MediapipeFaceROI | null = null;

export function getMediapipeFaceROI(): MediapipeFaceROI {
  if (!mediapipeInstance) {
    mediapipeInstance = new MediapipeFaceROI();
  }
  return mediapipeInstance;
}

export function resetMediapipeFaceROI(): void {
  if (mediapipeInstance) {
    mediapipeInstance.dispose();
    mediapipeInstance = null;
  }
}
