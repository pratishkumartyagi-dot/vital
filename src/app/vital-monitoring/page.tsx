"use client";

import { useRef, useState, useCallback, useEffect } from "react";

import Link from 'next/link';
import { analyzeVitals, type VitalAnalysis, type CreatinineMetrics, estimateEgfr, butterworthBandpass, posPulse, selectBestPpgChannel, estimateSpo2RedGreen, estimateHbRedGreen, resolveZlo } from './ppg-analysis';
import { 
  addVitalReading, 
  getVitalReadings, 
  downloadReadings, 
  getReadingCount,
  getMaxReadings,
  deleteVitalReading,
  type VitalReading 
} from '@/lib/vital-storage';

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    if (msg.includes("permission denied") || msg.includes("notallowed")) {
      return "Camera permission denied. Please allow camera access in your browser settings and try again.";
    }
    if (msg.includes("notfound")) {
      return "No camera found on this device.";
    }
    return err.message;
  }
  return "Unknown error accessing camera.";
}

// Generate synthetic PPG for demo mode (delegated to ppg-analysis logic)
function generateSyntheticPPG(seconds: number, fs: number, bpm: number, t0 = 0): number[] {
  const n = Math.floor(seconds * fs + 1e-9);
  const out: number[] = new Array(n);
  const heartHz = bpm / 60;
  const respHz = 0.25;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fs;
    const beat = Math.sin(2 * Math.PI * heartHz * t);
    const dicrotic = 0.35 * Math.sin(2 * Math.PI * heartHz * t - 0.6);
    const pulse = Math.max(0, beat + dicrotic);
    const baseline = 0.5 + 0.2 * Math.sin(2 * Math.PI * respHz * t);
    const noise = (Math.random() - 0.5) * 0.04;
    out[i] = 128 + 30 * pulse * baseline + noise * 6;
  }
  return out;
}

export default function VitalMonitoringPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const [permission, setPermission] = useState<string>("unknown");
  const [cameraMode, setCameraMode] = useState<"front" | "rear">("front");
  const [torchOn, setTorchOn] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [bpm, setBpm] = useState<number | null>(null);
  const [spo2, setSpo2] = useState<number | null>(null);
  const [bp, setBp] = useState<string | null>(null);
  const [sugar, setSugar] = useState<number | null>(null);
  const [hb, setHb] = useState<number | null>(null);
  const [creatinine, setCreatinine] = useState<CreatinineMetrics | null>(null);
  const [signalQuality, setSignalQuality] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "warming" | "measuring" | "finalized">("idle");
  const [countdown, setCountdown] = useState<number>(0);
  const [finalReading, setFinalReading] = useState<{
    bpm: number;
    spo2: number | null;
    hb: number | null;
    bpSys: number | null;
    bpDia: number | null;
    sugar: number | null;
    signalQuality: number;
    respRate: number;
    sdnn: number;
    rmssd: number;
    pnn50: number;
    creatinine: number | null;
    egfr: number | null;
    creatinineClass: string;
    creatinineConfidence: number;
    creatinineMethod: string;
    rhythm: string;
  } | null>(null);
  const [respRate, setRespRate] = useState<number | null>(null);
  const [hrv, setHrv] = useState<{ sdnn: number; rmssd: number; pnn50: number } | null>(null);
  const [rhythm, setRhythm] = useState<"regular" | "irregular" | "unknown">("unknown");
  const [fingerDetected, setFingerDetected] = useState(false);
  const [skinDetected, setSkinDetected] = useState(false);
  const [fingerPressure, setFingerPressure] = useState<"low" | "adequate" | "excessive" | null>(null);
  const [pressureDetail, setPressureDetail] = useState<"TOO_HEAVY" | "UNSTABLE" | "ADEQUATE" | "TOO_LIGHT" | "WEAK_SIGNAL" | "NO_FINGER" | null>(null);
  const valuesRef = useRef<number[]>([]);
  const redRef = useRef<number[]>([]);
  const greenRef = useRef<number[]>([]);
  const blueRef = useRef<number[]>([]);
  const ppgBufferRef = useRef<number[]>([]);
  const syntheticBufferRef = useRef<number[]>([]);
  const synthTimeRef = useRef<number>(0);
  const baseHrvRef = useRef<{ sdnn: number; rmssd: number; pnn50: number }>({ sdnn: 45, rmssd: 38, pnn50: 12 });
  const fpsRef = useRef<number>(30);
  const frameIntervalsRef = useRef<number[]>([]);
  const fingerHistoryRef = useRef<boolean[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const animFrame = useRef<number>(0);
  const lastBpmUpdate = useRef<number>(0);
  const sessionStartRef = useRef<number>(0);
  const samplesRef = useRef<{ bpm: number; spo2: number | null; hb: number | null; bpSys: number | null; bpDia: number | null; sugar: number | null; creatinine: number | null; signalQuality: number; respRate: number; sdnn: number; rmssd: number; pnn50: number; rhythm: string }[]>([]);
  const autoStopTimerRef = useRef<number | null>(null);
  const countdownTimerRef = useRef<number | null>(null);
  const spo2SmoothedRef = useRef<number | null>(null);
  const hbSmoothedRef = useRef<number | null>(null);
  // Pressure state machine refs
  const pressureStateRef = useRef<"NO_FINGER" | "TOO_HEAVY" | "UNSTABLE" | "ADEQUATE" | "TOO_LIGHT" | "WEAK_SIGNAL" | null>(null);
  const prevDcLevelRef = useRef<number | null>(null);
  const prevPerfusionRef = useRef<number | null>(null);
  const adequateDwellRef = useRef<number>(0);
  const unstableDwellRef = useRef<number>(0);

  const checkPermission = useCallback(async () => {
    try {
      const perms = (navigator as Navigator & { permissions?: { query?: (params: { name: string }) => Promise<{ state: string }> } }).permissions;
      if (perms && perms.query) {
        const result = await perms.query({ name: "camera" });
        setPermission(result.state);
      }
    } catch {
      setPermission("unknown");
    }
  }, []);

  const startedRef = useRef(false);
  const cameraOnRef = useRef(cameraOn);
  const loadingRef = useRef(loading);
  const phaseRef = useRef(phase);
  const toggleCameraRef = useRef<(() => void) | null>(null);

  const [patient, setPatient] = useState<{
    name: string;
    age: string;
    sex: "female" | "male" | "";
    lmp: string;
    pregnancy: "pregnant" | "non-pregnant" | "";
    menopause: "yes" | "no" | "";
  } | null>(null);
  const [form, setForm] = useState({
    name: "",
    age: "",
    sex: "" as "" | "female" | "male",
    lmp: "",
    pregnancy: "" as "" | "pregnant" | "non-pregnant",
    menopause: "" as "" | "yes" | "no",
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [readingCount, setReadingCount] = useState(0);
  const [readings, setReadings] = useState<VitalReading[]>([]);
  const [readingsLoading, setReadingsLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [exporting, setExporting] = useState(false);
  
  const loadReadings = useCallback(async () => {
    setReadingsLoading(true);
    try {
      const data = await getVitalReadings({ limit: 50 });
      setReadings(data);
      setReadingCount(data.length);
    } catch (error) {
      console.error("Failed to load readings:", error);
    } finally {
      setReadingsLoading(false);
    }
  }, []);
  
  useEffect(() => {
    loadReadings();
  }, [loadReadings]);
  
  const videoChunksRef = useRef<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingRef = useRef(false);

  useEffect(() => {
    cameraOnRef.current = cameraOn;
  }, [cameraOn]);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) window.clearInterval(countdownTimerRef.current);
      if (autoStopTimerRef.current) window.clearTimeout(autoStopTimerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    checkPermission();
    const timer = setTimeout(() => {
      if (!startedRef.current && !cameraOnRef.current && !loadingRef.current && patient) {
        startedRef.current = true;
        toggleCameraRef.current?.();
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [checkPermission, patient]);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = 300;
    canvas.height = 120;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const values = valuesRef.current;
    if (values.length < 2) return;
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const step = canvas.width / Math.max(values.length - 1, 1);
    
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = i * step;
      const y = canvas.height - ((values[i] - min) / range) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const greenValues = greenRef.current;
    if (greenValues.length > 1) {
      const gMin = Math.min(...greenValues);
      const gMax = Math.max(...greenValues);
      const gRange = gMax - gMin || 1;
      const gStep = canvas.width / Math.max(greenValues.length - 1, 1);
      for (let i = 0; i < greenValues.length; i++) {
        const x = i * gStep;
        const y = canvas.height - ((greenValues[i] - gMin) / gRange) * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }, []);

  const processFrame = useCallback(() => {
    if (!cameraOnRef.current) return;
    const video = videoRef.current;
    if (!demoMode && (!video || !video.videoWidth)) {
      animFrame.current = requestAnimationFrame(processFrame);
      return;
    }

    // Increased resolution from 64x64 to 160x120 for better skin pixel count
    // and more reliable facial ROI extraction (forehead, cheeks)
    const PROC_W = 160;
    const PROC_H = 120;
    const c = document.createElement("canvas");
    c.width = PROC_W;
    c.height = PROC_H;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    if (video) ctx.drawImage(video, 0, 0, PROC_W, PROC_H);
    const data = ctx.getImageData(0, 0, PROC_W, PROC_H).data;

    let r = 0, g = 0, b = 0, count = 0;
    let frameOk = false;  // Explicitly initialize to false
    if (cameraMode === "rear") {
      // Rear camera: center ROI (finger on lens) - smaller, more focused region
      // Use 25%-75% to get a tighter center region (48x36 at 160x120 -> 80x60)
      const cx0 = Math.floor(PROC_W * 0.25), cy0 = Math.floor(PROC_H * 0.25);
      const cx1 = Math.floor(PROC_W * 0.75), cy1 = Math.floor(PROC_H * 0.75);
      for (let y = cy0; y < cy1; y++) {
        for (let x = cx0; x < cx1; x++) {
          const i = (y * PROC_W + x) * 4;
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          count++;
        }
      }
      frameOk = count > 0;  // Valid if we got pixels
    } else {
      // Front camera: Focus on upper face region (forehead + upper cheeks)
      // This avoids mouth/nose movement artifacts and gets better rPPG signal
      const foreheadY0 = 0;
      const foreheadY1 = Math.floor(PROC_H * 0.45);  // Upper 45% = forehead region
      const foreheadX0 = Math.floor(PROC_W * 0.2);   // Center 60% width
      const foreheadX1 = Math.floor(PROC_W * 0.8);
      
      let skinR = 0, skinG = 0, skinB = 0, skinCount = 0;
      
      // Adaptive skin detection: compute frame statistics first
      let totalBrightness = 0, brightPixels = 0;
      for (let y = foreheadY0; y < foreheadY1; y++) {
        for (let x = foreheadX0; x < foreheadX1; x++) {
          const i = (y * PROC_W + x) * 4;
          const pr = data[i], pg = data[i + 1], pb = data[i + 2];
          const brightness = (pr + pg + pb) / 3;
          totalBrightness += brightness;
          brightPixels++;
        }
      }
      const avgBrightness = brightPixels > 0 ? totalBrightness / brightPixels : 128;
      
      // Adaptive brightness gate: reject very dark/bright relative to frame mean
      const brightLow = Math.max(30, avgBrightness * 0.3);
      const brightHigh = Math.min(250, avgBrightness * 1.8);
      
      for (let y = foreheadY0; y < foreheadY1; y++) {
        for (let x = foreheadX0; x < foreheadX1; x++) {
          const i = (y * PROC_W + x) * 4;
          const pr = data[i];
          const pg = data[i + 1];
          const pb = data[i + 2];
          const brightness = (pr + pg + pb) / 3;
          if (brightness < brightLow || brightness > brightHigh) continue;
          
          // YCbCr skin detection with slightly wider range for diverse skin tones
          const cb = 128 - 0.168736 * pr - 0.331264 * pg + 0.5 * pb;
          const cr = 128 + 0.5 * pr - 0.418688 * pg - 0.081312 * pb;
          if (cb >= 70 && cb <= 135 && cr >= 125 && cr <= 180) {
            skinR += pr;
            skinG += pg;
            skinB += pb;
            skinCount++;
          }
        }
      }
      
      // Require minimum skin pixels (0.5% of forehead ROI area)
      const roiArea = (foreheadY1 - foreheadY0) * (foreheadX1 - foreheadX0);
      const hasSkin = skinCount >= Math.max(50, 0.005 * roiArea);
      if (cameraMode === "front" && !demoMode) setSkinDetected(hasSkin);
      
      if (hasSkin) {
        r = skinR;
        g = skinG;
        b = skinB;
        count = skinCount;
        frameOk = true;
      } else {
        // Fallback: use full forehead region average (not full frame)
        let fallbackR = 0, fallbackG = 0, fallbackB = 0, fallbackCount = 0;
        for (let y = foreheadY0; y < foreheadY1; y++) {
          for (let x = foreheadX0; x < foreheadX1; x++) {
            const i = (y * PROC_W + x) * 4;
            fallbackR += data[i];
            fallbackG += data[i + 1];
            fallbackB += data[i + 2];
            fallbackCount++;
          }
        }
        if (fallbackCount > 0) {
          r = fallbackR;
          g = fallbackG;
          b = fallbackB;
          count = fallbackCount;
        } else {
          // Ultimate fallback: full frame
          for (let i = 0; i < data.length; i += 4) {
            r += data[i];
            g += data[i + 1];
            b += data[i + 2];
            count++;
          }
        }
        frameOk = false;
      }
    }
    const avgR = count ? r / count : 128;
    const avgG = count ? g / count : 128;
    const avgB = count ? b / count : 128;
    
    // Finger detection state machine for rear camera (temporal stability)
    // Track recent finger detection results to avoid flickering
    if (!fingerHistoryRef.current) fingerHistoryRef.current = [];
    
    const now = performance.now();
    if (lastFrameTimeRef.current > 0) {
      const dt = now - lastFrameTimeRef.current;
      if (dt > 0 && dt < 200) { // Ignore outliers >200ms (5fps)
        // Store recent frame intervals for robust median-based FPS estimation
        if (!frameIntervalsRef.current) frameIntervalsRef.current = [];
        frameIntervalsRef.current.push(dt);
        if (frameIntervalsRef.current.length > 30) frameIntervalsRef.current.shift();
        
        // Use median of recent intervals for stable FPS
        if (frameIntervalsRef.current.length >= 5) {
          const sorted = [...frameIntervalsRef.current].sort((a, b) => a - b);
          const medianDt = sorted[Math.floor(sorted.length / 2)];
          fpsRef.current = 1000 / medianDt;
        } else {
          // Fallback to EMA for first few frames
          const instFps = 1000 / dt;
          fpsRef.current = fpsRef.current * 0.9 + instFps * 0.1;
        }
      }
    }
    lastFrameTimeRef.current = now;

    // Finger detection for rear camera - uses color signature + temporal stability
    // Finger PPG signature: Red >> Green ≈ Blue (red penetrates tissue, green/blue absorbed by blood)
    let isFingerFrame = false;
    if (cameraMode === "rear") {
      // Adaptive thresholds based on frame statistics
      const brightness = (avgR + avgG + avgB) / 3;
      
      // Finger criteria:
      // 1. Red in good range (not too dark, not saturated)
      // 2. Red significantly higher than green AND blue (ratios > 1.3)
      // 3. Green and blue similar (both absorbed by blood) - within 40%
      // 4. Sufficient signal strength
      // 5. Brightness in reasonable range for finger on camera
      
      const redGreenRatio = avgG > 1 ? avgR / avgG : 0;
      const redBlueRatio = avgB > 1 ? avgR / avgB : 0;
      const greenBlueDiff = Math.abs(avgG - avgB);
      const greenBlueMax = Math.max(avgG, avgB);
      
      const redInRange = avgR >= 70 && avgR <= 245;
      const redDominant = redGreenRatio > 1.3 && redBlueRatio > 1.3;
      const greenBlueSimilar = greenBlueMax > 1 ? (greenBlueDiff / greenBlueMax) < 0.4 : true;
      const sufficientSignal = avgR > 70;
      const brightnessOk = brightness > 50 && brightness < 220;
      const greenBlueReasonable = avgG > 15 && avgB > 15 && avgG < 200 && avgB < 200;
      
      isFingerFrame = redInRange && redDominant && greenBlueSimilar && sufficientSignal && brightnessOk && greenBlueReasonable;
      
      // Temporal smoothing: require 3 consecutive consistent frames
      if (!fingerHistoryRef.current) fingerHistoryRef.current = [];
      fingerHistoryRef.current.push(isFingerFrame);
      if (fingerHistoryRef.current.length > 5) fingerHistoryRef.current.shift();
      
      // Majority vote over recent frames (need 3 of 5 or 60%)
      const fingerVotes = fingerHistoryRef.current.filter(v => v).length;
      const fingerMajority = fingerVotes >= Math.max(3, Math.floor(fingerHistoryRef.current.length * 0.6));
      
      if (!demoMode) setFingerDetected(fingerMajority);
    } else {
      // Front camera - no finger detection needed
      if (!demoMode) setFingerDetected(false);
      // Reset finger history when not in rear mode
      if (fingerHistoryRef.current) fingerHistoryRef.current = [];
    }

    if (demoMode) {
      const fs = Math.max(10, Math.min(60, fpsRef.current || 30));
      // Advance a continuous synthetic clock so the generated pulse does not
      // reset phase every frame (which previously created a 10 Hz artifact and
      // hid the 72 BPM fundamental).
      const seconds = 1 / fs;
      const sample = generateSyntheticPPG(seconds, fs, 72, synthTimeRef.current);
      synthTimeRef.current += seconds;
      syntheticBufferRef.current.push(...sample);
      if (syntheticBufferRef.current.length > 1800) {
        syntheticBufferRef.current.splice(0, syntheticBufferRef.current.length - 1800);
      }
      const collecting = phaseRef.current === "measuring";
      for (const v of sample) {
        valuesRef.current.push(128 + (v - 128) * 1.2);
        if (valuesRef.current.length > 180) valuesRef.current.shift();
        if (collecting) {
          redRef.current.push(128 + (v - 128) * 1.2);
          greenRef.current.push(v);
          blueRef.current.push(128 + (v - 128) * 0.4);
          if (redRef.current.length > 1800) redRef.current.shift();
          if (greenRef.current.length > 1800) greenRef.current.shift();
          if (blueRef.current.length > 1800) blueRef.current.shift();
        }
      }
      setFingerDetected(true);
    } else {
      // Only collect the analysis buffers during the measuring phase and only
      // when the frame actually contains the signal source. For rear mode, use
      // finger detection state (with temporal stability) to determine collection.
      const fingerPresent = fingerDetected;  // Uses temporally-smoothed detection
      const collecting = phaseRef.current === "measuring" && frameOk && (cameraMode !== "rear" || fingerPresent);
      if (collecting) {
        ppgBufferRef.current.push(avgG);
        if (ppgBufferRef.current.length > 1800) ppgBufferRef.current.shift();
        redRef.current.push(avgR);
        greenRef.current.push(avgG);
        blueRef.current.push(avgB);
        if (redRef.current.length > 1800) redRef.current.shift();
        if (greenRef.current.length > 1800) greenRef.current.shift();
        if (blueRef.current.length > 1800) blueRef.current.shift();
      }

      valuesRef.current.push(avgR);
      if (valuesRef.current.length > 180) valuesRef.current.shift();
    }

    // Run comprehensive vitals analysis using centralized module
    const sessionStart = sessionStartRef.current;
    const elapsedMs = sessionStart > 0 ? Date.now() - sessionStart : 0;
    const isMeasuring = phaseRef.current === "measuring" && elapsedMs >= 10000 && elapsedMs < 130000;
    
    // Update signal quality and vitals every 2 seconds
    const nowMs = Date.now();
    if (nowMs - lastBpmUpdate.current > 2000) {
      lastBpmUpdate.current = nowMs;

      const fs = Math.max(10, Math.min(60, fpsRef.current || 30));
      // All buffers are cut to the same most-recent 30 s window so the pulse
      // analysis, the POS projection and the AC/DC ratios always operate on
      // exactly the same samples.
      const windowLen = Math.floor(fs * 30);
      const ppgWin = ppgBufferRef.current.slice(-Math.min(ppgBufferRef.current.length, windowLen));
      const redWin = redRef.current.slice(-Math.min(redRef.current.length, windowLen));
      const greenWin = greenRef.current.slice(-Math.min(greenRef.current.length, windowLen));
      const blueWin = blueRef.current.slice(-Math.min(blueRef.current.length, windowLen));
      let analysisSource: number[];
      if (demoMode) {
        analysisSource = syntheticBufferRef.current.slice(-Math.min(syntheticBufferRef.current.length, windowLen));
      } else if (cameraMode === "front") {
        // Front camera: drive the pulse analysis from the POS chrominance
        // pulse, which rejects illumination/motion artifacts that a raw
        // averaged channel keeps. Fall back to the green skin-ROI trace while
        // the POS window is still filling. With winLen=120 (4s at 30fps) and
        // 50% overlap, we need ~6s of data for reliable POS output.
        const pos = posPulse(redWin, greenWin, blueWin, Math.max(120, Math.round(fs * 4)));
        analysisSource = pos.length >= Math.floor(fs * 6) ? pos : ppgWin;
      } else {
        // Rear camera: pick the single channel (red or green) with the best
        // spectral SNR in the cardiac band. For finger PPG, RED is preferred
        // because it penetrates finger tissue better and gives higher SNR.
        // Green is fallback if red is saturated or has significantly lower SNR.
        const selected = selectBestPpgChannel(redWin, greenWin, blueWin, fs, true);
        analysisSource = selected.length >= Math.floor(fs * 5) ? selected : ppgWin;
      }
      const analysis: VitalAnalysis = analyzeVitals(
        analysisSource,
        fs,
        redWin,
        greenWin,
        blueWin,
        {
          cameraMode,
          ageYears: patient ? parseInt(patient.age, 10) || null : null,
          sex: patient ? (patient.sex as "male" | "female") : null,
          // Respiratory rate is extracted from baseline wander (0.05-0.5 Hz),
          // which is BELOW the cardiac band. For rear/demo mode analysisSource
          // is the 0.5-3.5 Hz bandpassed cardiac pulse, so passing it here
          // removed the respiratory band entirely and the estimator returned
          // ~0.0 for every rear-camera measurement. The raw green skin-ROI
          // trace (and the synthetic buffer in demo mode) carry the full
          // respiratory modulation and are used instead. For the front camera
          // the raw green trace is used directly too — the POS chroma pulse is
          // detrended but still carries the respiratory amplitude modulation.
          respSource:
            greenWin.length >= Math.floor(fs * 5) ? greenWin : analysisSource,
        }
      );
      const hr = analysis.bpm !== null ? Math.min(Math.max(analysis.bpm, 40), 180) : null;

      // Use SQI from analyzeVitals (0-100)
      const sqi = analysis.heart.signalQuality;
      setSignalQuality(sqi / 100);

      // Finger pressure state machine (rear camera only)
      if (cameraMode === "rear" && !demoMode) {
        // Use raw green window for DC and perfusion
        const dcLevel = greenWin.length > 0 ? greenWin.reduce((a, b) => a + b, 0) / greenWin.length : 0;
        const perfusion = analysis.heart.perfusionIndex ?? 0;
        const snr = sqi; // Use composite SQI as SNR proxy (0-100)
        const prevDc = prevDcLevelRef.current;
        const prevPi = prevPerfusionRef.current;

        // Update history
        prevDcLevelRef.current = dcLevel;
        prevPerfusionRef.current = perfusion;

        // Finger presence check (min coverage) - more lenient
        // Allow some frames where finger detection briefly fails
        const fingerCoverage = (fingerDetected || dcLevel > 40) && dcLevel > 20 && dcLevel < 240;
        if (!fingerCoverage) {
          pressureStateRef.current = "NO_FINGER";
          adequateDwellRef.current = 0;
          unstableDwellRef.current = 0;
        } else {
          // Trend detection
          const dcRising = prevDc !== null && dcLevel > prevDc + 2; // DC increasing (finger pressing down)
          const piCollapsed = prevPi !== null && perfusion < prevPi * 0.7 && prevPi > 0.5; // PI dropping sharply
          const dcDriftHigh = prevDc !== null && Math.abs(dcLevel - prevDc) > 5; // DC unstable (>5 units/2s)
          const piTooHigh = perfusion > 8; // PI implausibly high = motion artifact
          const saturated = dcLevel > 230 || dcLevel < 20; // Sensor clipping
          const lightLeakEvidence = dcLevel > 180 && perfusion < 1.0 && (blueWin.length > 0 ? blueWin.reduce((a, b) => a + b, 0) / blueWin.length : 0) > dcLevel * 0.6; // Blue ~ Green = ambient light

          let newState: typeof pressureStateRef.current = pressureStateRef.current;

          // Priority 1: TOO_HEAVY - occlusion evidence
          if (saturated || (dcRising && piCollapsed)) {
            newState = "TOO_HEAVY";
            adequateDwellRef.current = 0;
            unstableDwellRef.current = 0;
          }
          // Priority 2: UNSTABLE - motion/artifact, not pressure
          else if (piTooHigh || dcDriftHigh) {
            newState = "UNSTABLE";
            adequateDwellRef.current = 0;
            unstableDwellRef.current = Math.min(5, unstableDwellRef.current + 1);
          }
          // Priority 3: ADEQUATE - stable good signal (with dwell + hysteresis)
          else if (perfusion >= 0.5 && perfusion <= 5.0 && dcLevel >= 80 && dcLevel <= 180 && snr >= 30) {
            adequateDwellRef.current = Math.min(5, adequateDwellRef.current + 1); // ~10s dwell at 2s intervals
            if (adequateDwellRef.current >= 3) { // 6s stable
              newState = "ADEQUATE";
              unstableDwellRef.current = 0;
            } else {
              newState = "WEAK_SIGNAL"; // Building up
            }
          }
          // Priority 4: TOO_LIGHT - ambient light leak
          else if (lightLeakEvidence) {
            newState = "TOO_LIGHT";
            adequateDwellRef.current = 0;
            unstableDwellRef.current = 0;
          }
          // Priority 5: WEAK_SIGNAL - neutral fallback
          else {
            adequateDwellRef.current = 0;
            unstableDwellRef.current = 0;
            newState = "WEAK_SIGNAL";
          }

          // Hysteresis: don't flip-flop TOO_HEAVY <-> TOO_LIGHT rapidly
          if (newState === "TOO_HEAVY" && pressureStateRef.current === "TOO_LIGHT") {
            newState = "TOO_LIGHT"; // Stick with light until clear heavy evidence
          } else if (newState === "TOO_LIGHT" && pressureStateRef.current === "TOO_HEAVY") {
            newState = "TOO_HEAVY";
          }

          pressureStateRef.current = newState;
        }

        // Map to display state
        const displayMap: Record<string, "low" | "adequate" | "excessive" | null> = {
          NO_FINGER: null,
          TOO_HEAVY: "excessive",
          UNSTABLE: "adequate", // Show as adequate but with "Hold still" hint
          ADEQUATE: "adequate",
          TOO_LIGHT: "low",
          WEAK_SIGNAL: "low",
        };
        setFingerPressure(displayMap[pressureStateRef.current ?? "NO_FINGER"] ?? null);
        setPressureDetail(pressureStateRef.current);
      } else {
        setFingerPressure(null);
        setPressureDetail(null);
        pressureStateRef.current = null;
        adequateDwellRef.current = 0;
        unstableDwellRef.current = 0;
      }

      // Compute SpO2 and Hb from the gated (measuring-phase) color buffers,
      // but only when the composite signal quality is high enough for
      // ratio-of-ratios math to be meaningful (weak/motion signals previously
      // produced plausible-looking but worthless numbers).
      // Note: smartphone cameras lack IR (940nm), so SpO2 is an estimate only.
      let spo2Val: number | null = null;
      let hbVal: number | null = null;
      const minSamples = Math.floor(fs * 5);
      if (sqi >= 60 && redWin.length >= minSamples && greenWin.length >= minSamples && blueWin.length >= minSamples) {
        // Zero-phase bandpass at 0.5-3.5 Hz for proper AC/DC.
        const redFiltered = butterworthBandpass(redWin, 0.5, 3.5, fs, true);
        const greenFiltered = butterworthBandpass(greenWin, 0.5, 3.5, fs, true);
        const blueFiltered = butterworthBandpass(blueWin, 0.5, 3.5, fs, true);

        // ZLO (zero-level offset) correction: smartphone sensor black level /
        // tone-mapped DC sits above the true zero, biasing every AC/DC ratio.
        // The calibration database carries a per-model ZLO; the browser uses
        // the default when the device model is unknown.
        const zlo = resolveZlo(null);

        // Red/green ratio-of-ratios. Green is the highest-SNR camera channel
        // and a red/green absorption ratio is the accepted contact-camera proxy
        // for the red/IR ratio. The calibration constants in ppg-analysis.ts are
        // linearised for the red/green RoR (which is ~0.8-2.5, not the ~0.4-0.8
        // of red/IR), so the old `110 - 25 * R` formula returned SpO2 in the
        // 60-85% range for every healthy subject. Absolute accuracy is
        // device-specific — accuracy is device-dependent.
        spo2Val = estimateSpo2RedGreen(redWin, greenWin, redFiltered, greenFiltered, zlo);
        hbVal = estimateHbRedGreen(redWin, greenWin, redFiltered, greenFiltered, zlo);

        // Fallback: red/blue ratio-of-ratios (used only when green is
        // saturated or absent — rare under torch, but keep the branch).
        if (spo2Val === null && redWin.length > 0 && blueWin.length > 0) {
          const dcR = redWin.reduce((a, b) => a + b, 0) / redWin.length - zlo;
          const dcB = blueWin.reduce((a, b) => a + b, 0) / blueWin.length - zlo;
          if (dcR > 0 && dcB > 0) {
            let sq = 0;
            for (const v of redFiltered) sq += v * v;
            const acR = Math.sqrt(sq / redFiltered.length);
            sq = 0;
            for (const v of blueFiltered) sq += v * v;
            const acB = Math.sqrt(sq / blueFiltered.length);
            const R = (acR / dcR) / (acB / dcB);
            spo2Val = Math.min(100, Math.max(88, 110 - 25 * R));
          }
        }
      }

      // BP and Glucose estimation from PPG morphological analysis
      const bpSysVal = analysis.bp?.systolic ?? (demoMode ? 110 + Math.round((Math.random() - 0.5) * 20) : null);
      const bpDiaVal = analysis.bp?.diastolic ?? (demoMode ? 70 + Math.round((Math.random() - 0.5) * 15) : null);
      const glucoseVal = analysis.glucose?.glucose ?? (demoMode ? 95 + Math.round((Math.random() - 0.5) * 20) : null);

      // Always update with new values when available, preserving last valid otherwise
      setBpm(hr);
      // SpO2/Hb are independent of HR validity: show them whenever the color
      // buffers are usable (previously they only updated when HR was valid, so
      // any HR failure also blanked SpO2/Hb).
      if (hbVal !== null) setHb(hbVal);
      else if (demoMode) setHb(13.5 + (Math.random() - 0.5) * 1.0);
      if (spo2Val !== null) setSpo2(Math.round(spo2Val));
      else if (demoMode) setSpo2(Math.round(97 + (Math.random() - 0.5) * 2));
      // Update BP and Glucose from PPG analysis
      setBp(bpSysVal !== null && bpDiaVal !== null ? `${bpSysVal}/${bpDiaVal}` : null);
      setSugar(glucoseVal);
      setCreatinine(analysis.creatinine);
      if (analysis.respRate !== null) {
        setRespRate(Math.round(analysis.respRate * 10) / 10);
      } else if (demoMode) {
        setRespRate(Math.round((15 + (Math.random() - 0.5) * 2) * 10) / 10);
      }
      if (analysis.hrv) {
        setHrv({ sdnn: analysis.hrv.sdnn, rmssd: analysis.hrv.rmssd, pnn50: analysis.hrv.pnn50 });
      } else if (demoMode) {
        const sdnn = baseHrvRef.current.sdnn + (Math.random() - 0.5) * 4;
        const rmssd = baseHrvRef.current.rmssd + (Math.random() - 0.5) * 4;
        const pnn50 = Math.max(0, baseHrvRef.current.pnn50 + (Math.random() - 0.5) * 4);
        setHrv({ sdnn, rmssd, pnn50 });
      }
      setRhythm(analysis.rhythm === "unknown" && demoMode ? "regular" : analysis.rhythm);

      if (isMeasuring && hr !== null) {
        samplesRef.current.push({
          bpm: hr,
          spo2: spo2Val !== null ? Math.round(spo2Val) : null,
          hb: hbVal !== null ? Math.round(hbVal * 10) / 10 : null,
          bpSys: bpSysVal,
          bpDia: bpDiaVal,
          sugar: glucoseVal,
          creatinine: analysis.creatinine.creatinine,
          signalQuality: sqi / 100,
          respRate: analysis.respRate ?? 15,
          sdnn: analysis.hrv?.sdnn ?? 0,
          rmssd: analysis.hrv?.rmssd ?? 0,
          pnn50: analysis.hrv?.pnn50 ?? 0,
          rhythm: analysis.rhythm === "unknown" ? "regular" : analysis.rhythm,
        });
      }
    }

    drawWaveform();
    if (cameraOnRef.current) {
      animFrame.current = requestAnimationFrame(processFrame);
    }
  }, [drawWaveform, cameraMode, demoMode, patient]);

  useEffect(() => {
    if (cameraOn) {
      animFrame.current = requestAnimationFrame(processFrame);
    } else {
      cancelAnimationFrame(animFrame.current);
      valuesRef.current = [];
      redRef.current = [];
      greenRef.current = [];
      blueRef.current = [];
      ppgBufferRef.current = [];
      syntheticBufferRef.current = [];
      synthTimeRef.current = 0;
      frameIntervalsRef.current = [];
      fingerHistoryRef.current = [];
      setHb(null);
      setBpm(null);
      setRespRate(null);
      setHrv(null);
      setRhythm("unknown");
      setSpo2(null);
      setBp(null);
      setSugar(null);
      setCreatinine(null);
      setFingerDetected(false);
      setSkinDetected(false);
      setFingerPressure(null);
      setPressureDetail(null);
    }
    return () => cancelAnimationFrame(animFrame.current);
  }, [cameraOn, processFrame]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startVideoRecording = useCallback((stream: MediaStream) => {
    videoChunksRef.current = [];
    try {
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
      mediaRecorderRef.current = recorder;
      recordingRef.current = true;
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          videoChunksRef.current.push(event.data);
        }
      };
      
      recorder.start(1000); // Collect data every second
    } catch (e) {
      console.warn("Video recording not supported:", e);
    }
  }, []);

  const stopVideoRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }

      const timeout = setTimeout(() => {
        resolve(null);
      }, 2000);

      recorder.onstop = () => {
        clearTimeout(timeout);
        const blob = new Blob(videoChunksRef.current, { type: "video/webm" });
        videoChunksRef.current = [];
        recordingRef.current = false;
        resolve(blob);
      };

      recorder.stop();
    });
  }, []);

  const finalizeSession = useCallback(async () => {
    const samples = samplesRef.current;
    // Average only over samples where the value was actually produced; gated
    // metrics (SpO2/Hb/glucose/creatinine under low SQI) are null, and
    // averaging them as 0 previously dragged the final reading toward zero.
    const avg = (key: keyof typeof samples[number]): number | null => {
      const vals: number[] = [];
      for (const s of samples) {
        const v = s[key];
        if (typeof v === "number" && Number.isFinite(v)) vals.push(v);
      }
      return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const median = (vals: number[]): number | null => {
      if (vals.length === 0) return null;
      const sorted = vals.slice().sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    const mode = <T extends string>(items: T[]): T => {
      const m = new Map<T, number>();
      for (const it of items) m.set(it, (m.get(it) ?? 0) + 1);
      let best: T = items[0] ?? ("unknown" as T);
      let bestN = 0;
      for (const [k, v] of m.entries()) {
        if (v > bestN) {
          best = k;
          bestN = v;
        }
      }
      return best;
    };

    let computedReading: { bpm: number; spo2: number | null; hb: number | null; bpSys: number | null; bpDia: number | null; sugar: number | null; creatinine: number | null; egfr: number | null; creatinineClass: string; creatinineConfidence: number; creatinineMethod: string; signalQuality: number; respRate: number; sdnn: number; rmssd: number; pnn50: number; rhythm: "regular" | "irregular" | "unknown" } | null = null;

    const ageYears = patient ? parseInt(patient.age, 10) || null : null;
    const patientSex = patient ? (patient.sex as "male" | "female") : null;
    const assignCreatinine = (cr: number | null) => {
      const rounded = cr !== null && cr > 0 ? Math.round(cr * 100) / 100 : null;
      const egfr = estimateEgfr(rounded, ageYears, patientSex);
      return {
        creatinine: rounded,
        egfr: egfr === null ? null : Math.round(egfr),
        creatinineClass: rounded === null ? "" : rounded > 1.3 ? "high" : rounded < 0.6 ? "low" : "normal",
      };
    };

    if (samples.length > 0) {
      const bpSysAvg = avg("bpSys");
      const bpDiaAvg = avg("bpDia");
      const cr = assignCreatinine(avg("creatinine"));
      computedReading = {
        bpm: avg("bpm") ?? 0,
        spo2: avg("spo2") !== null ? Math.round(avg("spo2") as number) : null,
        hb: avg("hb") !== null ? Math.round((avg("hb") as number) * 10) / 10 : null,
        bpSys: bpSysAvg !== null ? Math.round(bpSysAvg) : null,
        bpDia: bpDiaAvg !== null ? Math.round(bpDiaAvg) : null,
        sugar: avg("sugar") !== null ? Math.round(avg("sugar") as number) : null,
        creatinine: cr.creatinine,
        egfr: cr.egfr,
        creatinineClass: cr.creatinineClass,
        creatinineConfidence: creatinine?.confidence ?? 0,
        creatinineMethod: creatinine?.method ?? cameraMode,
        signalQuality: avg("signalQuality") ?? 0,
        respRate: avg("respRate") ?? 0,
        sdnn: avg("sdnn") ?? 0,
        rmssd: avg("rmssd") ?? 0,
        pnn50: avg("pnn50") ?? 0,
        rhythm: mode(samples.map((s) => (s.rhythm as "regular" | "irregular" | "unknown"))),
      };
    } else {
      const bpStr = bp;
      const bpParts = bpStr ? bpStr.split("/") : [];
      const bpSysVal = bpParts.length === 2 ? parseInt(bpParts[0], 10) : 0;
      const bpDiaVal = bpParts.length === 2 ? parseInt(bpParts[1], 10) : 0;
      const cr = assignCreatinine(creatinine?.creatinine ?? null);
      computedReading = {
        bpm: bpm ?? 0,
        spo2,
        hb,
        bpSys: bpSysVal > 0 ? bpSysVal : null,
        bpDia: bpDiaVal > 0 ? bpDiaVal : null,
        sugar,
        creatinine: cr.creatinine,
        egfr: cr.egfr,
        creatinineClass: cr.creatinineClass,
        creatinineConfidence: creatinine?.confidence ?? 0,
        creatinineMethod: creatinine?.method ?? cameraMode,
        signalQuality: signalQuality ?? 0,
        respRate: respRate ?? 0,
        sdnn: hrv?.sdnn ?? 0,
        rmssd: hrv?.rmssd ?? 0,
        pnn50: hrv?.pnn50 ?? 0,
        rhythm: (rhythm as "regular" | "irregular" | "unknown"),
      };
    }

    stopStream();
    setCameraOn(false);
    setPhase("finalized");
    setCountdown(0);
    setFinalReading(computedReading);

    const videoBlob = await stopVideoRecording().catch(() => null);
    if (videoBlob && patient && computedReading) {
      // Storage keeps numbers (0 = not measurable this session); the on-screen
      // final reading keeps nulls so gated metrics render as "—".
      const storedVitals = {
        bpm: computedReading.bpm,
        spo2: computedReading.spo2 ?? 0,
        hb: computedReading.hb ?? 0,
        bpSys: computedReading.bpSys ?? 0,
        bpDia: computedReading.bpDia ?? 0,
        sugar: computedReading.sugar ?? 0,
        signalQuality: computedReading.signalQuality,
        respRate: computedReading.respRate,
        sdnn: computedReading.sdnn,
        rmssd: computedReading.rmssd,
        pnn50: computedReading.pnn50,
        creatinine: computedReading.creatinine ?? 0,
        egfr: computedReading.egfr ?? 0,
        rhythm: computedReading.rhythm,
      };
      await addVitalReading({
        patient: {
          name: patient.name,
          age: patient.age,
          sex: patient.sex as "male" | "female",
          lmp: patient.lmp,
          pregnancy: patient.pregnancy as "pregnant" | "non-pregnant" | undefined,
          menopause: patient.menopause as "yes" | "no" | undefined,
        },
        vitals: storedVitals,
        cameraMode,
        demoMode,
      }, videoBlob);
      await loadReadings();
    }
  }, [bpm, hb, spo2, bp, sugar, creatinine, signalQuality, respRate, hrv, rhythm, stopStream, stopVideoRecording, patient, cameraMode, demoMode, loadReadings]);

  const startAutoCountdown = useCallback(() => {
    sessionStartRef.current = Date.now();
    samplesRef.current = [];
    videoChunksRef.current = [];
    // Reset every signal buffer so a new session never inherits samples from a
    // previous one (previously the color/PPG buffers persisted across sessions).
    valuesRef.current = [];
    redRef.current = [];
    greenRef.current = [];
    blueRef.current = [];
    ppgBufferRef.current = [];
    syntheticBufferRef.current = [];
    synthTimeRef.current = 0;
    frameIntervalsRef.current = [];
    fingerHistoryRef.current = [];
    lastBpmUpdate.current = 0;
    setFinalReading(null);
    setPhase("warming");
    setCountdown(130);
    if (countdownTimerRef.current) window.clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = window.setInterval(() => {
      const elapsed = Date.now() - sessionStartRef.current;
      const remaining = Math.max(0, 130 - Math.floor(elapsed / 1000));
      setCountdown(remaining);
      if (elapsed >= 10000 && phaseRef.current === "warming") {
        setPhase("measuring");
      }
      if (remaining <= 0) {
        if (countdownTimerRef.current) window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
        finalizeSession();
      }
    }, 250);
  }, [finalizeSession]);

  const toggleCamera = useCallback(async () => {
    if (cameraOn) {
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      if (autoStopTimerRef.current) {
        window.clearTimeout(autoStopTimerRef.current);
        autoStopTimerRef.current = null;
      }
      
      await stopVideoRecording();
      stopStream();
      setCameraOn(false);
      setError(null);
      setPhase("idle");
      setCountdown(0);
      return;
    }
    setLoading(true);
    setError(null);
    if (demoMode) {
      syntheticBufferRef.current = [];
      setCameraOn(true);
      setError(null);
      startAutoCountdown();
      setLoading(false);
      return;
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is not supported in this browser.");
      setLoading(false);
      setCameraOn(false);
      return;
    }
    try {
      let stream: MediaStream;
      const videoConstraints: MediaTrackConstraints = {
        width: { ideal: 640 },
        height: { ideal: 480 }
      };

      if (cameraMode === "rear") {
        videoConstraints.facingMode = "environment";
      } else {
        videoConstraints.facingMode = "user";
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack && cameraMode === "rear") {
        try {
          const caps = videoTrack.getCapabilities?.() as { torch?: unknown } | undefined;
          if (caps?.torch) {
            await videoTrack.applyConstraints({ torch: torchOn } as any);
          }
        } catch {
          // Torch may not be supported on all devices
        }
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      
      startVideoRecording(stream);
      
      setCameraOn(true);
      setError(null);
      startAutoCountdown();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
      setCameraOn(false);
    } finally {
      setLoading(false);
    }
  }, [cameraOn, stopStream, startAutoCountdown, cameraMode, torchOn, demoMode, stopVideoRecording, startVideoRecording]);

  toggleCameraRef.current = toggleCamera;

  const switchCameraMode = useCallback(async (mode: "front" | "rear") => {
    const wasOn = cameraOnRef.current;
    // Reset finger history when switching camera modes
    if (fingerHistoryRef.current) fingerHistoryRef.current = [];
    setCameraMode(mode);
    if (mode === "rear") {
      setTorchOn(true);
    } else {
      setTorchOn(false);
    }
    if (wasOn) {
      stopStream();
      setCameraOn(false);
      setError(null);
      setPhase("idle");
      setCountdown(0);
      await new Promise((r) => setTimeout(r, 100));
    }
    toggleCameraRef.current?.();
  }, [stopStream]);

  const submitPatientInfo = useCallback(() => {
    if (!form.name.trim()) {
      setFormError("Please enter the patient's name.");
      return;
    }
    const ageNum = Number(form.age);
    if (!form.age.trim() || Number.isNaN(ageNum) || ageNum <= 0 || ageNum > 120) {
      setFormError("Please enter a valid age (1-120).");
      return;
    }
    if (form.sex !== "male" && form.sex !== "female") {
      setFormError("Please select sex.");
      return;
    }
    if (form.sex === "female") {
      if (form.menopause !== "yes" && form.menopause !== "no") {
        setFormError("Please indicate menopause status.");
        return;
      }
      if (form.menopause === "no") {
        if (!form.lmp) {
          setFormError("Please enter the date of last menstrual period (LMP).");
          return;
        }
        if (form.pregnancy !== "pregnant" && form.pregnancy !== "non-pregnant") {
          setFormError("Please indicate pregnancy status.");
          return;
        }
      }
    }
    setFormError(null);
    setPatient({
      name: form.name.trim(),
      age: form.age.trim(),
      sex: form.sex,
      lmp: form.lmp,
      pregnancy: form.pregnancy,
      menopause: form.menopause,
    });
  }, [form]);

  if (!patient) {
    return (
      <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
        <h1 className="text-4xl font-extrabold mb-2 tracking-tight text-center">PPG Vital Scan</h1>
        <p className="text-base text-neutral-400 mb-6 max-w-lg text-center">
          Enter patient details to begin a 130-second measurement session.
        </p>

        <div className="w-full max-w-lg bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 p-6">
          <h2 className="text-lg font-semibold text-neutral-100 mb-4">Patient Information</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">Name</label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Full name"
                className="w-full px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-100 placeholder-neutral-500 focus:outline-none focus:border-red-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1">Age</label>
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={form.age}
                  onChange={(e) => setForm({ ...form, age: e.target.value })}
                  placeholder="Years"
                  className="w-full px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-100 placeholder-neutral-500 focus:outline-none focus:border-red-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1">Sex</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setForm({ ...form, sex: "female", menopause: "", lmp: "", pregnancy: "" })}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                      form.sex === "female"
                        ? "bg-red-600 border-red-500 text-white"
                        : "bg-neutral-800 border-neutral-700 text-neutral-300 hover:border-neutral-600"
                    }`}
                  >
                    Female
                  </button>
                  <button
                    type="button"
                    onClick={() => setForm({ ...form, sex: "male", menopause: "", lmp: "", pregnancy: "" })}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                      form.sex === "male"
                        ? "bg-red-600 border-red-500 text-white"
                        : "bg-neutral-800 border border-neutral-700 text-neutral-300 hover:border-neutral-600"
                    }`}
                  >
                    Male
                  </button>
                </div>
              </div>
            </div>

            {form.sex === "female" && (
              <>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1">Menopause Status</label>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setForm({ ...form, menopause: "yes", lmp: "", pregnancy: "" })}
                      className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                        form.menopause === "yes"
                          ? "bg-red-600 border-red-500 text-white"
                          : "bg-neutral-800 border-neutral-700 text-neutral-300 hover:border-neutral-600"
                      }`}
                    >
                      Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => setForm({ ...form, menopause: "no", lmp: "", pregnancy: "" })}
                      className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                        form.menopause === "no"
                          ? "bg-red-600 border-red-500 text-white"
                          : "bg-neutral-800 border-neutral-700 text-neutral-300 hover:border-neutral-600"
                      }`}
                    >
                      No
                    </button>
                  </div>
                </div>

                {form.menopause === "no" && (
                  <>
                    <div>
                      <label className="block text-xs font-medium text-neutral-400 mb-1">Date of LMP (Last Menstrual Period)</label>
                      <input
                        type="date"
                        value={form.lmp}
                        onChange={(e) => setForm({ ...form, lmp: e.target.value })}
                        className="w-full px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-100 focus:outline-none focus:border-red-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-neutral-400 mb-1">Pregnancy Status</label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setForm({ ...form, pregnancy: "pregnant" })}
                          className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                            form.pregnancy === "pregnant"
                              ? "bg-red-600 border-red-500 text-white"
                              : "bg-neutral-800 border-neutral-700 text-neutral-300 hover:border-neutral-600"
                          }`}
                        >
                          Pregnant
                        </button>
                        <button
                          type="button"
                          onClick={() => setForm({ ...form, pregnancy: "non-pregnant" })}
                          className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition ${
                            form.pregnancy === "non-pregnant"
                              ? "bg-red-600 border-red-500 text-white"
                              : "bg-neutral-800 border-neutral-700 text-neutral-300 hover:border-neutral-600"
                          }`}
                        >
                          Non-pregnant
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </>
            )}

            {formError && (
              <div className="p-3 rounded-lg bg-red-950/60 border border-red-900/40">
                <p className="text-xs text-red-300 font-medium">{formError}</p>
              </div>
            )}

            <button
              type="button"
              onClick={submitPatientInfo}
              className="w-full mt-2 px-5 py-2.5 rounded-full text-sm font-semibold bg-red-600 hover:bg-red-700 active:bg-red-800 transition shadow-lg shadow-red-900/30"
            >
              Launch Vital Scan
            </button>
          </div>
        </div>

        <Link href="/" className="mt-6 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition">
          Home
        </Link>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
      <h1 className="text-4xl font-extrabold mb-4 tracking-tight text-center">Vital Scan</h1>
        <div className="mb-4 px-3 py-2 rounded-full bg-neutral-900 border border-neutral-800 text-xs text-neutral-300">
          <span className="text-neutral-500 mr-2">Patient:</span>
          <span className="font-semibold text-neutral-100">{patient.name}</span>
          <span className="text-neutral-500"> • {patient.age}y • {patient.sex === "female" ? "Female" : "Male"}</span>
          {patient.sex === "female" && (
            <span className="text-neutral-500">
              {" • "}{patient.menopause === "yes" ? "Menopause" : patient.pregnancy === "pregnant" ? "Pregnant" : "Non-pregnant"}
              {patient.menopause === "no" && patient.lmp && ` • LMP ${patient.lmp}`}
            </span>
          )}
        </div>
      <p className="text-base text-neutral-400 mb-6 max-w-lg text-center">
        Front camera: position your face in good lighting. Rear camera + flash: cover the lens with your fingertip.
      </p>
      {phase === "finalized" && finalReading && (
        <div className="w-full max-w-lg mb-6 p-4 rounded-xl border-2 border-emerald-500/60 bg-emerald-950/30 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-bold text-emerald-300">Measurement Complete</h2>
            <span className="text-xs text-emerald-200/80">130s session • 120s recorded after 10s warm-up</span>
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-center">
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Heart Rate</p>
              <p className="text-2xl font-bold text-red-400">{finalReading.bpm.toFixed(1)} BPM</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">SpO₂</p>
              <p className="text-2xl font-bold text-cyan-400">{finalReading.spo2 !== null ? `${finalReading.spo2}%` : "—"}</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Resp Rate</p>
              <p className="text-xl font-bold text-sky-400">{finalReading.respRate > 0 ? finalReading.respRate.toFixed(1) : "—"} /min</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">HRV SDNN</p>
              <p className="text-xl font-bold text-violet-400">{finalReading.sdnn > 0 ? finalReading.sdnn.toFixed(1) : "—"} ms</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Hb</p>
              <p className="text-xl font-bold text-rose-400">{finalReading.hb !== null ? `${finalReading.hb.toFixed(1)} g/dL` : "—"}</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Creatinine</p>
              <p className="text-xl font-bold text-teal-400">{finalReading.creatinine !== null ? `${finalReading.creatinine.toFixed(2)} mg/dL` : "—"}</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">eGFR</p>
              <p className="text-xl font-bold text-teal-300">{finalReading.egfr !== null ? finalReading.egfr : "—"}</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Signal Quality</p>
              <p className="text-xl font-bold text-violet-400">{(finalReading.signalQuality * 100).toFixed(1)}%</p>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2 justify-center text-xs">
            <span className={`px-2 py-1 rounded-full border ${
              finalReading.rhythm === "regular" ? "text-emerald-400 border-emerald-500 bg-emerald-500/10" :
              finalReading.rhythm === "irregular" ? "text-amber-400 border-amber-500 bg-amber-500/10" :
              "text-neutral-500 border-neutral-500 bg-neutral-500/10"
            }`}>
              Rhythm: {finalReading.rhythm}
            </span>
            <span className="px-2 py-1 rounded-full border text-violet-400 border-violet-500 bg-violet-500/10">
              RMSSD: {finalReading.rmssd > 0 ? finalReading.rmssd.toFixed(1) : "—"} ms
            </span>
            <span className="px-2 py-1 rounded-full border text-rose-400 border-rose-500 bg-rose-500/10">
              Hb: {finalReading.hb !== null ? `${finalReading.hb.toFixed(1)} g/dL` : "—"}
            </span>
            <span className={`px-2 py-1 rounded-full border ${
              finalReading.creatinineClass === "high" ? "text-red-400 border-red-500 bg-red-500/10" :
              finalReading.creatinineClass === "low" ? "text-sky-400 border-sky-500 bg-sky-500/10" :
              "text-teal-400 border-teal-500 bg-teal-500/10"
            }`}>
              Creatinine: {finalReading.creatinine !== null ? `${finalReading.creatinine.toFixed(2)} mg/dL` : "—"} ({finalReading.creatinineClass || "n/a"})
            </span>
            <span className="px-2 py-1 rounded-full border text-teal-300 border-teal-500 bg-teal-500/10">
              eGFR: {finalReading.egfr !== null ? finalReading.egfr : "—"} mL/min/1.73m²
            </span>
            <span className="px-2 py-1 rounded-full border text-violet-400 border-violet-500 bg-violet-500/10">
              Signal: {(finalReading.signalQuality * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}
      <div className="w-full max-w-lg bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 overflow-hidden">
        <div className="relative h-64 bg-black">
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline autoPlay />
          {cameraOn && cameraMode === "rear" && (
            <div className="absolute top-2 right-2 flex flex-col gap-1 items-end">
              <div className={`text-xs px-2 py-1 rounded ${fingerDetected ? "bg-emerald-700/80 text-emerald-100" : "bg-amber-700/80 text-amber-100"}`}>
                {fingerDetected ? "Finger detected" : "Place fingertip over lens"}
              </div>
              {fingerDetected && fingerPressure && (
                <div className={`text-xs px-2 py-1 rounded font-medium ${
                  fingerPressure === "adequate" ? "bg-emerald-700/90 text-emerald-50" :
                  fingerPressure === "low" ? "bg-amber-700/90 text-amber-50" :
                  "bg-red-700/90 text-red-50"
                } flex items-center gap-1`}>
                  {pressureDetail === "ADEQUATE" && <span className="text-emerald-300">✓</span>}
                  {pressureDetail === "UNSTABLE" && <span className="text-sky-300">↻</span>}
                  {fingerPressure === "low" && <span className="text-amber-300">⚠</span>}
                  {fingerPressure === "excessive" && <span className="text-red-300">⚠</span>}
                  <span>
                    {pressureDetail === "TOO_HEAVY" ? "Press lighter — occlusion detected" :
                     pressureDetail === "TOO_LIGHT" ? "Press firmer — light leak" :
                     pressureDetail === "UNSTABLE" ? "Hold still — motion detected" :
                     pressureDetail === "WEAK_SIGNAL" ? "Adjust gently, keep still" :
                     pressureDetail === "ADEQUATE" ? "Pressure OK" :
                     "Adjust pressure"}
                  </span>
                </div>
              )}
            </div>
          )}
          {cameraOn && cameraMode === "front" && !demoMode && (
            <div className={`absolute top-2 right-2 ${skinDetected ? "bg-emerald-700/80 text-emerald-100" : "bg-amber-700/80 text-amber-100"} text-xs px-2 py-1 rounded`}>
              {skinDetected ? "Skin ROI locked" : "Looking for skin..."}
            </div>
          )}
          {!cameraOn && <div className="absolute inset-0 flex items-center justify-center"><span className="text-neutral-500 text-sm">Camera off</span></div>}
                    {cameraOn && (
            <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
              {demoMode ? "Demo: synthetic PPG" : `Signal: ${((signalQuality ?? 0) * 100).toFixed(1)}%`}
            </div>
          )}
          {demoMode && cameraOn && (
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-rose-900/40 via-rose-700/30 to-amber-700/30">
              <div className="text-center">
                <p className="text-rose-100 text-sm font-semibold">Demo Mode</p>
                <p className="text-rose-200/80 text-xs">Synthetic PPG waveform</p>
              </div>
            </div>
          )}
        </div>
        <div className="p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex-1 min-w-0">
              <span className="block text-sm text-neutral-300 truncate">
                {phase === "warming" && countdown > 0
                  ? `Warming up — discarding first 10s (${countdown}s total remaining)`
: phase === "measuring"
                   ? `Measuring — recording next 120s (${countdown}s remaining)`
                  : phase === "finalized"
                  ? "Measurement complete — final reading below"
                  : loading
                  ? "Starting camera..."
                  : cameraOn
                  ? "Camera active"
                  : "Ready"}
              </span>
              {error && (
                <div className="mt-2 p-3 rounded-lg bg-red-950/60 border border-red-900/40">
                  <p className="text-xs text-red-300 font-medium">{error}</p>
                  {(error.includes("denied") || permission === "denied") && (
                    <p className="text-xs text-red-200/80 mt-2">Tap the lock / site-info icon in your address bar, choose &quot;Camera&quot;, set to &quot;Allow&quot;, then refresh.</p>
                  )}
                </div>
              )}
            </div>
            <button onClick={toggleCamera} disabled={loading} className="px-5 py-2.5 rounded-full text-sm font-semibold bg-red-600 hover:bg-red-700 active:bg-red-800 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg shadow-red-900/30 whitespace-nowrap">
              {loading ? "Starting..." : cameraOn ? "Stop Camera" : "Start Camera"}
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => switchCameraMode("front")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition border ${
                cameraMode === "front"
                  ? "bg-neutral-700 border-neutral-500 text-neutral-100"
                  : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500"
              }`}
            >
              Front Camera
            </button>
            <button
              onClick={() => switchCameraMode("rear")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition border ${
                cameraMode === "rear"
                  ? "bg-neutral-700 border-neutral-500 text-neutral-100"
                  : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500"
              }`}
            >
              Rear Camera
            </button>
            {cameraMode === "rear" && (
              <button
                 onClick={async () => {
                   const next = !torchOn;
                   setTorchOn(next);
                   const track = streamRef.current?.getVideoTracks()[0];
                   if (track) {
                     try {
                       const caps = track.getCapabilities?.() as { torch?: unknown } | undefined;
                       if (caps?.torch) {
                         await track.applyConstraints({ torch: next } as any);
                       } else {
                         setError("This device/browser does not expose a torch/flash API for the selected track.");
                       }
                     } catch (e) {
                       setError(getErrorMessage(e));
                     }
                   }
                 }}
                className={`px-3 py-2 rounded-lg text-xs font-semibold transition border ${
                  torchOn
                    ? "bg-amber-600 border-amber-500 text-neutral-100"
                    : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500"
                }`}
              >
                {torchOn ? "Flash On" : "Flash Off"}
              </button>
            )}
            <button
              onClick={() => setDemoMode((d) => !d)}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition border ${
                demoMode
                  ? "bg-rose-600 border-rose-500 text-neutral-100"
                  : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:border-neutral-500"
              }`}
            >
              {demoMode ? "Demo: On" : "Demo: Off"}
            </button>
          </div>
        </div>
        <div className="px-4 pb-4">
          <div className="relative h-32 bg-neutral-950 rounded-lg border border-neutral-800 overflow-hidden">
            <canvas ref={canvasRef} className="w-full h-full" />
            {!cameraOn && <div className="absolute inset-0 flex items-center justify-center"><span className="text-neutral-600 text-xs">Start camera to see waveform</span></div>}
            {bpm !== null && <div className="absolute top-2 right-2 bg-red-900/80 text-red-200 text-xs font-bold px-2 py-0.5 rounded">{bpm.toFixed(0)} BPM</div>}
          </div>
        </div>
        {(bpm !== null || cameraOn) && (
          <div className="px-4 pb-4 grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Heart Rate</p>
              <p className="text-2xl font-bold text-red-400">{bpm !== null ? `${bpm.toFixed(0)} BPM` : "--"}</p>
              <p className={`text-[10px] mt-1 ${rhythm === "regular" ? "text-emerald-400" : rhythm === "irregular" ? "text-amber-400" : "text-neutral-500"}`}>
                Rhythm: {rhythm}
              </p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Resp Rate</p>
              <p className="text-2xl font-bold text-sky-400">{respRate !== null ? `${respRate.toFixed(1)} /min` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">breaths/min</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">HRV (SDNN)</p>
              <p className="text-2xl font-bold text-violet-400">{hrv ? `${hrv.sdnn.toFixed(1)} ms` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">RMSSD {hrv ? hrv.rmssd.toFixed(1) : "--"}</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">SpO₂</p>
              <p className="text-2xl font-bold text-cyan-400">{spo2 !== null ? `${spo2}%` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">estimate only</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Hb</p>
              <p className="text-base font-bold text-rose-400">{hb !== null ? `${hb.toFixed(1)} g/dL` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">estimate only</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">BP</p>
              <p className="text-2xl font-bold text-emerald-400">{bp !== null ? bp : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">estimate only</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Glucose</p>
              <p className="text-2xl font-bold text-amber-400">{sugar !== null ? `${sugar} mg/dL` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">not measurable from RGB alone</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Creatinine</p>
              <p className="text-2xl font-bold text-teal-400">{creatinine?.creatinine != null ? `${creatinine.creatinine.toFixed(2)} mg/dL` : "--"}</p>
              <p className="text-[10px] mt-1 text-neutral-500">
                {creatinine?.creatinineClass ? `Class: ${creatinine.creatinineClass} • ${cameraMode === "front" ? "rPPG" : "PPG"}` : "estimate only"}
              </p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Signal Quality</p>
              <p className="text-2xl font-bold text-violet-400">{signalQuality !== null ? `${(signalQuality * 100).toFixed(1)}%` : "--"}</p>
            </div>
          </div>
        )}
        {phase === "finalized" && finalReading && (
          <div className="px-4 pb-4">
            <div className="rounded-xl border-2 border-emerald-500/60 bg-emerald-950/30 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-base font-bold text-emerald-300">Final Reading (130s analysis)</h2>
                <span className="text-[10px] text-emerald-200/80">120s recording after 10s warm-up</span>
              </div>
              <div className="mb-4 p-3 rounded-lg bg-neutral-900/80 border border-emerald-700/40 text-xs">
                <p className="text-neutral-400 mb-1 font-semibold">Patient</p>
                <p className="text-neutral-100">
                  {patient.name} • {patient.age} yrs • {patient.sex === "female" ? "Female" : "Male"}
                </p>
                {patient.sex === "female" && (
                  <p className="text-neutral-300 mt-1">
                    LMP: {patient.lmp || "—"} • {patient.pregnancy === "pregnant" ? "Pregnant" : "Non-pregnant"}
                  </p>
                )}
              </div>
              <div className="mb-4 p-3 rounded-lg bg-neutral-900/80 border border-emerald-700/40 text-xs">
                <p className="text-neutral-400 mb-2 font-semibold">Cardiac & Respiratory (PPG-derived)</p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <p className="text-neutral-500 text-[10px]">Heart Rate</p>
                    <p className="text-xl font-bold text-red-400">{finalReading.bpm.toFixed(1)} BPM</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-[10px]">Rhythm</p>
                    <p className={`text-base font-bold ${finalReading.rhythm === "regular" ? "text-emerald-400" : finalReading.rhythm === "irregular" ? "text-amber-400" : "text-neutral-400"}`}>
                      {finalReading.rhythm}
                    </p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-[10px]">Resp Rate</p>
                    <p className="text-xl font-bold text-sky-400">{finalReading.respRate > 0 ? finalReading.respRate.toFixed(1) : "—"} /min</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-[10px]">HRV (SDNN)</p>
                    <p className="text-xl font-bold text-violet-400">{finalReading.sdnn > 0 ? finalReading.sdnn.toFixed(1) : "—"} ms</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-[10px]">RMSSD</p>
                    <p className="text-base font-bold text-violet-300">{finalReading.rmssd > 0 ? finalReading.rmssd.toFixed(1) : "—"} ms</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-[10px]">pNN50</p>
                    <p className="text-base font-bold text-violet-300">{finalReading.pnn50 > 0 ? finalReading.pnn50.toFixed(1) : "—"}%</p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">BPM</p>
                  <p className="text-2xl font-bold text-red-400">{finalReading.bpm.toFixed(1)}</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">SpO₂</p>
                  <p className="text-2xl font-bold text-cyan-400">{finalReading.spo2 !== null ? `${finalReading.spo2}%` : "--"}</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Hb</p>
                  <p className="text-2xl font-bold text-rose-400">{finalReading.hb !== null ? `${finalReading.hb.toFixed(1)} g/dL` : "--"}</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">BP</p>
                  <p className="text-xl font-bold text-emerald-400">{finalReading.bpSys !== null && finalReading.bpDia !== null ? `${finalReading.bpSys}/${finalReading.bpDia}` : "--"}</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Glucose</p>
                  <p className="text-xl font-bold text-amber-400">{finalReading.sugar !== null ? `${finalReading.sugar} mg/dL` : "--"}</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-teal-700/40">
                  <p className="text-neutral-400 mb-1">Serum Creatinine</p>
                  <p className="text-xl font-bold text-teal-400">{finalReading.creatinine !== null ? `${finalReading.creatinine.toFixed(2)} mg/dL` : "--"}</p>
                  <p className={`text-[10px] mt-1 font-semibold ${
                    finalReading.creatinineClass === "high" ? "text-red-400" :
                    finalReading.creatinineClass === "low" ? "text-sky-400" :
                    finalReading.creatinineClass === "normal" ? "text-emerald-400" : "text-neutral-500"
                  }`}>
                    {finalReading.creatinineClass === "high" ? "High (>1.3)" : finalReading.creatinineClass === "low" ? "Low (<0.6)" : finalReading.creatinineClass === "normal" ? "Normal (0.6–1.3)" : "Signal quality too low"}
                  </p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-teal-700/40">
                  <p className="text-neutral-400 mb-1">eGFR (CKD-EPI 2021)</p>
                  <p className="text-xl font-bold text-teal-300">{finalReading.egfr !== null ? `${finalReading.egfr}` : "--"}</p>
                  <p className="text-[10px] mt-1 text-neutral-500">mL/min/1.73m²</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Signal</p>
                  <p className="text-xl font-bold text-violet-400">{(finalReading.signalQuality * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-3 mb-2 p-3 rounded-lg bg-neutral-900/80 border border-teal-700/40 text-[11px] text-neutral-400">
                <p className="text-teal-300 font-semibold mb-1">Creatinine method</p>
                <p>
                  {finalReading.creatinineMethod === "rppg-front" ? "Front camera rPPG (facial)" : "Rear camera PPG (fingertip)"} • 45-feature PPG morphology ensemble
                  {" • "}confidence {(finalReading.creatinineConfidence * 100).toFixed(0)}%
                </p>
                <p className="mt-1 text-neutral-500">
                  Research estimate based on Sridevi et al. (Smart Health 2025; Healthcare Analytics 2025). Not a substitute for a blood test.
                </p>
              </div>
              <div className="mt-4 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={async () => {
                    setExporting(true);
                    try {
                      const data = await getVitalReadings({ limit: 1000 });
                      await downloadReadings(data, "csv");
                      await loadReadings();
                    } finally {
                      setExporting(false);
                    }
                  }}
                  disabled={exporting || readingsLoading}
                  className="flex-1 px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 transition"
                >
                  {exporting ? "Exporting..." : "Export CSV & Clear"}
                </button>
                <button
                  onClick={async () => {
                    setExporting(true);
                    try {
                      const data = await getVitalReadings({ limit: 1000 });
                      await downloadReadings(data, "excel");
                      await loadReadings();
                    } finally {
                      setExporting(false);
                    }
                  }}
                  disabled={exporting || readingsLoading}
                  className="flex-1 px-4 py-2 rounded-lg text-sm font-semibold bg-sky-600 hover:bg-sky-700 disabled:opacity-50 transition"
                >
                  {exporting ? "Exporting..." : "Export Excel & Clear"}
                </button>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className="flex-1 px-4 py-2 rounded-lg text-sm font-semibold bg-neutral-700 hover:bg-neutral-600 transition"
                >
                  {showHistory ? "Hide History" : `View History (${readingCount}/${getMaxReadings()})`}
                </button>
              </div>
              <button
                onClick={() => {
                  setPhase("idle");
                  setFinalReading(null);
                  setCountdown(0);
                  setSpo2(null);
                  setHb(null);
                  setBpm(null);
                  setBp(null);
                  setSugar(null);
                  setCreatinine(null);
                  setSignalQuality(null);
                  setRespRate(null);
                  setHrv(null);
                  setRhythm("unknown");
                  setPatient(null);
                  setForm({ name: "", age: "", sex: "", menopause: "", lmp: "", pregnancy: "" });
                  startedRef.current = false;
                }}
                className="mt-4 w-full px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 transition"
              >
                 Start New Vital Scan
              </button>
            </div>
          </div>
        )}
        {showHistory && (
          <div className="px-4 pb-4">
            <div className="rounded-xl border border-neutral-700 bg-neutral-900 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-base font-bold text-neutral-100">Stored Readings</h2>
                <span className="text-[10px] text-neutral-400">{readingCount}/{getMaxReadings()} readings</span>
              </div>
              <div className="max-h-64 overflow-y-auto space-y-2">
                {readings.length > 0 ? (
                  readings.map((reading: VitalReading) => (
                    <div key={reading.id} className="bg-neutral-800 rounded-lg p-3 border border-neutral-700 text-xs">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-neutral-100 font-medium">{reading.patient.name}</p>
                          <p className="text-neutral-400">
                            {new Date(reading.timestamp).toLocaleString()} • {reading.vitals.bpm.toFixed(0)} BPM • SpO₂ {reading.vitals.spo2}% • {reading.vitals.rhythm}
                            {reading.vitals.creatinine ? ` • Cr ${reading.vitals.creatinine.toFixed(2)} mg/dL` : ""}
                          </p>
                        </div>
                        <button
                          onClick={async () => {
                            await deleteVitalReading(reading.id);
                            await loadReadings();
                          }}
                          className="px-2 py-1 text-[10px] text-neutral-400 hover:text-red-400 transition"
                        >
                          Delete
                        </button>
                      </div>
                      {reading.videoData && (
                        <div className="mt-2">
                          <video src={reading.videoData} controls className="w-full max-h-32 rounded bg-black" />
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <p className="text-neutral-500 text-center py-4">No readings stored yet</p>
                )}
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  onClick={async () => {
                    setExporting(true);
                    try {
                      const data = await getVitalReadings({ limit: 1000 });
                      await downloadReadings(data, "csv");
                      await loadReadings();
                    } finally {
                      setExporting(false);
                    }
                  }}
                  disabled={readings.length === 0 || exporting || readingsLoading}
                  className="flex-1 px-3 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 transition"
                >
                  Export All as CSV & Clear
                </button>
                <button
                  onClick={async () => {
                    setExporting(true);
                    try {
                      const data = await getVitalReadings({ limit: 1000 });
                      await downloadReadings(data, "excel");
                      await loadReadings();
                    } finally {
                      setExporting(false);
                    }
                  }}
                  disabled={readings.length === 0 || exporting || readingsLoading}
                  className="flex-1 px-3 py-2 rounded-lg text-sm font-semibold bg-sky-600 hover:bg-sky-700 disabled:opacity-50 transition"
                >
                  Export All as Excel & Clear
                </button>
              </div>
            </div>
          </div>
        )}
        {(cameraOn || phase === "warming" || phase === "measuring") && (
          <div className="px-4 pb-3">
            <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden border border-neutral-700">
              <div
                className="h-full bg-red-600 transition-all duration-300"
                style={{
                  width: phase === "finalized" ? "100%" : phase === "measuring" ? `${Math.min(100, ((130 - countdown) / 130) * 100)}%` : phase === "warming" ? `${Math.min(100, ((130 - countdown) / 130) * 100)}%` : "0%",
                }}
              />
            </div>
            <p className="mt-1 text-[10px] text-neutral-500 text-center">
              {phase === "warming" && `Warm-up — discarding first 10s (${countdown}s)`}
              {phase === "measuring" && `Measuring — recording next 120s (${countdown}s left)`}
              {phase === "finalized" && "Done"}
            </p>
          </div>
        )}
        <div className="px-4 pb-3">
          <p className="text-[10px] text-neutral-500 leading-tight">Research estimates. Not clinical measurements.</p>
        </div>
      </div>

      <div className="mt-6 flex gap-4">
        <Link href="/training" className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition">
          View Training Dashboard
        </Link>
        <Link href="/" className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition">
          Home
        </Link>
      </div>

      <p className="mt-6 text-xs text-neutral-500">Research-grade PPG signal processing</p>
    </main>
  );
}
