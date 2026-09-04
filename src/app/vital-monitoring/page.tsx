import { useRef, useState, useCallback, useEffect } from "react";

import Link from 'next/link';
import { analyzeVitals, type VitalAnalysis, butterworthBandpass } from './ppg-analysis';
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
function generateSyntheticPPG(seconds: number, fs: number, bpm: number): number[] {
  const n = Math.floor(seconds * fs);
  const out: number[] = new Array(n);
  const heartHz = bpm / 60;
  const respHz = 0.25;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
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
  const [spo2Offset, setSpo2Offset] = useState(0);
  const [bpm, setBpm] = useState<number | null>(null);
  const [spo2, setSpo2] = useState<number | null>(null);
  const [bp, setBp] = useState<string | null>(null);
  const [sugar, setSugar] = useState<number | null>(null);
  const [hb, setHb] = useState<number | null>(null);
  const [signalQuality, setSignalQuality] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "warming" | "measuring" | "finalized">("idle");
  const [countdown, setCountdown] = useState<number>(0);
  const [finalReading, setFinalReading] = useState<{
    bpm: number;
    spo2: number;
    hb: number;
    bpSys: number;
    bpDia: number;
    sugar: number;
    signalQuality: number;
    respRate: number;
    sdnn: number;
    rmssd: number;
    pnn50: number;
    rhythm: string;
  } | null>(null);
  const [respRate, setRespRate] = useState<number | null>(null);
  const [hrv, setHrv] = useState<{ sdnn: number; rmssd: number; pnn50: number } | null>(null);
  const [rhythm, setRhythm] = useState<"regular" | "irregular" | "unknown">("unknown");
  const [fingerDetected, setFingerDetected] = useState(false);
  const valuesRef = useRef<number[]>([]);
  const redRef = useRef<number[]>([]);
  const greenRef = useRef<number[]>([]);
  const blueRef = useRef<number[]>([]);
  const ppgBufferRef = useRef<number[]>([]);
  const syntheticBufferRef = useRef<number[]>([]);
  const baseHrvRef = useRef<{ sdnn: number; rmssd: number; pnn50: number }>({ sdnn: 45, rmssd: 38, pnn50: 12 });
  const fpsRef = useRef<number>(30);
  const lastFrameTimeRef = useRef<number>(0);
  const animFrame = useRef<number>(0);
  const lastBpmUpdate = useRef<number>(0);
  const roiDataRef = useRef<{ roi: number[][]; coords: { x: number; y: number; w: number; h: number } } | null>(null);
  const sessionStartRef = useRef<number>(0);
  const samplesRef = useRef<{ bpm: number; spo2: number; hb: number; bpSys: number; bpDia: number; sugar: number; signalQuality: number; respRate: number; sdnn: number; rmssd: number; pnn50: number; rhythm: string }[]>([]);
  const autoStopTimerRef = useRef<number | null>(null);
  const countdownTimerRef = useRef<number | null>(null);

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
  } | null>(null);
  const [form, setForm] = useState({
    name: "",
    age: "",
    sex: "" as "" | "female" | "male",
    lmp: "",
    pregnancy: "" as "" | "pregnant" | "non-pregnant",
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [readingCount, setReadingCount] = useState(0);
  const [showHistory, setShowHistory] = useState(false);
  const [exporting, setExporting] = useState(false);
  
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
    setReadingCount(getReadingCount());
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

  const detectFaceROI = useCallback((video: HTMLVideoElement): { roi: number[][]; coords: { x: number; y: number; w: number; h: number } } | null => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, 64, 64);
    const imageData = ctx.getImageData(0, 0, 64, 64);
    const data = imageData.data;
    
    let minX = 64, maxX = 0, minY = 64, maxY = 0;
    let found = false;
    
    for (let y = 0; y < 64; y++) {
      for (let x = 0; x < 64; x++) {
        const r = data[(y * 64 + x) * 4];
        const g = data[(y * 64 + x) * 4 + 1];
        const b = data[(y * 64 + x) * 4 + 2];
        const brightness = (r + g + b) / 3;
        
        if (brightness > 20 && brightness < 240) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }
    
    if (!found || maxX - minX < 10 || maxY - minY < 10) return null;
    
    const roi: number[][] = [];
    const regions = [
      { x: minX + (maxX - minX) * 0.1, y: minY + (maxY - minY) * 0.1, w: (maxX - minX) * 0.3, h: (maxY - minY) * 0.3 },
      { x: minX + (maxX - minX) * 0.6, y: minY + (maxY - minY) * 0.1, w: (maxX - minX) * 0.3, h: (maxY - minY) * 0.3 },
      { x: minX + (maxX - minX) * 0.35, y: minY + (maxY - minY) * 0.35, w: (maxX - minX) * 0.3, h: (maxY - minY) * 0.3 },
      { x: minX + (maxX - minX) * 0.1, y: minY + (maxY - minY) * 0.6, w: (maxX - minX) * 0.35, h: (maxY - minY) * 0.25 },
      { x: minX + (maxX - minX) * 0.55, y: minY + (maxY - minY) * 0.6, w: (maxX - minX) * 0.35, h: (maxY - minY) * 0.25 },
    ];
    
    for (const region of regions) {
      const regionData: number[] = [];
      const startX = Math.max(0, Math.floor(region.x));
      const endX = Math.min(64, Math.floor(region.x + region.w));
      const startY = Math.max(0, Math.floor(region.y));
      const endY = Math.min(64, Math.floor(region.y + region.h));
      
      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const idx = (y * 64 + x) * 4;
          regionData.push(data[idx], data[idx + 1], data[idx + 2]);
        }
      }
      roi.push(regionData);
    }
    
    return { roi, coords: { x: minX, y: minY, w: maxX - minX, h: maxY - minY } };
  }, []);

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

    const c = document.createElement("canvas");
    c.width = 64;
    c.height = 64;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    if (video) ctx.drawImage(video, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;

    let r = 0, g = 0, b = 0, count = 0;
    if (cameraMode === "rear") {
      const cx0 = 22, cy0 = 22, cx1 = 42, cy1 = 42;
      for (let y = cy0; y < cy1; y++) {
        for (let x = cx0; x < cx1; x++) {
          const i = (y * 64 + x) * 4;
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          count++;
        }
      }
    } else {
      for (let i = 0; i < data.length; i += 4) {
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
        count++;
      }
    }
    const avgR = count ? r / count : 128;
    const avgG = count ? g / count : 128;
    const avgB = count ? b / count : 128;
    
    const now = performance.now();
    if (lastFrameTimeRef.current > 0) {
      const dt = now - lastFrameTimeRef.current;
      if (dt > 0) {
        const instFps = 1000 / dt;
        fpsRef.current = fpsRef.current * 0.9 + instFps * 0.1;
      }
    }
    lastFrameTimeRef.current = now;

    const isFingerFrame = cameraMode === "rear"
      ? (avgR > 90 && avgR < 245 && avgR > avgG + 15 && avgR > avgB + 15 && Math.abs(avgR - avgG) > 8)
      : true;
    if (cameraMode === "rear" && !demoMode) setFingerDetected(isFingerFrame);

    if (demoMode) {
      const fs = Math.max(10, Math.min(60, fpsRef.current || 30));
      const sample = generateSyntheticPPG(0.1, fs, 72);
      syntheticBufferRef.current.push(...sample);
      if (syntheticBufferRef.current.length > 1800) {
        syntheticBufferRef.current.splice(0, syntheticBufferRef.current.length - 1800);
      }
      for (const v of sample) {
        valuesRef.current.push(128 + (v - 128) * 1.2);
        if (valuesRef.current.length > 180) valuesRef.current.shift();
        redRef.current.push(128 + (v - 128) * 1.2);
        greenRef.current.push(v);
        blueRef.current.push(128 + (v - 128) * 0.4);
        if (redRef.current.length > 1800) redRef.current.shift();
        if (greenRef.current.length > 1800) greenRef.current.shift();
        if (blueRef.current.length > 1800) blueRef.current.shift();
      }
      setFingerDetected(true);
    } else {
      if (phaseRef.current === "measuring" && isFingerFrame) {
        ppgBufferRef.current.push(avgG);
        if (ppgBufferRef.current.length > 1800) ppgBufferRef.current.shift();
      }

      valuesRef.current.push(avgR);
      if (valuesRef.current.length > 180) valuesRef.current.shift();
      redRef.current.push(avgR);
      greenRef.current.push(avgG);
      blueRef.current.push(avgB);
      if (redRef.current.length > 1800) redRef.current.shift();
      if (greenRef.current.length > 1800) greenRef.current.shift();
      if (blueRef.current.length > 1800) blueRef.current.shift();
    }

    // Run comprehensive vitals analysis using centralized module
    const sessionStart = sessionStartRef.current;
    const elapsedMs = sessionStart > 0 ? Date.now() - sessionStart : 0;
    const isMeasuring = phaseRef.current === "measuring" && elapsedMs >= 10000 && elapsedMs < 100000;
    
    // Update signal quality and vitals every 2 seconds
    const nowMs = Date.now();
    if (nowMs - lastBpmUpdate.current > 2000) {
      lastBpmUpdate.current = nowMs;

      const fs = Math.max(10, Math.min(60, fpsRef.current || 30));
      const analysisSource = demoMode
        ? syntheticBufferRef.current.slice(-Math.min(syntheticBufferRef.current.length, Math.floor(fs * 30)))
        : ppgBufferRef.current.slice(-Math.min(ppgBufferRef.current.length, Math.floor(fs * 30)));
      const analysis: VitalAnalysis = analyzeVitals(analysisSource, fs);
      const hr = analysis.bpm !== null ? Math.min(Math.max(analysis.bpm, 40), 180) : null;

      // Use SQI from analyzeVitals (0-100)
      const sqi = analysis.heart.signalQuality;
      setSignalQuality(sqi / 100);

      // Compute SpO2 and Hb from bandpass-filtered color signals
      // Note: Smartphone cameras lack IR (940nm), so SpO2 is an estimate only
      let spo2Val: number | null = null;
      let hbVal: number | null = null;
      if (redRef.current.length >= fs * 3 && greenRef.current.length >= fs * 3 && blueRef.current.length >= fs * 3) {
        // Bandpass filter color signals at 0.5-3.5 Hz for proper AC/DC
        const redFiltered = butterworthBandpass(redRef.current, 0.5, 3.5, fs);
        const greenFiltered = butterworthBandpass(greenRef.current, 0.5, 3.5, fs);
        const blueFiltered = butterworthBandpass(blueRef.current, 0.5, 3.5, fs);
        
        const getAcDc = (raw: number[], filtered: number[]) => {
          const dc = raw.reduce((a, b) => a + b, 0) / raw.length;
          let ac = 0;
          for (const v of filtered) ac += Math.abs(v - (filtered.reduce((a, b) => a + b, 0) / filtered.length));
          ac /= filtered.length;
          return { ac, dc };
        };
        
        const { ac: acR, dc: dcR } = getAcDc(redRef.current, redFiltered);
        const { ac: acG, dc: dcG } = getAcDc(greenRef.current, greenFiltered);
        const { ac: acB, dc: dcB } = getAcDc(blueRef.current, blueFiltered);
        
        // SpO2: R = (AC_Red/DC_Red) / (AC_Blue/DC_Blue) - Note: needs IR (940nm) for accuracy
        // Using blue as proxy for IR (not clinically valid, for display only)
        if (dcR > 0 && dcB > 0 && acR > 0 && acB > 0) {
          const Rratio = (acR / dcR) / (acB / dcB);
          // Standard formula: SpO2 = 110 - 25*R (calibrated for 660nm/940nm)
          // With blue as IR proxy, values will be off - cap and offset
          spo2Val = 110 - 25 * Rratio + spo2Offset;
          spo2Val = Math.min(Math.max(spo2Val, 70), 100);
        }
        
        // Hb: Hgb = 18.2 - 6.8 * (PI_Red / PI_Green)
        if (dcR > 0 && dcG > 0 && acR > 0 && acG > 0) {
          const HbRatio = (acR / dcR) / (acG / dcG);
          hbVal = 18.2 - 6.8 * HbRatio;
          hbVal = Math.min(Math.max(hbVal, 5), 20);
        }
      }

      if (hr !== null) {
        setBpm(hr);
        setHb(hbVal ?? (demoMode ? 13.5 + (Math.random() - 0.5) * 1.0 : null));
        setSpo2(spo2Val ?? (demoMode ? Math.round(97 + (Math.random() - 0.5) * 2) : null));
        // BP and Glucose cannot be reliably measured from PPG - display as unavailable
        setBp(null);
        setSugar(null);
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

        if (isMeasuring) {
          samplesRef.current.push({
            bpm: hr,
            spo2: spo2Val ?? (demoMode ? Math.round(97 + (Math.random() - 0.5) * 2) : 0),
            hb: hbVal ?? (demoMode ? 13.5 + (Math.random() - 0.5) * 1.0 : 0),
            bpSys: 0,
            bpDia: 0,
            sugar: 0,
            signalQuality: sqi / 100,
            respRate: analysis.respRate ?? 15,
            sdnn: analysis.hrv?.sdnn ?? 0,
            rmssd: analysis.hrv?.rmssd ?? 0,
            pnn50: analysis.hrv?.pnn50 ?? 0,
            rhythm: analysis.rhythm === "unknown" ? "regular" : analysis.rhythm,
          });
        }
      }
    } else {
      setBpm(null);
      setHb(null);
      setHrv(null);
      setRespRate(null);
      setRhythm("unknown");
      setSpo2(null);
      setBp(null);
      setSugar(null);
    }

    drawWaveform();
    if (cameraOnRef.current) {
      animFrame.current = requestAnimationFrame(processFrame);
    }
  }, [drawWaveform, spo2Offset, cameraMode, demoMode]);

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
      setHb(null);
      setBpm(null);
      setRespRate(null);
      setHrv(null);
      setRhythm("unknown");
      setSpo2(null);
      setBp(null);
      setSugar(null);
      setFingerDetected(false);
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
    const avg = (key: keyof typeof samples[number]) =>
      samples.length > 0 ? samples.reduce((a, s) => a + (s[key] as number), 0) / samples.length : 0;
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

    let computedReading: { bpm: number; spo2: number; hb: number; bpSys: number; bpDia: number; sugar: number; signalQuality: number; respRate: number; sdnn: number; rmssd: number; pnn50: number; rhythm: "regular" | "irregular" | "unknown" } | null = null;

    if (samples.length > 0) {
      const bpSysAvg = Math.round(avg("bpSys"));
      const bpDiaAvg = Math.round(avg("bpDia"));
      computedReading = {
        bpm: avg("bpm"),
        spo2: Math.round(avg("spo2")),
        hb: avg("hb"),
        // BP and Glucose not reliably measurable from PPG - set to 0
        bpSys: 0,
        bpDia: 0,
        sugar: 0,
        signalQuality: avg("signalQuality"),
        respRate: avg("respRate"),
        sdnn: avg("sdnn"),
        rmssd: avg("rmssd"),
        pnn50: avg("pnn50"),
        rhythm: mode(samples.map((s) => (s.rhythm as "regular" | "irregular" | "unknown"))),
      };
    } else {
      computedReading = {
        bpm: bpm ?? 0,
        spo2: spo2 ?? 0,
        hb: hb ?? 0,
        bpSys: 0,
        bpDia: 0,
        sugar: 0,
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
      addVitalReading({
        patient: {
          name: patient.name,
          age: patient.age,
          sex: patient.sex as "male" | "female",
          lmp: patient.lmp,
          pregnancy: patient.pregnancy as "pregnant" | "non-pregnant" | undefined,
        },
        vitals: computedReading,
      }, videoBlob);
      setReadingCount(getReadingCount());
    }
  }, [bpm, hb, spo2, signalQuality, respRate, hrv, rhythm, stopStream, stopVideoRecording, patient]);

  const startAutoCountdown = useCallback(() => {
    sessionStartRef.current = Date.now();
    samplesRef.current = [];
    videoChunksRef.current = [];
    setFinalReading(null);
    setPhase("warming");
    setCountdown(100);
    if (countdownTimerRef.current) window.clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = window.setInterval(() => {
      const elapsed = Date.now() - sessionStartRef.current;
      const remaining = Math.max(0, 100 - Math.floor(elapsed / 1000));
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
      if (!form.lmp) {
        setFormError("Please enter the date of last menstrual period (LMP).");
        return;
      }
      if (form.pregnancy !== "pregnant" && form.pregnancy !== "non-pregnant") {
        setFormError("Please indicate pregnancy status.");
        return;
      }
    }
    setFormError(null);
    setPatient({
      name: form.name.trim(),
      age: form.age.trim(),
      sex: form.sex,
      lmp: form.lmp,
      pregnancy: form.pregnancy,
    });
  }, [form]);

  if (!patient) {
    return (
      <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
        <h1 className="text-4xl font-extrabold mb-2 tracking-tight text-center">PPG Vital Monitor</h1>
        <p className="text-base text-neutral-400 mb-6 max-w-lg text-center">
          Enter patient details to begin a 100-second measurement session.
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
                    onClick={() => setForm({ ...form, sex: "female" })}
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
                    onClick={() => setForm({ ...form, sex: "male", lmp: "", pregnancy: "" })}
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
                          : "bg-neutral-800 border border-neutral-700 text-neutral-300 hover:border-neutral-600"
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
                          : "bg-neutral-800 border border-neutral-700 text-neutral-300 hover:border-neutral-600"
                      }`}
                    >
                      Non-pregnant
                    </button>
                  </div>
                </div>
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
              Launch Monitor
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
      <h1 className="text-4xl font-extrabold mb-4 tracking-tight text-center">PPG Vital Monitor</h1>
      <div className="mb-4 px-3 py-2 rounded-full bg-neutral-900 border border-neutral-800 text-xs text-neutral-300">
        <span className="text-neutral-500 mr-2">Patient:</span>
        <span className="font-semibold text-neutral-100">{patient.name}</span>
        <span className="text-neutral-500"> • {patient.age}y • {patient.sex === "female" ? "Female" : "Male"}</span>
{patient.sex === "female" && (
          <span className="text-neutral-500"> • LMP {patient.lmp} • {patient.pregnancy === "pregnant" ? "Pregnant" : "Non-pregnant"}</span>
        )}
      </div>
      <p className="text-base text-neutral-400 mb-6 max-w-lg text-center">
        Front camera: position your face in good lighting. Rear camera + flash: cover the lens with your fingertip.
      </p>
      {phase === "finalized" && finalReading && (
        <div className="w-full max-w-lg mb-6 p-4 rounded-xl border-2 border-emerald-500/60 bg-emerald-950/30 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-bold text-emerald-300">Measurement Complete</h2>
            <span className="text-xs text-emerald-200/80">100s session • 90s recorded after 10s warm-up</span>
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-center">
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">Heart Rate</p>
              <p className="text-2xl font-bold text-red-400">{finalReading.bpm.toFixed(1)} BPM</p>
            </div>
            <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
              <p className="text-neutral-500 text-[10px] mb-1">SpO₂</p>
              <p className="text-2xl font-bold text-cyan-400">{finalReading.spo2}%</p>
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
              <p className="text-xl font-bold text-rose-400">{finalReading.hb.toFixed(1)} g/dL</p>
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
              Hb: {finalReading.hb.toFixed(1)} g/dL
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
            <div className={`absolute top-2 right-2 ${fingerDetected ? "bg-emerald-700/80 text-emerald-100" : "bg-amber-700/80 text-amber-100"} text-xs px-2 py-1 rounded`}>
              {fingerDetected ? "Finger detected" : "Place fingertip over lens"}
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
                   ? `Measuring — recording next 90s (${countdown}s remaining)`
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
            <div className="ml-auto flex items-center gap-2">
              <label className="text-[10px] text-neutral-400">SpO₂ offset</label>
              <input
                type="range"
                min="-10"
                max="10"
                step="1"
                value={spo2Offset}
                onChange={(e) => setSpo2Offset(Number(e.target.value))}
                className="w-24"
              />
              <span className="text-[10px] text-neutral-500 w-6">{spo2Offset > 0 ? `+${spo2Offset}` : spo2Offset}</span>
            </div>
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
              <p className="text-neutral-400 mb-1">Signal Quality</p>
              <p className="text-2xl font-bold text-violet-400">{signalQuality !== null ? `${(signalQuality * 100).toFixed(1)}%` : "--"}</p>
            </div>
          </div>
        )}
        {phase === "finalized" && finalReading && (
          <div className="px-4 pb-4">
            <div className="rounded-xl border-2 border-emerald-500/60 bg-emerald-950/30 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-base font-bold text-emerald-300">Final Reading (100s analysis)</h2>
                <span className="text-[10px] text-emerald-200/80">90s recording after 10s warm-up</span>
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
                  <p className="text-2xl font-bold text-cyan-400">{finalReading.spo2}%</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Hb</p>
                  <p className="text-2xl font-bold text-rose-400">{finalReading.hb.toFixed(1)} g/dL</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">BP</p>
                  <p className="text-xl font-bold text-emerald-400">N/A</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Glucose</p>
                  <p className="text-xl font-bold text-amber-400">N/A</p>
                </div>
                <div className="bg-neutral-900/80 rounded-lg p-3 border border-emerald-700/40">
                  <p className="text-neutral-400 mb-1">Signal</p>
                  <p className="text-xl font-bold text-violet-400">{(finalReading.signalQuality * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => {
                    downloadReadings(getVitalReadings(), "csv");
                    setReadingCount(getReadingCount());
                  }}
                  disabled={exporting}
                  className="flex-1 px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 transition"
                >
                  {exporting ? "Exporting..." : "Export CSV & Clear"}
                </button>
                <button
                  onClick={() => {
                    downloadReadings(getVitalReadings(), "excel");
                    setReadingCount(getReadingCount());
                  }}
                  disabled={exporting}
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
                  setSignalQuality(null);
                  setRespRate(null);
                  setHrv(null);
                  setRhythm("unknown");
                  setPatient(null);
                  setForm({ name: "", age: "", sex: "", lmp: "", pregnancy: "" });
                  startedRef.current = false;
                }}
                className="mt-4 w-full px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 transition"
              >
                Start New Measurement
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
                {getVitalReadings().map((reading: VitalReading) => (
                  <div key={reading.id} className="bg-neutral-800 rounded-lg p-3 border border-neutral-700 text-xs">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-neutral-100 font-medium">{reading.patient.name}</p>
                        <p className="text-neutral-400">
                          {new Date(reading.timestamp).toLocaleString()} • {reading.vitals.bpm.toFixed(0)} BPM • SpO₂ {reading.vitals.spo2}% • {reading.vitals.rhythm}
                        </p>
                      </div>
                      <button
                        onClick={() => {
                          deleteVitalReading(reading.id);
                          setReadingCount(getReadingCount());
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
                ))}
                {getVitalReadings().length === 0 && (
                  <p className="text-neutral-500 text-center py-4">No readings stored yet</p>
                )}
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  onClick={() => {
                    downloadReadings(getVitalReadings(), "csv");
                    setReadingCount(getReadingCount());
                  }}
                  disabled={getVitalReadings().length === 0 || exporting}
                  className="flex-1 px-3 py-2 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 transition"
                >
                  Export All as CSV & Clear
                </button>
                <button
                  onClick={() => {
                    downloadReadings(getVitalReadings(), "excel");
                    setReadingCount(getReadingCount());
                  }}
                  disabled={getVitalReadings().length === 0 || exporting}
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
                  width: phase === "finalized" ? "100%" : phase === "measuring" ? `${Math.min(100, ((100 - countdown) / 100) * 100)}%` : phase === "warming" ? `${Math.min(100, ((100 - countdown) / 100) * 100)}%` : "0%",
                }}
              />
            </div>
            <p className="mt-1 text-[10px] text-neutral-500 text-center">
              {phase === "warming" && `Warm-up — discarding first 10s (${countdown}s)`}
              {phase === "measuring" && `Measuring — recording next 90s (${countdown}s left)`}
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
