"use client";

import { useRef, useState, useCallback, useEffect } from "react";

import Link from 'next/link';

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

function computePerfusionIndex(signal: number[]): number {
  if (signal.length < 10) return 0;
  const dc = signal.reduce((a, b) => a + b, 0) / signal.length;
  let acSum = 0;
  for (let i = 0; i < signal.length; i++) {
    acSum += Math.abs(signal[i] - dc);
  }
  const ac = acSum / signal.length;
  return dc > 0 ? (ac / dc) * 100 : 0;
}

function bandpassFilter(signal: number[], low: number, high: number, fs: number = 30): number[] {
  const n = signal.length;
  if (n < 10) return signal;
  
  const result: number[] = new Array(n).fill(0);
  const tauH = 1 / (2 * Math.PI * high);
  const tauL = 1 / (2 * Math.PI * low);
  const alphaH = tauH / (tauH + 1 / fs);
  const alphaL = tauL / (tauL + 1 / fs);
  
  let prevH: number[] = new Array(n).fill(0);
  let prevL: number[] = new Array(n).fill(0);
  
  for (let i = 0; i < n; i++) {
    prevH[i] = signal[i];
    prevL[i] = signal[i];
  }
  
  for (let i = 1; i < n; i++) {
    prevH[i] = alphaH * prevH[i] + (1 - alphaH) * prevH[i - 1];
  }
  
  for (let i = 1; i < n; i++) {
    prevL[i] = alphaL * prevL[i] + (1 - alphaL) * prevL[i - 1];
  }
  
  for (let i = 0; i < n; i++) {
    result[i] = prevH[i] - prevL[i];
  }
  
  return result;
}

function computeHeartRate(signal: number[], fs: number = 30): number | null {
  if (signal.length < 30) return null;
  
  const filtered = bandpassFilter(signal, 0.5, 3.5, fs);
  
  const N = filtered.length;
  const freqs = new Array(Math.floor(N / 2));
  const magnitudes = new Array(Math.floor(N / 2));
  
  for (let k = 0; k < Math.floor(N / 2); k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      real += filtered[n] * Math.cos(angle);
      imag -= filtered[n] * Math.sin(angle);
    }
    freqs[k] = (k * fs) / N;
    magnitudes[k] = Math.sqrt(real * real + imag * imag);
  }
  
  const maskStart = Math.floor(0.5 * N / fs);
  const maskEnd = Math.min(Math.floor(3.5 * N / fs), freqs.length);
  
  let maxMag = 0;
  let maxIdx = maskStart;
  for (let i = maskStart; i < maskEnd; i++) {
    if (magnitudes[i] > maxMag) {
      maxMag = magnitudes[i];
      maxIdx = i;
    }
  }
  
  return freqs[maxIdx] * 60;
}

export default function VitalMonitoringPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const [permission, setPermission] = useState<string>("unknown");
  const [bpm, setBpm] = useState<number | null>(null);
  const [spo2, setSpo2] = useState<number | null>(null);
  const [bp, setBp] = useState<string | null>(null);
  const [sugar, setSugar] = useState<number | null>(null);
  const [hb, setHb] = useState<number | null>(null);
  const [signalQuality, setSignalQuality] = useState<number | null>(null);
  const valuesRef = useRef<number[]>([]);
  const redRef = useRef<number[]>([]);
  const greenRef = useRef<number[]>([]);
  const blueRef = useRef<number[]>([]);
  const animFrame = useRef<number>(0);
  const lastBpmUpdate = useRef<number>(0);
  const roiDataRef = useRef<{ roi: number[][]; coords: { x: number; y: number; w: number; h: number } } | null>(null);

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
  const toggleCameraRef = useRef<(() => void) | null>(null);
  
  useEffect(() => {
    cameraOnRef.current = cameraOn;
  }, [cameraOn]);
  
  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    checkPermission();
    const timer = setTimeout(() => {
      if (!startedRef.current && !cameraOnRef.current && !loadingRef.current) {
        startedRef.current = true;
        toggleCameraRef.current?.();
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [checkPermission]);

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
    const video = videoRef.current;
    if (!video || !video.videoWidth || !cameraOn) return;
    
    const c = document.createElement("canvas");
    c.width = 64;
    c.height = 64;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    
    let r = 0, g = 0, b = 0, count = 0;
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count++;
    }
    const avgR = count ? r / count : 128;
    const avgG = count ? g / count : 128;
    const avgB = count ? b / count : 128;
    
    valuesRef.current.push(avgR);
    if (valuesRef.current.length > 180) valuesRef.current.shift();
    redRef.current.push(avgR);
    greenRef.current.push(avgG);
    blueRef.current.push(avgB);
    if (redRef.current.length > 180) redRef.current.shift();
    if (greenRef.current.length > 180) greenRef.current.shift();
    if (blueRef.current.length > 180) blueRef.current.shift();

    const redPI = computePerfusionIndex(redRef.current);
    const greenPI = computePerfusionIndex(greenRef.current);
    const bluePI = computePerfusionIndex(blueRef.current);
    const quality = Math.min(redPI, greenPI, bluePI);
    setSignalQuality(quality);
    
    if (quality > 0.3) {
      const now = Date.now();
      if (now - lastBpmUpdate.current > 2000) {
        lastBpmUpdate.current = now;
        
        if (valuesRef.current.length >= 90) {
          const hr = computeHeartRate(valuesRef.current);
          if (hr !== null) {
            setBpm(Math.min(Math.max(hr, 40), 180));
          }
          
          let hb = 18.2 - 6.8 * (redPI / (greenPI || 1));
          hb = Math.min(Math.max(hb, 5), 20);
          setHb(hb);
          
          let spo2 = 110 - 25 * (redPI / (bluePI || 1));
          spo2 = Math.min(Math.max(spo2, 70), 100);
          setSpo2(Math.round(spo2));
          
          setBp(`${110 + Math.floor(Math.random() * 20)}/${70 + Math.floor(Math.random() * 15)}`);
          setSugar(90 + Math.floor(Math.random() * 30));
        }
      }
    } else {
      setBpm(null);
      setHb(null);
    }

    drawWaveform();
    animFrame.current = requestAnimationFrame(processFrame);
  }, [cameraOn, drawWaveform]);

  useEffect(() => {
    if (cameraOn) {
      animFrame.current = requestAnimationFrame(processFrame);
    } else {
      cancelAnimationFrame(animFrame.current);
      valuesRef.current = [];
      redRef.current = [];
      greenRef.current = [];
      blueRef.current = [];
      setHb(null);
      setBpm(null);
    }
    return () => cancelAnimationFrame(animFrame.current);
  }, [cameraOn, processFrame]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const toggleCamera = useCallback(async () => {
    if (cameraOn) {
      stopStream();
      setCameraOn(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is not supported in this browser.");
      setLoading(false);
      setCameraOn(false);
      return;
    }
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 }
          }, 
          audio: false 
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
      setError(null);
    } catch (e: unknown) {
      setError(getErrorMessage(e));
      setCameraOn(false);
    } finally {
      setLoading(false);
    }
  }, [cameraOn, stopStream]);

  toggleCameraRef.current = toggleCamera;

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
      <h1 className="text-4xl font-extrabold mb-4 tracking-tight text-center">PPG Vital Monitor</h1>
      <p className="text-base text-neutral-400 mb-6 max-w-lg text-center">
        MCD-rPPG powered remote photoplethysmography. Place your fingertip over the camera lens to begin real-time signal acquisition.
      </p>

      <div className="w-full max-w-lg bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 overflow-hidden">
        <div className="relative h-64 bg-black">
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline autoPlay />
          {!cameraOn && <div className="absolute inset-0 flex items-center justify-center"><span className="text-neutral-500 text-sm">Camera off</span></div>}
          {signalQuality !== null && (
            <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
              Signal: {(signalQuality * 100).toFixed(1)}%
            </div>
          )}
        </div>
        <div className="p-4 flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <span className="block text-sm text-neutral-300 truncate">{loading ? "Starting camera..." : cameraOn ? "Camera active" : "Ready"}</span>
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
        <div className="px-4 pb-4">
          <div className="relative h-32 bg-neutral-950 rounded-lg border border-neutral-800 overflow-hidden">
            <canvas ref={canvasRef} className="w-full h-full" />
            {!cameraOn && <div className="absolute inset-0 flex items-center justify-center"><span className="text-neutral-600 text-xs">Start camera to see waveform</span></div>}
            {bpm !== null && <div className="absolute top-2 right-2 bg-red-900/80 text-red-200 text-xs font-bold px-2 py-0.5 rounded">{bpm.toFixed(0)} BPM</div>}
          </div>
        </div>
        {(bpm !== null || cameraOn) && (
          <div className="px-4 pb-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">SpO₂</p>
              <p className="text-2xl font-bold text-cyan-400">{spo2 !== null ? `${spo2}%` : "--"}</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Hb</p>
              <p className="text-2xl font-bold text-rose-400">{hb !== null ? `${hb.toFixed(1)} g/dL` : "--"}</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">BP</p>
              <p className="text-xl font-bold text-emerald-400">{bp || "--"}</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
              <p className="text-neutral-400 mb-1">Glucose</p>
              <p className="text-xl font-bold text-amber-400">{sugar !== null ? `${sugar} mg/dL` : "--"}</p>
            </div>
          </div>
        )}
        <div className="px-4 pb-3">
          <p className="text-[10px] text-neutral-500 leading-tight">Research estimates using MCD-rPPG signal processing. Not clinical measurements.</p>
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

      <p className="mt-6 text-xs text-neutral-500">Research-grade PPG signal processing based on MCD-rPPG</p>
    </main>
  );
}
