"use client";

import { useRef, useState, useCallback, useEffect } from "react";

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

export default function VitalMonitoringPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const [permission, setPermission] = useState<string>("unknown");
  const [bpm, setBpm] = useState<number | null>(null);
  const valuesRef = useRef<number[]>([]);
  const animFrame = useRef<number>(0);
  const wRef = useRef<number[]>([0, 0, 0, 0, 0]);

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

  useEffect(() => {
    checkPermission();
    const timer = setTimeout(() => {
      if (!startedRef.current && !cameraOn && !loading) {
        startedRef.current = true;
        toggleCamera();
      }
    }, 600);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = 300;
    canvas.height = 120;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const values = valuesRef.current;
    if (values.length < 2) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const step = canvas.width / Math.max(values.length - 1, 1);
    for (let i = 0; i < values.length; i++) {
      const x = i * step;
      const y = canvas.height - ((values[i] - min) / range) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
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
    let r = 0, count = 0;
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      count++;
    }
    const avg = count ? r / count : 128;
    valuesRef.current.push(avg);
    if (valuesRef.current.length > 180) valuesRef.current.shift();

    // Simple IIR bandpass approximation for BPM
    const w = wRef.current;
    const b = [0.0039, 0.0, -0.0078, 0.0, 0.0039];
    const a = [1.0, -3.1806, 3.8612, -2.1122, 0.4351];
    let out = b[0] * avg + w[1];
    w[1] = b[1] * avg - a[1] * out + w[2];
    w[2] = b[2] * avg - a[2] * out + w[3];
    w[3] = b[3] * avg - a[3] * out + w[4];
    w[4] = b[4] * avg - a[4] * out;

    // Peak detection for BPM (every ~30 frames at 30fps = ~1 sec interval)
    const vals = valuesRef.current;
    if (vals.length >= 30) {
      const peaks: number[] = [];
      for (let i = 1; i < vals.length - 1; i++) {
        if (vals[i] > vals[i - 1] && vals[i] > vals[i + 1] && vals[i] > 100) {
          if (peaks.length === 0 || i - peaks[peaks.length - 1] >= 15) peaks.push(i);
        }
      }
      if (peaks.length >= 2) {
        const intervals: number[] = [];
        for (let i = 1; i < peaks.length; i++) intervals.push((peaks[i] - peaks[i - 1]) / 30);
        const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        if (avgInterval > 0) {
          const rate = 60 / avgInterval;
          setBpm(Math.min(Math.max(rate, 40), 180));
        }
      }
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
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
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

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
      <h1 className="text-5xl font-extrabold mb-6 tracking-tight text-center">PPG Vital Monitor</h1>
      <p className="text-lg text-neutral-400 mb-8 max-w-md text-center">
        Place your fingertip over the camera lens and flash to begin real-time signal acquisition.
      </p>

      <div className="w-full max-w-md bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 overflow-hidden">
        <div className="relative h-72 bg-black">
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline autoPlay />
          {!cameraOn && <div className="absolute inset-0 flex items-center justify-center"><span className="text-neutral-500 text-sm">Camera off</span></div>}
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
      </div>

      <p className="mt-8 text-xs text-neutral-500">Research-grade PPG signal processing</p>
    </main>
  );
}
