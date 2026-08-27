"use client";

import { useRef, useState, useCallback } from "react";

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
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => {
      t.stop();
    });
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
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
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
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
      <h1 className="text-5xl font-extrabold mb-6 tracking-tight text-center">
        PPG Vital Monitor
      </h1>
      <p className="text-lg text-neutral-400 mb-8 max-w-md text-center">
        Place your fingertip over the camera lens and flash to begin real-time
        signal acquisition.
      </p>

      <div className="w-full max-w-md bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 overflow-hidden">
        <div className="relative h-72 bg-black">
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            muted
            playsInline
            autoPlay
          />
          {!cameraOn && (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-neutral-500 text-sm">Camera off</span>
            </div>
          )}
        </div>
        <div className="p-4 flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <span className="block text-sm text-neutral-300 truncate">
              {loading ? "Starting camera..." : cameraOn ? "Camera active" : "Ready"}
            </span>
            {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
          </div>
          <button
            onClick={toggleCamera}
            disabled={loading}
            className="px-5 py-2.5 rounded-full text-sm font-semibold bg-red-600 hover:bg-red-700 active:bg-red-800 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg shadow-red-900/30 whitespace-nowrap"
          >
            {loading ? "Starting..." : cameraOn ? "Stop Camera" : "Start Camera"}
          </button>
        </div>
      </div>

      <p className="mt-8 text-xs text-neutral-500">Research-grade PPG signal processing</p>
    </main>
  );
}
