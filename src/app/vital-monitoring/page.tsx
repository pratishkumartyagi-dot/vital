"use client";

import { useRef, useState, useCallback } from "react";

export default function VitalMonitoringPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const toggleCamera = useCallback(async () => {
    if (cameraOn) {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setCameraOn(false);
      setError(null);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraOn(true);
      setError(null);
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Camera access denied or unavailable"
      );
      setCameraOn(false);
    }
  }, [cameraOn]);

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
          />
          {!cameraOn && (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-neutral-500 text-sm">Camera off</span>
            </div>
          )}
        </div>
        <div className="p-4 flex items-center justify-between">
          <span className="text-sm text-neutral-300">
            {cameraOn ? "Camera active" : "Ready"}
          </span>
          <button
            onClick={toggleCamera}
            className="px-5 py-2.5 rounded-full text-sm font-semibold bg-red-600 hover:bg-red-700 active:bg-red-800 transition shadow-lg shadow-red-900/30"
          >
            {cameraOn ? "Stop Camera" : "Start Camera"}
          </button>
        </div>
        {error && (
          <div className="px-4 pb-4">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}
      </div>

      <p className="mt-8 text-xs text-neutral-500">Research-grade PPG signal processing</p>
    </main>
  );
}
