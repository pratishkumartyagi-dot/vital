export default function VitalMonitoringPage() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6">
      <h1 className="text-5xl font-extrabold mb-6 tracking-tight">PPG Vital Monitor</h1>
      <p className="text-lg text-neutral-400 mb-10 max-w-md text-center">
        Place your fingertip over the camera lens and flash to begin real-time signal acquisition.
      </p>
      <div className="w-full max-w-md h-64 bg-neutral-900 rounded-2xl border border-neutral-800 flex items-center justify-center shadow-2xl shadow-red-900/20">
        <span className="text-neutral-600 text-sm">Live waveform preview</span>
      </div>
      <p className="mt-8 text-xs text-neutral-500">Research-grade PPG signal processing</p>
    </main>
  );
}
