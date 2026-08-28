export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900 flex items-center justify-center">
      <div className="text-center px-6">
        <h1 className="text-4xl font-bold mb-4">Vital Monitoring App</h1>
        <p className="text-lg text-neutral-600 mb-2">
          Research-grade PPG signal processing and real-time waveform analysis.
        </p>
        <p className="text-sm text-neutral-500 mb-6">
          Powered by MCD-rPPG: Multi-Camera Dataset for Remote Photoplethysmography
        </p>
        <div className="flex gap-4 justify-center">
          <a
            href="/vital-monitoring"
            className="inline-block px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
          >
            Launch Monitor
          </a>
          <a
            href="/training"
            className="inline-block px-6 py-3 bg-neutral-800 text-white rounded-lg hover:bg-neutral-700 transition"
          >
            Training Dashboard
          </a>
        </div>
      </div>
    </main>
  );
}
