export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900 flex items-center justify-center">
      <div className="text-center px-6">
        <h1 className="text-4xl font-bold mb-4">Vital Monitoring App</h1>
        <p className="text-lg text-neutral-600">
          Research-grade PPG signal processing and real-time waveform analysis.
        </p>
        <a
          href="/vital-monitoring"
          className="inline-block mt-6 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
        >
          Launch Monitor
        </a>
      </div>
    </main>
  );
}
