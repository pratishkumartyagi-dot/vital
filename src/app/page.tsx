import VitalScanApp from "./vital-monitoring/vital-scan-app";

export default function Home() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-50">
      <header className="sticky top-0 z-10 border-b border-neutral-800 bg-neutral-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div>
            <h1 className="text-xl font-extrabold tracking-tight">Vital Scan App</h1>
            <p className="text-xs text-neutral-400">
              Research-grade PPG signal processing and real-time waveform analysis.
            </p>
          </div>
          <nav className="flex flex-wrap gap-2 text-sm">
            <a
              href="#vital-scan"
              className="rounded-lg bg-red-600 px-4 py-2 font-semibold text-white transition hover:bg-red-700"
            >
              Vital Scan
            </a>
            <a
              href="/validation"
              className="rounded-lg bg-blue-600 px-4 py-2 font-semibold text-white transition hover:bg-blue-700"
            >
              Validation
            </a>
            <a
              href="/training"
              className="rounded-lg bg-neutral-800 px-4 py-2 font-semibold text-white transition hover:bg-neutral-700"
            >
              Training Dashboard
            </a>
          </nav>
        </div>
      </header>

      <section id="vital-scan" className="mx-auto max-w-5xl">
        <VitalScanApp />
      </section>
    </div>
  );
}
