import Link from "next/link";

const metrics = [
  { label: "Heart Rate", value: "BPM", detail: "Peak + spectral fusion" },
  { label: "Blood Oxygen", value: "SpO₂", detail: "Red / green ratio-of-ratios" },
  { label: "Hemoglobin", value: "Hb", detail: "Dual-wavelength estimate" },
  { label: "Blood Pressure", value: "mmHg", detail: "Pulse-morphology model" },
  { label: "Respiratory Rate", value: "/min", detail: "Baseline-wander analysis" },
  { label: "Creatinine / eGFR", value: "mg/dL", detail: "CKD-EPI 2021 staging" },
];

const steps = [
  {
    title: "Enter patient details",
    body: "Capture age, sex, and relevant clinical context before the session begins.",
  },
  {
    title: "Scan with any camera",
    body: "Front camera reads remote PPG from the face; rear camera reads contact PPG from a fingertip.",
  },
  {
    title: "Get a full vitals panel",
    body: "A 100-second session produces a signal-quality scored reading you can review, save, and export.",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50">
      <header className="border-b border-neutral-800/80">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/15 border border-emerald-500/30">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            </span>
            <span className="text-lg font-bold tracking-tight">Resilient Health</span>
          </div>
          <nav className="hidden sm:flex items-center gap-6 text-sm text-neutral-400">
            <Link href="/vital-monitoring" className="hover:text-white transition">
              Monitor
            </Link>
            <Link href="/validation" className="hover:text-white transition">
              Validation
            </Link>
            <Link href="/training" className="hover:text-white transition">
              Training
            </Link>
          </nav>
        </div>
      </header>

      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/10 via-transparent to-transparent pointer-events-none" />
        <div className="max-w-6xl mx-auto px-6 pt-20 pb-16 text-center relative">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-xs font-medium text-emerald-300 mb-6">
            Research-grade camera vitals
          </span>
          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight mb-5">
            Resilient Health
          </h1>
          <p className="text-lg text-neutral-400 max-w-2xl mx-auto mb-8">
            Turn any smartphone camera into a vital-signs monitor. Resilient Health
            extracts heart rate, oxygen saturation, hemoglobin, blood pressure,
            respiration, and kidney-function markers from a short PPG scan.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            <Link
              href="/vital-monitoring"
              className="inline-block px-7 py-3 bg-emerald-500 text-neutral-950 font-semibold rounded-full hover:bg-emerald-400 transition shadow-lg shadow-emerald-900/30"
            >
              Start a Scan
            </Link>
            <Link
              href="/validation"
              className="inline-block px-7 py-3 bg-neutral-900 border border-neutral-700 text-white font-semibold rounded-full hover:bg-neutral-800 transition"
            >
              Validation Dashboard
            </Link>
          </div>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-16">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-neutral-500 mb-6">
          What a session measures
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="rounded-2xl bg-neutral-900 border border-neutral-800 p-5 hover:border-emerald-500/40 transition"
            >
              <p className="text-neutral-400 text-sm">{m.label}</p>
              <p className="text-2xl font-bold text-white mt-1">{m.value}</p>
              <p className="text-xs text-neutral-500 mt-2">{m.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-20">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-neutral-500 mb-6">
          How it works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {steps.map((s, i) => (
            <div key={s.title} className="rounded-2xl bg-neutral-900 border border-neutral-800 p-6">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-sm font-bold text-emerald-300 mb-4">
                {i + 1}
              </span>
              <p className="font-semibold text-white mb-2">{s.title}</p>
              <p className="text-sm text-neutral-400">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-neutral-800/80">
        <div className="max-w-6xl mx-auto px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-neutral-500">
          <p>
            Resilient Health is a research tool. Readings are estimates and not a
            substitute for clinical diagnosis.
          </p>
          <Link href="/training" className="hover:text-neutral-300 transition">
            Model training dashboard
          </Link>
        </div>
      </footer>
    </main>
  );
}
