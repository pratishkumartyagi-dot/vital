"use client";

import { useState } from "react";
import { jsPDF } from "jspdf";

interface MethodRow {
  label: string;
  value?: string | string[];
}

interface MethodSection {
  heading: string;
  content: MethodRow[];
}

const methodologyContent: { title: string; subtitle: string; sections: MethodSection[] } = {
  title: "Vital Scan — Methodology Report",
  subtitle: "MCD-rPPG Remote Photoplethysmography & Vital Signs Estimation",
  sections: [
    {
      heading: "1. Signal Acquisition",
      content: [
        {"label": "Camera Modes", "value": "Front (face) and Rear (finger-on-lens contact PPG)"},
        {"label": "Frame Rate", "value": "10-60 fps (configurable, device-dependent)"},
        {"label": "ROI", "value": "8 facial regions via MediaPipe 68-point landmarks"},
        {"label": "Color Space", "value": "RGB (660nm red, 550nm green, 450nm blue)"},
      ],
    },
    {
      heading: "2. PPG Signal Extraction",
      content: [
        {"label": "Front Camera", "value": "POS (Plane-Orthogonal-to-Skin) algorithm (Wang et al. 2017)"},
        {"label": "Rear Camera", "value": "Raw RGB channel means from finger ROI"},
        {"label": "Bandpass Filter", "value": "RBJ Butterworth HP(0.5 Hz) + LP(3.5 Hz), zero-phase filtfilt"},
        {"label": "Signal Quality", "value": "Composite SQI: 0.40 x matched-bandwidth SNR + 0.35 x perfusion (sqrt law) + 0.25 x RR regularity (MAD-robust CV), x cardiac spectral prominence gate"},
        {"label": "ZLO Correction", "value": "Sensor black level (16 DN default) subtracted from DC"},
      ],
    },
    {
      heading: "3. Heart Rate (HR)",
      content: [
        {"label": "Method", "value": "Peak detection (percentile-threshold) + spectral fusion (Hann-windowed FFT)"},
        {"label": "Acceptance Window", "value": ">= 55% of estimated beat period (rejects dicrotic notches)"},
        {"label": "Ectopic Rejection", "value": "RR intervals > 30% from median rejected before HRV"},
        {"label": "HRV Metrics", "value": "SDNN, RMSSD, pNN50, LF, HF, LF/HF"},
      ],
    },
    {
      heading: "4. SpO2 (Peripheral Perfusion Index)",
      content: [
        {"label": "Method", "value": "Ratio-of-ratios (empirical fallback) or ROSE-Net CNN+BiLSTM+attention"},
        {"label": "Ratio", "value": "R = (AC_red/DC_red) / (AC_green/DC_green)"},
        {"label": "Empirical Formula", "value": "SpO2 = 106 - 8*R, clamped [88, 100]"},
        {"label": "AC Computation", "value": "Time-domain: mean absolute deviation of bandpass-filtered (0.5-3.5 Hz) signal from its mean"},
        {"label": "DC Computation", "value": "mean(raw) - ZLO (16 DN)"},
        {"label": "Smartphone Proxy", "value": "Red/green ratio used because smartphones lack IR (940nm); green (550nm) absorption tracks deoxyhemoglobin"},
        {"label": "ROSE-Net", "value": "CNN+BiLSTM+attention on STMaps, cross-domain trained (BIDMC contact PPG -> rPPG video)"},
      ],
    },
    {
      heading: "5. Hemoglobin (Hb)",
      content: [
        {"label": "Formula", "value": "Hb = 18.2 - 6.8*R (red/green ratio, g/dL)"},
        {"label": "Range", "value": "8 - 18 g/dL"},
      ],
    },
    {
      heading: "6. Blood Pressure (BP)",
      content: [
        {"label": "PPG Features", "value": "Delta (dicrotic notch), PEP, MT, rise time, systolic/diastolic area ratio"},
        {"label": "Method", "value": "Single-site MAP estimation + pulse transit time (PTT) for systolic/diastolic"},
        {"label": "Calibration", "value": "Per-user offset (mmHg)"},
      ],
    },
    {
      heading: "7. Respiratory Rate (RR)",
      content: [
        {"label": "Front Camera", "value": "Raw green baseline wander bandpass (0.06-0.7 Hz) -> peak + DFT fusion"},
        {"label": "Rear Camera", "value": "Analysis source channel via SNR-weighted selection"},
        {"label": "Acceptance", "value": "RR regularity gate via MAD-robust CV on breath intervals"},
      ],
    },
    {
      heading: "8. Glucose (Research Estimate)",
      content: [
        {"label": "Features", "value": "PPG waveform morphology: AC/DC, pulse rise time, reflection timing, spectral centroid, harmonic ratios"},
        {"label": "Model", "value": "Gradient boosting regression (XGBoost)"},
        {"label": "Confidence", "value": "Capped at 0.55 (lower than direct-contact vitals)"},
        {"label": "Note", "value": "Research estimate only — not a substitute for a blood glucose meter"},
      ],
    },
    {
      heading: "9. Creatinine & eGFR (Research Estimate)",
      content: [
        {"label": "Features", "value": "19 descriptors: peak/dicrotic amplitude, pulse interval, RR CV, augmentation index, stiffness ratio, rise slope, 2nd-derivative timings, perfusion index, R-G/R-B ratios, harmonic ratio"},
        {"label": "Model", "value": "Camera-mode aware regression (rear: RF MAE 0.102 mg/dL; front: XGBoost+Optuna 85.2% acc, ROC-AUC 0.86)"},
        {"label": "eGFR", "value": "CKD-EPI 2021 equation (race-free)"},
        {"label": "KDIGO Stage", "value": "G1-G5"},
        {"label": "Confidence", "value": "Rear: 0.80 cap; Front: 0.55 cap (modeScale applied)"},
        {"label": "Note", "value": "Research estimate only — not a substitute for a serum creatinine blood test"},
      ],
    },
    {
      heading: "10. Model Training (MCD-rPPG)",
      content: [
        {"label": "Architecture", "value": "SCNN_8rois (1D Feature Pyramid Network with attention, 8 ROI channels)"},
        {"label": "Dataset", "value": "MCD-rPPG (Hugging Face: kyegorov/mcd_rppg)"},
        {"label": "Folds", "value": "5-fold cross-validation"},
        {"label": "Batch Size", "value": "36"},
        {"label": "Results", "value": "PPG MAE: 0.5556 | HR MAE: 4.32 BPM"},
      ],
    },
  ],
};

export default function TrainingMethodologyToggle() {
  const [expanded, setExpanded] = useState(false);
  const [exporting, setExporting] = useState(false);

  function handleExportPdf() {
    setExporting(true);
    try {
      const doc = new jsPDF({ unit: "mm", format: "a4" });
      const lineHeight = 7;
      let y = 20;

      doc.setFontSize(20);
      doc.text(methodologyContent.title, 20, y);
      y += lineHeight * 1.5;

      doc.setFontSize(11);
      doc.setTextColor(100, 100, 100);
      doc.text(methodologyContent.subtitle, 20, y);
      y += lineHeight * 2;

      doc.setTextColor(0, 0, 0);

      methodologyContent.sections.forEach((section) => {
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text(section.heading, 20, y);
        y += lineHeight * 1.3;

        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        section.content.forEach((row) => {
          const label = `${row.label}:`;
          const value = Array.isArray(row.value) ? row.value.join(", ") : (row.value ?? "—");
          doc.setTextColor(80, 80, 80);
          doc.text(label, 25, y);
          doc.setTextColor(40, 40, 40);
          const maxWidth = 150;
          const lines = doc.splitTextToSize(value, maxWidth);
          doc.text(lines, 45, y);
          y += lineHeight * (lines.length > 0 ? lines.length : 1);
        });
        y += lineHeight * 0.5;

        if (y > 260) {
          doc.addPage();
          y = 20;
        }
      });

      doc.save("vital-scan-methodology.pdf");
    } catch (err) {
      console.error("PDF export failed:", err);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="w-full max-w-2xl mx-auto mt-6">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-neutral-800 hover:bg-neutral-700 text-white rounded-lg transition"
      >
        <span className="font-medium">Calculation & Measurement Parameters</span>
        <span className={`transform transition-transform ${expanded ? "rotate-180" : ""}`}>
          ⌄
        </span>
      </button>

      {expanded && (
        <div className="mt-2 p-4 bg-neutral-900 rounded-lg border border-neutral-800">
          {methodologyContent.sections.map((section) => (
            <div key={section.heading} className="mb-4 last:mb-0">
              <h3 className="text-md font-bold text-red-400 mb-2">{section.heading}</h3>
              <table className="w-full text-sm">
                <tbody>
                  {section.content.map((row) => (
                    <tr key={row.label}>
                      <td className="text-neutral-400 py-1 pr-2 font-medium">{row.label}</td>
                      <td className="text-neutral-200 py-1">
                        {row.value !== undefined && row.value !== null
                          ? (typeof row.value === "string"
                              ? row.value
                              : row.value.join(", "))
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          <div className="mt-4 pt-4 border-t border-neutral-800">
            <button
              onClick={handleExportPdf}
              disabled={exporting}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 text-white rounded-lg transition text-sm font-medium"
            >
              {exporting ? "Exporting..." : "Export as PDF"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
