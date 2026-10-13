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
  title: "Resilient Health — Methodology Report",
  subtitle: "Contact PPG (Rear Camera) & Remote PPG (Front Camera) Vital Signs Estimation",
  sections: [
    {
      heading: "1. Signal Acquisition",
      content: [
        {"label": "Contact PPG (Rear)", "value": "Camera: rear-facing. Torch: ON (auto-enabled). Finger placed over lens."},
        {"label": "rPPG (Front)", "value": "Camera: front-facing. Face positioned in good lighting. No torch."},
        {"label": "Frame Rate", "value": "Constant 30 fps (fixed in camera constraints: { ideal: 30, min: 30 })"},
        {"label": "Nyquist Criterion", "value": "Cardiac band 0.5-3.5 Hz requires >= 7 fps minimum; 30 fps provides 15x oversampling for HRV frequency analysis"},
        {"label": "ROI (Contact PPG)", "value": "Center 150x120 pixel region on rear camera lens"},
        {"label": "ROI (rPPG)", "value": "5 facial regions via MediaPipe FaceMesh (468 3D landmarks): forehead, left/right cheeks, nose bridge, lower face"},
        {"label": "ROI Update Rate", "value": "MediaPipe face detection every 10 frames, cached for 30 frames (front camera)"},
        {"label": "Color Space", "value": "RGB (660nm red, 550nm green, 450nm blue)"},
        {"label": "Frame Size", "value": "64x64 downscaled from video (rear: 640x480)"},
      ],
    },
    {
      heading: "2. PPG Signal Extraction",
      content: [
        {"label": "Contact PPG (Rear)", "value": "Raw RGB channel means from center 20x20 ROI (finger on lens)"},
        {"label": "rPPG (Front)", "value": "Spatial averaging of MediaPipe-detected face ROI regions"},
        {"label": "Bandpass Filter", "value": "RBJ Butterworth HP(0.5 Hz) + LP(3.5 Hz), zero-phase filtfilt"},
        {"label": "Detrending", "value": "Linear detrend (subtract best-fit line) before bandpass"},
        {"label": "Buffer Size", "value": "1800 samples (30 sec at 30 fps)"},
        {"label": "Warm-up", "value": "First 10 seconds discarded (filtfilt edge transients)"},
        {"label": "ZLO Correction", "value": "Sensor black level (16 DN default) subtracted from DC before AC/DC ratio"},
        {"label": "Finger Detection", "value": "Red-dominant + moderate brightness: avgR in [30, 200], avgR > avgG + 5, avgR > avgB + 5"},
      ],
    },
    {
      heading: "3. Heart Rate (HR)",
      content: [
        {"label": "Input Signal (Contact PPG)", "value": "Green channel (avgG) from finger ROI"},
        {"label": "Input Signal (rPPG)", "value": "Green channel from face ROI regions"},
        {"label": "Peak Detection", "value": "Percentile-based adaptive threshold (p10 + 0.20 * (p90 - p10))"},
        {"label": "Refractory Period", "value": ">= 55% of estimated beat period (rejects dicrotic notches)"},
        {"label": "Acceptance Window", "value": "RR intervals 300-2000 ms"},
        {"label": "Ectopic Rejection", "value": "RR intervals > 30% from median rejected before HRV"},
        {"label": "HRV Metrics", "value": "SDNN, RMSSD, pNN50 (50 ms threshold)"},
        {"label": "HRV CV Method", "value": "MAD-robust CV (1.4826 * MAD / mean) for ectopic-filtered RR series"},
        {"label": "Method", "value": "Peak detection + spectral fusion (Hann-windowed FFT)"},
        {"label": "Range", "value": "40-180 BPM (rejected outside this range)"},
      ],
    },
    {
      heading: "4. Signal Quality (SQI)",
      content: [
        {"label": "Formula", "value": "SQI = round(0.45 * PI_score + 0.35 * SNR_score + 0.20 * RR_regularity_score) * 100"},
        {"label": "Perfusion Index", "value": "Filtered PI: AC = RMS(bandpass-filtered, 0.5-3.5 Hz); DC = mean(raw) - ZLO (16 DN)"},
        {"label": "PI Threshold", "value": "1.5% full credit (sqrt law: piScore = sqrt(PI / 1.5))"},
        {"label": "Spectral SNR", "value": "Cardiac band (0.5-3.5 Hz) signal power / noise band power (3.5 Hz to fs*0.4 Hz)"},
        {"label": "SNR Threshold", "value": "SNR >= 4 for full credit"},
        {"label": "RR Regularity", "value": "MAD-robust CV on ectopic-filtered RR intervals, < 10% CV = 1.0"},
        {"label": "Pulse Gate", "value": "Cardiac spectral prominence (peak/median ratio) — gates out no-pulse signals"},
      ],
    },
    {
      heading: "5. SpO2 (Peripheral Perfusion Index)",
      content: [
        {"label": "Contact PPG (Rear)", "value": "Ratio-of-ratios: R = (AC_red/DC_red) / (AC_green/DC_green)"},
        {"label": "rPPG (Front)", "value": "Same ratio-of-ratios; green (550 nm) used as IR proxy (smartphones lack IR 940 nm)"},
        {"label": "Empirical Formula", "value": "SpO2 = 106 - 8*R, clamped [88, 100]"},
        {"label": "AC Computation", "value": "Time-domain: mean absolute deviation of bandpass-filtered (0.5-3.5 Hz) signal from its mean"},
        {"label": "DC Computation", "value": "mean(raw) - ZLO (16 DN, adaptive resolution)"},
        {"label": "ZLO Resolution", "value": "Returns 16.0 if signal min > 30; otherwise max(0, min(signal))"},
        {"label": "SQI Gate", "value": "SpO2 computed when SQI >= 30 (0.3 threshold)"},
        {"label": "ROSE-Net", "value": "CNN+BiLSTM+attention on STMaps, cross-domain trained (BIDMC contact PPG -> rPPG video). Fallback to empirical when unavailable."},
        {"label": "ROSE-Net Unavailable", "value": "Model not yet downloaded/loaded, IndexedDB cache miss, or WebGL inference failed — empirical formula provides immediate readings with lower accuracy"},
        {"label": "ROSE-Net Rationale", "value": "Smartphones lack IR (940nm) LEDs; RGB-only SpO2 is inherently limited. ROSE-Net models non-linear dynamics (skin perfusion, motion artifacts, temporal patterns) that simple R/G ratios cannot. Cross-domain training (BIDMC medical PPG -> synthesized RGB via Beer-Lambert -> rPPG video) bridges the IR gap."},
        {"label": "Confidence", "value": "0.5 (empirical) or model-derived (ROSE-Net)"},
      ],
    },
    {
      heading: "6. Hemoglobin (Hb)",
      content: [
        {"label": "Contact PPG (Rear)", "value": "Hb = 18.2 - 6.8 * R (red/green ratio, g/dL)"},
        {"label": "rPPG (Front)", "value": "Same formula, clamped [8, 18 g/dL]"},
        {"label": "AC", "value": "Time-domain mean absolute deviation of bandpass-filtered (0.5-3.5 Hz)"},
        {"label": "DC", "value": "mean(raw) - ZLO (16 DN, adaptive resolution)"},
        {"label": "ZLO", "value": "Black-level correction applied to both red and green channels"},
      ],
    },
    {
      heading: "7. Blood Pressure (BP)",
      content: [
        {"label": "PPG Features", "value": "Stiffness index (rise time), dicrotic notch amplitude, reflection timing"},
        {"label": "Method", "value": "Base MAP from HRV-derived estimates + pulse waveform morphology features"},
        {"label": "Confidence", "value": "Max 0.70 (lower than direct-contact vitals)"},
        {"label": "Note", "value": "Research estimate — not a substitute for a validated blood pressure cuff"},
      ],
    },
    {
      heading: "8. Respiratory Rate (RR)",
      content: [
        {"label": "Contact PPG (Rear)", "value": "Analysis source channel via SNR-weighted selection from red/green/blue buffers"},
        {"label": "rPPG (Front)", "value": "Raw green baseline wander bandpass (0.1-0.5 Hz) -> autocorrelation + DFT peak fusion"},
        {"label": "Acceptance", "value": "RR regularity gate via MAD-robust CV on breath intervals"},
        {"label": "Range", "value": "5-50 breaths per minute"},
        {"label": "Method Fusion", "value": "Autocorrelation (peak at respiratory period) + FFT (dominant frequency in 0.1-0.5 Hz band)"},
      ],
    },
    {
      heading: "9. Glucose (Research Estimate)",
      content: [
        {"label": "Features", "value": "PPG waveform morphology: AC/DC, pulse rise time, reflection timing, spectral centroid, harmonic ratios, perfusion index"},
        {"label": "Method", "value": "Heuristic regression from perfusion index ratios + waveform shape descriptors"},
        {"label": "Confidence", "value": "Capped at 0.55 (lower than direct-contact vitals)"},
        {"label": "Note", "value": "Research estimate only — not a substitute for a blood glucose meter"},
      ],
    },
    {
      heading: "10. Creatinine & eGFR (Research Estimate)",
      content: [
        {"label": "Features", "value": "19 descriptors: peak/dicrotic amplitude, pulse interval, RR CV, augmentation index, stiffness ratio, rise slope, 2nd-derivative timings, perfusion index, R-G/R-B ratios, harmonic ratio"},
        {"label": "Model", "value": "Camera-mode aware regression"},
        {"label": "eGFR", "value": "CKD-EPI 2021 equation (race-free)"},
        {"label": "KDIGO Stage", "value": "G1-G5"},
        {"label": "Confidence (Rear)", "value": "0.80 cap (contact PPG, higher signal quality)"},
        {"label": "Confidence (Front)", "value": "0.55 cap (rPPG, modeScale applied)"},
        {"label": "Note", "value": "Research estimate only — not a substitute for a serum creatinine blood test"},
      ],
    },
    {
      heading: "11. Model Training (MCD-rPPG)",
      content: [
        {"label": "Architecture", "value": "SCNN_8rois (1D Feature Pyramid Network with attention, 8 ROI channels)"},
        {"label": "Dataset", "value": "MCD-rPPG (Hugging Face: kyegorov/mcd_rppg)"},
        {"label": "Folds", "value": "5-fold cross-validation"},
        {"label": "Batch Size", "value": "36"},
        {"label": "Results", "value": "PPG MAE: 0.5556 | HR MAE: 4.32 BPM"},
      ],
    },
    {
      heading: "12. Validation: Contact PPG & rPPG Dual Measurement",
      content: [
        {"label": "Contact PPG (Rear)", "value": "150x120 px center ROI, torch ON, 30 fps fixed"},
        {"label": "rPPG (Front)", "value": "MediaPipe FaceMesh 468 landmarks, 5 ROI regions, 30 fps fixed"},
        {"label": "Signal Extraction (Contact)", "value": "OD_c(t) = -ln(I_c/I0_c) or p(t) = -detrended(C(t)); bandpass 0.5-4 Hz zero-phase"},
        {"label": "Signal Extraction (rPPG)", "value": "Cn(t) = C(t)/mean(C) detrended + bandpass 0.7-3.0 Hz. Methods: Green, CHROM (3R-2G, 1.5R+G-1.5B), POS overlap-added over 1.6s windows"},
        {"label": "Heart Rate", "value": "Beat detection with parabolic interpolation or spectral argmax in 0.7-3.0 Hz. Sub-frame timing via parabolic peak interpolation or spline upsampling to >=120 Hz"},
        {"label": "Per-beat AC/DC", "value": "AC = mean(P_max - P_min) per beat; DC = mean intensity. PI = (AC/DC)*100% as quality gate"},
        {"label": "SpO2 (Contact)", "value": "R = (AC_red/DC_red)/(AC_x/DC_x), x=green or blue. Fit: SpO2 = a - b*R (e.g. 110 - 25R). Also available: quadratic (a + bR + cR²) and log form"},
        {"label": "SpO2 (rPPG)", "value": "AC_c = spectral amplitude at f_hr (+/-0.15 Hz); DC_c = mean(C). R = (AC_red/DC_red)/(AC_green/DC_green). Empirical fit per phone model"},
        {"label": "Respiratory Rate", "value": "Three modulations: baseline wander (0.1-0.5 Hz), amplitude modulation (Hilbert envelope), frequency modulation (IBI series). SQI-weighted fusion of PSD peaks"},
        {"label": "HRV", "value": "Upsample to 120 Hz (cubic spline) before peak detection. SDNN, RMSSD, pNN50 (>50ms). LF(0.04-0.15 Hz)/HF(0.15-0.40 Hz) via Lomb-Scargle on NN series"},
        {"label": "Blood Pressure", "value": "Waveform features: stiffness index SI = height/ΔT, reflection index RI = b/a, APG ratios (b/a, c/a, d/a, e/a), aging index. Model: SBP = β0 + Σβi·xi (ridge/gradient boosting). Two-site adds PTT = a + b·ln(PTT). Per-user calibration offset required"},
        {"label": "Hemoglobin", "value": "Contact PPG: HMALCNN (Smarandache et al., CMES 2025, DOI:10.32604/cmes.2025.068736) — 4 conv blocks (Conv1D+BN+ReLU+MaxPool) + 3 FC layers on multi-wavelength PPG (R/G/B). Trained on 2 multi-regional datasets (68 & 152 subjects, 5-fold CV). RMSE 0.90/1.20 g/dL, r=0.82/0.72, 91.3%/86.7% within ±1 g/dL. Fallback: Absorbance per channel A_c = -ln(I_c/I0_c), AC/DC ratios, log-ratios, PI. Hb = 18.2 - 6.8*R (red/green). Per-device calibration. rPPG: empirical formula only."},
        {"label": "Glucose/Creatinine", "value": "y = f(waveform features, HR, HRV, age, sex, BMI). ML regression benchmarked against covariate-only baseline. Glucose: exploratory estimate only. Creatinine: CKD-EPI 2021 eGFR + KDIGO stages"},
        {"label": "Quality Index", "value": "SQI = corr(beat_i, mean_template) > 0.85 acceptance threshold. Combined: perfusion index, spectral SNR at f_hr, skewness/kurtosis, PI threshold > 0.2% or reject"},
        {"label": "Evaluation Metrics", "value": "d = app - reference; bias = mean(d); LoA = bias +/- 1.96*SD(d); MAE, RMSE, ARMS. Lin CCC = 2ρσxσy / (σx² + σy² + (μx - μy)²)"},
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
