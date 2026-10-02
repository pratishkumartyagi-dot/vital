# Technical Context: Vital Monitoring App

## Technology Stack

| Technology   | Version | Purpose                         |
| ------------ | ------- | ------------------------------- |
| Next.js      | 16.x    | React framework with App Router |
| React        | 19.x    | UI library                      |
| TypeScript   | 5.9.x   | Type-safe JavaScript            |
| Tailwind CSS | 4.x     | Utility-first CSS               |
| Bun          | Latest  | Package manager & runtime       |
| Drizzle ORM  | 0.45.x  | SQLite database ORM             |
| @kilocode/app-builder-db | 0.0.1 (vendored `vendor/app-builder-db`, 186d1c1 build) | SQLite-via-HTTP Drizzle client |
| PyTorch      | 2.6.0   | Deep learning framework         |
| Python       | 3.10+   | Training pipeline               |

## Development Environment

### Prerequisites
- Bun installed
- Python 3.10+ with pip (for training)

### Commands

```bash
bun install        # Install JS dependencies
bun dev            # Start dev server
bun build          # Production build
bun lint           # Run ESLint
bun typecheck      # Run TypeScript type checking

# Training (separate environment)
cd training
pip install -r requirements.txt
python run_training.py --model SCNN_8rois
```

## Project Configuration

### Next.js Config
- App Router enabled
- Default settings

### TypeScript Config
- Strict mode enabled
- Path alias: `@/*` → `src/*`

### Tailwind CSS 4
- Uses `@tailwindcss/postcss` plugin
- CSS-first configuration

## Training Pipeline

### Models Available
- SCNN_8rois (primary, 8 ROI attention-based CNN)
- POS, PBV, OMIT (baseline methods)
- PhysFormer, RhythmFormer, TSCAN, iBVPNet

### Dataset Format
- Preprocessed `.npy` files with video and PPG
- CSV metadata with fold assignments
- Supports MCD-rPPG, MMPD, SCAMPS, UBFC_rPPG

## PPG Vital-Sign Models (browser + `training/ppg_analysis.py`)

- **Filtering**: 2nd-order Butterworth highpass (0.5 Hz) + lowpass (3.5 Hz) biquad cascade (RBJ/bilinear-transform), applied zero-phase (forward-backward) and valid at any sampling rate. Do **not** reintroduce fixed SOS tables — a bad table previously centred the passband on ~0.5 Hz and broke every derived metric.
- **Peak/HR**: adaptive percentile threshold on the filtered signal; peaks accepted iteratively against the last accepted peak; ectopic RR (>30% from median) rejected before HRV.
- **Heart Rate / HRV**: HeartPy-style peak detection, RR-interval SDNN/RMSSD/pNN50, rhythm regularity.
- **Respiratory Rate**: RRest-style autocorrelation + FFT fusion (0.1–0.5 Hz).
- **SpO₂ / Hb**: ratio-of-ratios on **red/green** (green is the highest-SNR channel; red/green absorption is the accepted contact-camera proxy) from gated, bandpass-filtered AC/DC. SpO2 = clamp(110 − 25·R, 88, 100); Hb = clamp(18.2 − 6.8·R, 5, 20). Estimate only — no 940 nm IR channel; the `SpO₂ offset` slider is the per-device calibration hook.
- **Regression guard**: `bun run test:ppg` (`scripts/ppg-selfcheck.ts`) checks HR across 50–150 BPM, HRV sanity, respiratory rate, and SpO2/Hb ranges on synthetic signals.
- **Serum Creatinine (research estimate)**: PPG/rPPG morphology regression/classification per Sridevi et al. — *Smart Health* 2025 (rear fingertip PPG, RF MAE 0.102 mg/dL, n=213, 0.6–1.4 mg/dL) and *Healthcare Analytics* 2025 (front facial rPPG, MIMIC-III, XGBoost+Optuna 85.2% acc / ROC-AUC 0.86; classes <0.6 / 0.6–1.3 / >1.3 mg/dL). Camera-mode-aware confidence (rear cap 0.80; front `modeScale=0.5`, cap 0.55). eGFR via **CKD-EPI 2021** (race-free); KDIGO G1–G5 staging. `creatinineUmol = creatinine × 88.4`. Validation target ±0.3 mg/dL. **Not a substitute for a laboratory blood test.**

## File Structure

```
/
├── src/app/                      # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   ├── vital-monitoring/page.tsx
│   ├── training/page.tsx
│   └── api/training/route.ts
├── training/
│   ├── run_training.py
│   ├── requirements.txt
│   └── rppglib/
│       ├── models/
│       ├── train.py
│       ├── processing.py
│       ├── face_utils.py
│       └── data_utils.py
└── vital_monitoring_app/         # Flutter app (legacy)
```
