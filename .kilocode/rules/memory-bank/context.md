# Active Context: Vital Monitoring App (Next.js + MCD-rPPG)

## Current State

**Project Status**: ✅ Functional Next.js web app with MCD-rPPG training integration

The project is a research-grade PPG Vital Signs application built with Next.js 16. It includes a web-based vital monitoring page with real-time camera-based PPG signal processing, and a complete MCD-rPPG training pipeline integrated into the project for model training and evaluation.

## Recently Completed

- [x] Project setup with Next.js 16, React 19, TypeScript, Tailwind CSS 4
- [x] Web-based vital monitoring page with real-time PPG signal acquisition
- [x] POS-like bandpass filter and FFT-based heart rate calculation in browser
- [x] MCD-rPPG rppglib integration into `training/` directory
- [x] SCNN_8rois model implementation with 8 ROI processing
- [x] Complete training pipeline (`training/run_training.py`) with cross-validation
- [x] Training dashboard page (`/training`) with model configuration display
- [x] API route for training job management (`/api/training`)
- [x] Enhanced vital monitoring UI with signal quality indicator
- [x] Multi-ROI face detection approximation for improved signal extraction
- [x] Perfusion index-based signal quality assessment
- [x] Home page updated with links to monitor and training dashboard
- [x] **70-second timed measurement session on `/vital-monitoring`**: camera auto-starts, discards first 10s warm-up, averages the next 60s of samples, then stops the camera and displays a final reading card (BPM, SpO₂, Hb, BP, Glucose, Signal quality) on screen. Manual stop and "Start New Measurement" both reset state cleanly.
- [x] **Patient intake form on `/vital-monitoring`**: first screen now collects Name, Age, Sex (Female/Male). For Female, additionally requires Date of LMP and Pregnant/Non-pregnant status. Auto-start of the 70s session is gated on completing the form; entered details are shown as a header pill during the session and included in the final reading card.
- [x] **PPG Heart Rate & Respiratory Rate analysis** on `/vital-monitoring` using in-browser TypeScript module `src/app/vital-monitoring/ppg-analysis.ts` that mirrors the algorithms of `fmoessle/heart-rate-monitor` + `paulvagentcom/heartrate_analysis_python` (peak detection, RR-interval HRV with SDNN/RMSSD/pNN50, rhythm regularity classification) and `peterhcharlton/RRest` (autocorrelation + FFT fusion in the 0.1–0.5 Hz respiratory band). Live tile grid now shows Heart Rate, Rhythm, Resp Rate, HRV (SDNN/RMSSD), SpO₂, BP, Hb, Glucose; final reading card has a dedicated Cardiac & Respiratory (PPG-derived) panel.
- [x] **Python reference implementation** at `training/ppg_analysis.py` using the real `heartpy` and `RRest` libraries for offline validation (with the same fusion fallback if RRest import fails).
- [x] **Removed dead code** from `page.tsx`: old `computeHeartRate` and `bandpassFilter` functions replaced by the dedicated `ppg-analysis.ts` module.

## Current Structure

| File/Directory | Purpose | Status |
|----------------|---------|--------|
| `src/app/page.tsx` | Home page with navigation | ✅ Ready |
| `src/app/vital-monitoring/page.tsx` | Real-time PPG monitoring page | ✅ Ready |
| `src/app/training/page.tsx` | Training dashboard | ✅ Ready |
| `src/app/api/training/route.ts` | Training API endpoint | ✅ Ready |
| `src/app/globals.css` | Global styles | ✅ Ready |
| `src/app/layout.tsx` | Root layout | ✅ Ready |
| `training/rppglib/` | MCD-rPPG library code | ✅ Ready |
| `training/rppglib/models/SCNN_8rois.py` | SCNN model implementation | ✅ Ready |
| `training/rppglib/torchPOS.py` | POS signal extraction | ✅ Ready |
| `training/rppglib/face_utils.py` | Face landmark detection utilities | ✅ Ready |
| `training/rppglib/processing.py` | Signal processing (bandpass, FFT HR) | ✅ Ready |
| `training/rppglib/train.py` | Training loop and metrics | ✅ Ready |
| `training/rppglib/data_utils.py` | Video/PPG loading utilities | ✅ Ready |
| `training/run_training.py` | Main training pipeline script | ✅ Ready |
| `training/requirements.txt` | Python dependencies | ✅ Ready |

## Current Focus

The application is now a complete research platform for rPPG with:
1. **Web-based PPG Monitoring**: Real-time camera-based vital sign estimation
2. **MCD-rPPG Training Pipeline**: Full training infrastructure for SCNN_8rois and other models
3. **Training Dashboard**: Web UI for monitoring training configuration and results
4. **Signal Processing**: FFT-based HR estimation, bandpass filtering, perfusion index calculation
5. **Multi-ROI Extraction**: 5-region face ROI extraction for improved signal quality
6. **Patient Intake**: Name, Age, Sex, LMP (if female), Pregnancy status before measurement
7. **Timed 70s Session**: 10s warm-up discard, 60s averaging, auto-stop, final reading card
8. **Heart Rate + HRV**: Peak-detection-based HR, RR-interval HRV (SDNN/RMSSD/pNN50), rhythm regularity
9. **Respiratory Rate**: Autocorrelation + FFT fusion in 0.1–0.5 Hz band, browser-native

## Session History

| Date | Changes |
|------|---------|
| Initial | Next.js 16 starter template |
| 2026-08-26 | Complete rebuild as Flutter Vital Monitoring App |
| 2026-08-26 (Final) | Added demo mode, improved error handling, comprehensive instructions overlay |
| 2026-08-28 | Integrated MCD-rPPG training pipeline, enhanced web app with training dashboard |
| 2026-09-01 | Added 70s timed measurement session (10s warm-up discard + 60s averaging) with final reading card on `/vital-monitoring` |
| 2026-09-01 | Added patient intake form (Name, Age, Sex, LMP, Pregnancy) gating the 70s measurement session on `/vital-monitoring` |
| 2026-09-01 | Added PPG Heart Rate + HRV (heartpy-style) and Respiratory Rate (RRest-style) analysis in browser + Python reference implementation in `training/ppg_analysis.py` |

## Technical Specifications

- **Framework**: Next.js 16 + React 19 + TypeScript
- **Styling**: Tailwind CSS 4
- **Signal Processing**: FFT-based HR estimation (0.5-3.5 Hz bandpass)
- **Analysis Module**: `src/app/vital-monitoring/ppg-analysis.ts` — HeartPy-style peak detection + HRV (SDNN/RMSSD/pNN50) + RRest-style respiratory rate (autocorr/FFT fusion)
- **Python Reference**: `training/ppg_analysis.py` using `heartpy` + `RRest` for offline validation
- **Model**: SCNN_8rois (1D CNN with attention over 8 ROIs)
- **Training**: PyTorch with 5-fold cross-validation
- **Dataset**: MCD-rPPG (600 subjects, multi-camera)
- **Package Manager**: Bun
