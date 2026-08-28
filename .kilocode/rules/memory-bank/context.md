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

## Session History

| Date | Changes |
|------|---------|
| Initial | Next.js 16 starter template |
| 2026-08-26 | Complete rebuild as Flutter Vital Monitoring App |
| 2026-08-26 (Final) | Added demo mode, improved error handling, comprehensive instructions overlay |
| 2026-08-28 | Integrated MCD-rPPG training pipeline, enhanced web app with training dashboard |

## Technical Specifications

- **Framework**: Next.js 16 + React 19 + TypeScript
- **Styling**: Tailwind CSS 4
- **Signal Processing**: FFT-based HR estimation (0.5-3.5 Hz bandpass)
- **Model**: SCNN_8rois (1D CNN with attention over 8 ROIs)
- **Training**: PyTorch with 5-fold cross-validation
- **Dataset**: MCD-rPPG (600 subjects, multi-camera)
- **Package Manager**: Bun
