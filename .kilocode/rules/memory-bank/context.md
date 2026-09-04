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
- [x] **Bugfix (SpO2)**: replaced frame-average perfusion-index ratio with proper AC/DC ratio from bandpass-filtered red/blue PPG buffers (`computeAcDc`) for clinically meaningful SpO2.
- [x] **Bugfix (all-zero readings)**: `ppg-analysis.ts` `bandpass()` was a bandstop (notch) rejecting the 0.7–3.5 Hz HR band; rewrote as proper Butterworth bandpass (HP→LP cascade). Peak detection now uses 2nd-order butterBandpass with percentile-based adaptive threshold (20% above 10th-percentile baseline).
- [x] **Bugfix (all-zero readings recurrence)**: The "fixed" bandpass in commit `5cd9aa5` had HP and LP time constants swapped (`tauH` used `high` for HP cutoff, `tauL` used `low` for LP cutoff), creating a bandpass with NO passband (HP at 3.5 Hz removes all signal below 3.5 Hz including heart rate, LP at 0.7 Hz then removes everything above 0.7 Hz). Swapped so HP cutoff uses `low` (0.7 Hz) and LP cutoff uses `high` (3.5 Hz), correctly passing 0.7-3.5 Hz. Fixed in both `ppg-analysis.ts` (`bandpass`) and `page.tsx` (`bandpassFilter`).
- [x] **Bugfix (warm-up leak)**: `ppgBufferRef` was filled during warm-up phase, contaminating the signal buffer used for HR/HRV analysis. Now `ppgBufferRef` only fills during `phase === "measuring"` (checked via `phaseRef.current`).
- [x] **Bugfix (ppgBuffer cap / all-zero live parameters)**: The previous "cap ppgBuffer at 180" change was itself a regression. `analyzeVitals` requires `fs*5` samples for heart-rate (300 at 60fps) and `fs*10` for respiration (600), but `ppgBufferRef` (the real-camera analysis buffer) was capped at 180 while `fpsRef` tracks `requestAnimationFrame` (~60fps). 180 < 300, so `analyzeVitals` always returned `bpm: null`, `hr` stayed null, the entire `if (hr !== null)` results block was skipped, `samplesRef` never filled, and `finalizeSession` fell into its zero-default branch -> waveforms rendered fine but all parameters read 0. Restored the `ppgBufferRef` cap to 1800 to match `syntheticBufferRef` and the `fs*30` analysis window requested by the slice (1800 = 60*30, covers `fs*5`/`fs*10` at any fs in [10,60]). Verified end-to-end: real-mode simulation now yields BPM~72, RespRate~15, HRV populated.
- [x] **Bugfix (AC/DC SpO2/Hb)**: SpO2 and Hb were using raw perfusion-index ratios (PI = AC/DC but computed on unfiltered signal). Now use proper AC/DC from bandpass-filtered (0.7–3.5 Hz) buffers: `R = (AC_r/dc_r)/(AC_b/dc_b)` for SpO2, `R = (AC_r/dc_r)/(AC_g/dc_g)` for Hb (18.2–6.8R).
- [x] **Bugfix (stale measuring gate)**: `processFrame` used `measuring` from stale `phase` in callback closure. Replaced with `isMeasuring` using `phaseRef.current`. Removed stale `phase` from `useCallback` dependency array.
- [x] **Bugfix (rear-camera fingertip PPG all-zero readings)**: torch defaulted to OFF so fingertip PPG had no illumination; torch toggle only re-applied on next `toggleCamera`. Sampling averaged the entire 64x64 frame so a well-covered fingertip produced no AC. The perfusion-index quality gate (0.05) was too strict for fingertip mode. Fixes: (1) selecting Rear Camera auto-enables `torchOn`; (2) Flash button now calls `track.applyConstraints({advanced:[{torch}]})` on the live track, with `getCapabilities()` check and a graceful error if the device/browser doesn't expose torch; (3) `processFrame` samples a centered 20×20 ROI in rear mode (where the finger fills the frame) and the full frame in front mode (face rPPG); (4) quality gate relaxed to 0.005 in rear mode; (5) added "Finger detected / Place fingertip over lens" badge based on red-dominance heuristic; (6) `ppgBufferRef` only fills when `phaseRef.current === "measuring"` AND a finger-like frame is detected, so warm-up frames and non-finger frames no longer contaminate the analysis buffer.
- [x] **Demo Mode (synthetic PPG)**: when no usable camera PPG can be extracted (camera permission denied, no finger on lens, room noise, browser without torch support), the user can now toggle **Demo Mode** to drive the entire pipeline with a synthetic pulse generator (`generateSyntheticPPG`: fundamental + dicrotic notch + respiratory baseline + small noise). In Demo Mode: (1) `toggleCamera` skips `getUserMedia` entirely; (2) `processFrame` ignores the video element, fills `syntheticBufferRef` instead of `ppgBufferRef`, and renders the synthetic pulse on the waveform canvas; (3) HR/HRV/RespRate come from `analyzeVitals` against the synthetic buffer, so the analysis pipeline is exercised end-to-end; (4) SpO₂/BP/Glucose/Hb use sensible priors; (5) the camera area shows a "Demo: synthetic PPG" overlay so it is obvious no real measurement is happening. This guarantees the app always produces a complete 70s reading session.
- [x] **Native Android PPG DSP Pipeline**: Implemented production-grade Kotlin `PpgSignalProcessor` in the Flutter Android native module (`vital_monitoring_app/android/app/src/main/kotlin/com/vitals/app/PpgSignalProcessor.kt`). Includes rolling buffer, 4th-order IIR Butterworth bandpass (0.5–4.0 Hz @ 30fps), detrending, and adaptive peak detection with physiological bounds check (45–180 BPM). Runs on a dedicated `HandlerThread` to avoid UI jitter.
- [x] **MethodChannel PPG Integration**: Exposed native DSP to Flutter via new `com.vitals.app/ppg_processor` MethodChannel in `MainActivity.kt`. Added `processPpgSample` and `resetPpgBuffer` handlers that post work to the background thread and return BPM to the UI thread. Updated `NativeHardwareBridge` (`lib/native_hardware_bridge.dart`) with `processPpgSample()` and `resetPpgBuffer()` methods.
- [x] **Flutter Live PPG + Native DSP Fusion**: `VitalSignController.attachStreamPipeline` now pipes each green-channel sample through `NativeHardwareBridge.processPpgSample()` for native BPM estimation. Buffer reset paths (PI gate failure, state rejection) also call `NativeHardwareBridge.resetPpgBuffer()` to keep native state consistent.
- [x] **Corrected signal strength (strength of signal) calculations**: Fixed Butterworth bandpass filter coefficients in Dart (`ppg_signal_engine.dart`) using proper 4th-order SOS cascaded biquads (0.5-3.5 Hz @ 30 Hz). Fixed perfusion index calculation to use proper AC/DC from individual pulse peak-to-trough amplitudes instead of global max-min. Fixed TypeScript signal quality index (SQI) in `ppg-analysis.ts` to use multi-component scoring (perfusion index, spectral SNR, pulse regularity) instead of arbitrary scaling. Fixed frequency range inconsistency in `params.py` (ppg_high_freq: 10 Hz → 3.5 Hz) and TypeScript (0.7 Hz → 0.5 Hz lower cutoff). Verified haemoglobin formula (Hgb = 18.2 - 6.8*R_Hb) is consistent with dual-wavelength PPG literature.

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
| 2026-09-01 | Fixed bandpass filter bug (was bandstop) and peak detection threshold in `ppg-analysis.ts` that caused all-zero readings |
| 2026-09-01 | Fixed recurring bandpass cutoff swap bug: HP filter must use `low` cutoff (0.7 Hz), LP must use `high` cutoff (3.5 Hz). Previous fix had them swapped, producing an empty passband. Fixed in both `ppg-analysis.ts` and `page.tsx` |
| 2026-09-01 | Fixed rear camera and flash function: added `switchCameraMode` that stops the current stream and restarts with the new facing mode; the camera mode buttons now call this function instead of just setting state; fixed torch constraint application to use direct `{ torch: bool }` instead of `advanced: [{ torch: bool }]` |
| 2026-09-01 | Fixed rear-camera fingertip PPG: auto-enable torch on Rear Camera selection, dynamic torch constraint on the live track, center-ROI sampling, relaxed quality gate (0.005), finger-detection badge |
| 2026-09-01 | Added Demo Mode (synthetic PPG generator) so the 70s measurement session always produces a complete reading, even when no real camera PPG is available (no finger, no face, no torch support) |
| 2026-09-02 | Audit of the Launch Monitor pipeline: live-camera measurements returned 0 for all parameters despite waveforms rendering. Root cause: `ppgBufferRef` capped at 180 samples (< `fs*5`=300 required by `analyzeVitals` at 60fps rAF) made HR/resp analysis always return null, skipping all parameter updates and the `samplesRef` accumulation so `finalizeSession` emitted zeros. Raised the cap to 1800 (matching `syntheticBufferRef` and the `fs*30` analysis window). Verified via simulation: BPM~72, RespRate~15, HRV populated. `bun typecheck` and `bun lint` pass. |
| 2026-09-03 | Added native Android PPG DSP pipeline: `PpgSignalProcessor.kt` (rolling buffer, 4th-order IIR Butterworth bandpass 0.5–4.0 Hz, detrending, adaptive peak detection) running on a dedicated `HandlerThread`. Exposed via new `com.vitals.app/ppg_processor` MethodChannel in `MainActivity.kt` and `NativeHardwareBridge` Dart wrapper. Flutter live stream now pipes green-channel samples to native DSP for more accurate BPM estimation. |
| 2026-09-03 | Extended web PPG session from 70s to 100s: first 10s warm-up discarded, next 90s recorded for waveform/analysis. Updated `src/app/vital-monitoring/page.tsx` timing constants (`isMeasuring` window, countdown start, progress bar math) and all UI labels (patient form, session badge, status text, final reading card). `bun typecheck` and `bun lint` pass. |
| 2026-09-03 | Implemented adaptive Android PPG camera manager to prevent rear-camera signal saturation (>250). Replaced hardcoded `applyStrictLocks` with `configureSafePPG` MethodChannel in `MainActivity.kt` that queries `INFO_SUPPORT_LEVEL`, `REQUEST_AVAILABLE_CAPABILITIES`, ISO/exposure/AE-compensation ranges, and flash availability. Added `_applyAdaptivePpgSettings()` in `main.dart` to push exposure offset toward minimum underexposure, lock focus, and enable torch. Added software frame guard in `vital_sign_controller.dart` to reject frames where R/G/B > 250 or all channels < 15. Updated `AndroidManifest.xml` with optional autofocus feature. |
| 2026-09-04 | Corrected signal strength (strength of signal) calculations and mathematical formulas: Fixed Butterworth bandpass filter coefficients in Dart (`ppg_signal_engine.dart`) using proper 4th-order SOS cascaded biquads (0.5-3.5 Hz @ 30 Hz). Fixed perfusion index calculation to use proper AC/DC from individual pulse peak-to-trough amplitudes. Fixed TypeScript signal quality index (SQI) in `ppg-analysis.ts` to use multi-component scoring (perfusion index, spectral SNR, pulse regularity). Fixed frequency range inconsistency in `params.py` (ppg_high_freq: 10 Hz → 3.5 Hz) and TypeScript (0.7 Hz → 0.5 Hz lower cutoff). Verified haemoglobin formula (Hgb = 18.2 - 6.8*R_Hb) is consistent with dual-wavelength PPG literature. |

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
