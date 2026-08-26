# Active Context: Vital Monitoring App (Flutter)

## Current State

**Project Status**: ✅ Fully functional research tool with demo mode and comprehensive UI

The project is a research-grade PPG Vital Signs application built with Flutter that accesses native Android hardware for high-fidelity photoplethysmography (PPG) signal processing. It implements hardware-level camera controls, real-time signal processing, a state machine for guided measurements, enhanced UI with real-time waveform visualization, and now includes a demo mode for testing/research without requiring camera access.

## Recently Completed

- [x] Project setup with Flutter and necessary dependencies (camera, ffi, provider)
- [x] Android platform configuration: permissions, manifest, minimum SDK version 21
- [x] Native Android hardware ISP lock implementation via MethodChannel to disable auto white balance and exposure
- [x] High-performance YUV420 frame extraction with zero-allocation RGB conversion
- [x] Real-time signal processing: 4th-order Butterworth bandpass filter and perfusion index calculation
- [x] State machine for measurement guidance (baseline, profiling, complete, rejected states)
- [x] Flutter UI with camera preview and real-time feedback prompts
- [x] Python validation script for Leave-One-Subject-Out (LOSO) clinical cross-validation
- [x] Enhanced UI with real-time PPG waveform visualization
- [x] Custom waveform painter widget for displaying PPG signals
- [x] Heart rate display when measurement completes
- [x] Display buffer management for smooth waveform rendering
- [x] **Demo/Test Mode** - App starts in demo mode by default for immediate research/testing
- [x] **Improved Error Handling** - Graceful fallback to demo mode when camera unavailable
- [x] **Comprehensive Instructions Overlay** - Detailed information about app purpose and usage
- [x] **Explicit Research Documentation** - Clear explanation of what the app is and how to use for research

## Current Structure

| File/Directory | Purpose | Status |
|----------------|---------|--------|
| `pubspec.yaml` | Flutter dependencies and configuration | ✅ Ready |
| `android/app/src/main/AndroidManifest.xml` | Android permissions and hardware features | ✅ Ready |
| `android/app/build.gradle` | Android app build configuration with minSdkVersion 21 | ✅ Ready |
| `android/app/src/main/kotlin/com/vitals/app/MainActivity.kt` | Native hardware lock implementation | ✅ Ready |
| `lib/native_hardware_bridge.dart` | Flutter interface to native hardware locks | ✅ Ready |
| `lib/zero_allocation_extractor.dart` | Zero-allocation YUV to RGB conversion | ✅ Ready |
| `lib/ppg_signal_engine.dart` | Butterworth filtering and perfusion index calculation | ✅ Ready |
| `lib/vital_sign_controller.dart` | State machine, stream processing, waveform buffer, demo mode | ✅ Ready |
| `lib/waveform_chart.dart` | Custom waveform visualization widget | ✅ Ready |
| `lib/main.dart` | Main application with camera integration, UI, waveform display, and info overlay | ✅ Ready |
| `validation_script.py` | LOSO validation for clinical analysis | ✅ Ready |

## Current Focus

The vital monitoring app is now fully functional for research purposes. Key features for research use:

1. **Immediate Usability**: Starts in demo mode by default, allowing instant testing without camera permissions or hardware
2. **Signal Visualization**: Real-time PPG waveform display for signal quality assessment
3. **Quantitative Output**: Heart rate measurement displayed when signal quality is sufficient
4. **Guided Measurement**: Step-by-step prompts to ensure proper usage technique
5. **Research Documentation**: Comprehensive built-in explanation of technical approach and usage
6. **Validation Ready**: Includes LOSO validation script for clinical-grade analysis
7. **Error Resilient**: Gracefully handles camera unavailability and continues in demo mode

## Session History

| Date | Changes |
|------|---------|
| Initial | Next.js 16 starter template |
| 2026-08-26 | Complete rebuild as Flutter Vital Monitoring App with native hardware integration, signal processing, and state machine |
| 2026-08-26 (Continued) | Enhanced UI with real-time PPG waveform visualization, custom waveform painter, and heart rate display |
| 2026-08-26 (Final) | Added demo mode, improved error handling, comprehensive instructions overlay, and explicit research documentation |

## Quick Start Guide

### To run the application for research:
1. **No setup required**: The app starts in demo mode immediately
2. **For live camera testing**:
   - Ensure Flutter SDK is installed (version 2.19.0 or higher)
   - Connect an Android device with camera and USB debugging enabled
   - Grant camera permissions when prompted
   - Place fingertip over camera lens and flash as instructed
3. **To access information**: Tap "Show Info" button for detailed usage instructions
4. **To reset measurement**: Use the floating action button (red refresh icon)
5. **For data analysis**: Use the included `validation_script.py` for LOSO validation

### Demo Mode Features:
- Simulates realistic PPG signals with heartbeat-like patterns
- Includes occasional motion artifacts for realism
- Follows the same state machine and processing pipeline as live mode
- Provides identical UI experience including waveform visualization and heart rate calculation
- Automatically starts when app launches or when camera is unavailable

### Live Mode Features (when camera available):
- Hardware ISP locking to prevent auto white balance/exposure changes
- Zero-allocation frame processing for efficient operation
- Real-time signal acquisition at ~30 FPS
- Perfusion index-based signal quality assessment
- Adaptive measurement duration based on signal stability

## Technical Specifications:

- **Signal Processing**: 4th-order Butterworth bandpass filter (0.5-3.5 Hz)
- **Heart Rate Range**: 40-180 BPM (constrained for safety)
- **Sampling Rate**: ~30 FPS from camera preview
- **Quality Metric**: Perfusion index (AC/DC * 100) with 0.5% threshold
- **Buffer Sizes**: 120 samples for processing (~4 sec), 300 for display (~10 sec)
- **Language**: Flutter/Dart with Kotlin native integration
- **Platform**: Android (min SDK 21, target SDK 34)