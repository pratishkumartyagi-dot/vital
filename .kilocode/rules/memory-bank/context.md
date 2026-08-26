# Active Context: Vital Monitoring App (Flutter)

## Current State

**Project Status**: ✅ Core implementation completed

The project is a research-grade PPG Vital Signs application built with Flutter that accesses native Android hardware for high-fidelity photoplethysmography (PPG) signal processing. It implements hardware-level camera controls, real-time signal processing, and a state machine for guided measurements.

## Recently Completed

- [x] Project setup with Flutter and necessary dependencies (camera, ffi, provider)
- [x] Android platform configuration: permissions, manifest, minimum SDK version 21
- [x] Native Android hardware ISP lock implementation via MethodChannel to disable auto white balance and exposure
- [x] High-performance YUV420 frame extraction with zero-allocation RGB conversion
- [x] Real-time signal processing: 4th-order Butterworth bandpass filter and perfusion index calculation
- [x] State machine for measurement guidance (baseline, profiling, complete, rejected states)
- [x] Flutter UI with camera preview and real-time feedback prompts
- [x] Python validation script for Leave-One-Subject-Out (LOSO) clinical cross-validation

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
| `lib/vital_sign_controller.dart` | State machine and stream processing logic | ✅ Ready |
| `lib/main.dart` | Main application with camera integration and UI | ✅ Ready |
| `validation_script.py` | LOSO validation for clinical analysis | ✅ Ready |

## Current Focus

The core vital monitoring functionality is implemented. Next steps could include:

1. Enhancing the UI to display real-time waveforms and vital signs
2. Adding data persistence to store measurement sessions
3. Implementing export functionality for recorded PPG signals
4. Adding unit tests for signal processing components
5. Refining the state machine with additional states for different measurement types
6. Optimizing performance for lower-end devices

## Session History

| Date | Changes |
|------|---------|
| Initial | Next.js 16 starter template |
| 2026-08-26 | Complete rebuild as Flutter Vital Monitoring App with native hardware integration, signal processing, and state machine |

## Quick Start Guide

### To run the application:
1. Ensure Flutter SDK is installed (version 2.19.0 or higher)
2. Connect an Android device with USB debugging enabled or use an emulator
3. Run `flutter pub get` to install dependencies
4. Run `flutter run` to launch the application

### To modify hardware locking:
- Edit `android/app/src/main/kotlin/com/vitals/app/MainActivity.kt` to adjust camera parameters
- Update the method channel communication in `native_hardware_bridge.dart` if needed

### To adjust signal processing:
- Modify filter coefficients in `ppg_signal_engine.dart` for different sampling rates or frequency bands
- Adjust the perfusion index calculation or downsampling factors in `zero_allocation_extractor.dart`

### To change UI behavior:
- Update `vital_sign_controller.dart` to modify state transitions and timing
- Edit `main.dart` to change the visual layout or feedback mechanisms