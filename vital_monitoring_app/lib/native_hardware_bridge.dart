import 'dart:convert';
import 'package:flutter/services.dart';

class CalibrationProfile {
  final String source;
  final bool isStale;
  final String model;
  final String marketingName;
  final String manufacturer;
  final double zlo;
  final double sensorLinearSlope;
  final int apiLevelMin;
  final int apiLevelMax;
  final int calibrationVersion;

  CalibrationProfile({
    required this.source,
    required this.isStale,
    required this.model,
    required this.marketingName,
    required this.manufacturer,
    required this.zlo,
    required this.sensorLinearSlope,
    required this.apiLevelMin,
    required this.apiLevelMax,
    required this.calibrationVersion,
  });

  factory CalibrationProfile.fromMap(Map<dynamic, dynamic> map) {
    return CalibrationProfile(
      source: map['source']?.toString() ?? 'FALLBACK',
      isStale: map['isStale'] as bool? ?? false,
      model: map['model']?.toString() ?? '',
      marketingName: map['marketingName']?.toString() ?? '',
      manufacturer: map['manufacturer']?.toString() ?? '',
      zlo: (map['zlo'] as num?)?.toDouble() ?? 16.0,
      sensorLinearSlope: (map['sensorLinearSlope'] as num?)?.toDouble() ?? 1.0,
      apiLevelMin: (map['apiLevelMin'] as num?)?.toInt() ?? 0,
      apiLevelMax: (map['apiLevelMax'] as num?)?.toInt() ?? 999,
      calibrationVersion: (map['calibrationVersion'] as num?)?.toInt() ?? 1,
    );
  }

  static CalibrationProfile fallback() {
    return CalibrationProfile(
      source: 'FALLBACK',
      isStale: false,
      model: '',
      marketingName: 'Unknown Device',
      manufacturer: '',
      zlo: 16.0,
      sensorLinearSlope: 1.0,
      apiLevelMin: 0,
      apiLevelMax: 999,
      calibrationVersion: 1,
    );
  }
}

class NativeHardwareBridge {
  static const _hardwareChannel = MethodChannel('com.vitals.app/camera_hardware');
  static const _cameraChannel = MethodChannel('com.vitals.app/camera_control');
  static const _ppgChannel = MethodChannel('com.vitals.app/ppg_processor');
  static const _calibrationChannel = MethodChannel('com.vitals.app/calibration');

  static Future<void> lockHardwareISP() async {
    try {
      await _hardwareChannel.invokeMethod('applyStrictLocks');
    } on PlatformException catch (e) {
      print("ISP lock failure: ${e.message}");
    }
  }

  static Future<Map<String, dynamic>?> optimizeSensorForPPG({int facing = 0}) async {
    try {
      final result = await _cameraChannel.invokeMethod<Map<dynamic, dynamic>>('configureSafePPG', {'facing': facing});
      if (result != null) {
        return Map<String, dynamic>.from(result);
      }
      return null;
    } on PlatformException catch (e) {
      print("PPG camera optimization failure: ${e.message}");
      return null;
    }
  }

  static Future<double?> processPpgSample(double value) async {
    try {
      final result = await _ppgChannel.invokeMethod<double?>('processPpgSample', {'value': value});
      return result;
    } on PlatformException catch (e) {
      print("PPG processing failure: ${e.message}");
      return null;
    }
  }

  static Future<void> resetPpgBuffer() async {
    try {
      await _ppgChannel.invokeMethod('resetPpgBuffer');
    } on PlatformException catch (e) {
      print("PPG reset failure: ${e.message}");
    }
  }

  // Calibration methods
  static Future<CalibrationProfile?> fetchCalibration() async {
    try {
      final result = await _calibrationChannel.invokeMethod<Map<dynamic, dynamic>>('fetchCalibration');
      if (result != null) {
        return CalibrationProfile.fromMap(result);
      }
      return null;
    } on PlatformException catch (e) {
      print("Calibration fetch failure: ${e.message}");
      return null;
    }
  }

  static Future<CalibrationProfile?> getCurrentCalibration() async {
    try {
      final result = await _calibrationChannel.invokeMethod<Map<dynamic, dynamic>>('getCurrentCalibration');
      if (result != null) {
        return CalibrationProfile.fromMap(result);
      }
      return null;
    } on PlatformException catch (e) {
      print("Get current calibration failure: ${e.message}");
      return null;
    }
  }

  static Future<double?> applyZloCorrection(double value) async {
    try {
      final result = await _calibrationChannel.invokeMethod<double>('applyZloCorrection', {'value': value});
      return result;
    } on PlatformException catch (e) {
      print("ZLO correction failure: ${e.message}");
      return value;
    }
  }

  static Future<CalibrationProfile?> forceRefreshCalibration() async {
    try {
      final result = await _calibrationChannel.invokeMethod<Map<dynamic, dynamic>>('forceRefreshCalibration');
      if (result != null) {
        return CalibrationProfile.fromMap(result);
      }
      return null;
    } on PlatformException catch (e) {
      print("Force refresh calibration failure: ${e.message}");
      return null;
    }
  }

  // Apply ZLO correction locally (more efficient for real-time processing)
  static double applyZloCorrectionLocal(double rawValue, CalibrationProfile profile) {
    if (profile.zlo <= 0.0) return rawValue;
    final corrected = (rawValue - profile.zlo) / profile.sensorLinearSlope;
    return corrected > 0.0 ? corrected : 0.0;
  }

  static List<double> applyZloCorrectionToSignal(List<double> signal, CalibrationProfile profile) {
    if (profile.zlo <= 0.0) return List<double>.from(signal);
    return signal.map((v) => applyZloCorrectionLocal(v, profile)).toList();
  }
}