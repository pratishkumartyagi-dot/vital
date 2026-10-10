import 'dart:async';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

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

  Map<String, dynamic> toJson() => {
    'source': source,
    'isStale': isStale,
    'model': model,
    'marketingName': marketingName,
    'manufacturer': manufacturer,
    'zlo': zlo,
    'sensorLinearSlope': sensorLinearSlope,
    'apiLevelMin': apiLevelMin,
    'apiLevelMax': apiLevelMax,
    'calibrationVersion': calibrationVersion,
  };
}

class RemoteCalibrationResponse {
  final List<CalibrationProfile> profiles;
  final int version;
  final int minAppVersion;
  final double fallbackZlo;
  final double fallbackSlope;

  RemoteCalibrationResponse({
    required this.profiles,
    required this.version,
    required this.minAppVersion,
    required this.fallbackZlo,
    required this.fallbackSlope,
  });

  factory RemoteCalibrationResponse.fromJson(Map<String, dynamic> json) {
    return RemoteCalibrationResponse(
      profiles: (json['profiles'] as List<dynamic>?)
          ?.map((e) => CalibrationProfile.fromMap(e as Map<dynamic, dynamic>))
          .toList() ?? [],
      version: json['version'] as int? ?? 1,
      minAppVersion: json['min_app_version'] as int? ?? 1,
      fallbackZlo: (json['fallback_zlo'] as num?)?.toDouble() ?? 16.0,
      fallbackSlope: (json['fallback_slope'] as num?)?.toDouble() ?? 1.0,
    );
  }
}

class NativeHardwareBridge {
  static const _hardwareChannel = MethodChannel('com.vitals.app/camera_hardware');
  static const _cameraChannel = MethodChannel('com.vitals.app/camera_control');
  static const _ppgChannel = MethodChannel('com.vitals.app/ppg_processor');
  static const _calibrationChannel = MethodChannel('com.vitals.app/calibration');

  // Remote calibration endpoint
  static const String _calibrationApiUrl = 'https://api.yourdomain.com/v1/calibration';
  
  static CalibrationProfile? _cachedProfile;
  static Timer? _refreshTimer;

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
        _cachedProfile = CalibrationProfile.fromMap(result);
        return _cachedProfile;
      }
      return null;
    } on PlatformException catch (e) {
      print("Calibration fetch failure: ${e.message}");
      return null;
    }
  }

  static Future<CalibrationProfile?> getCurrentCalibration() async {
    if (_cachedProfile != null) return _cachedProfile;
    
    try {
      final result = await _calibrationChannel.invokeMethod<Map<dynamic, dynamic>>('getCurrentCalibration');
      if (result != null) {
        _cachedProfile = CalibrationProfile.fromMap(result);
        return _cachedProfile;
      }
      return null;
    } on PlatformException catch (e) {
      print("Get current calibration failure: ${e.message}");
      return null;
    }
  }

  /// Fetches calibration from remote server with device model matching
  static Future<CalibrationProfile?> fetchRemoteCalibration({
    required String model,
    required String manufacturer,
    required int apiLevel,
  }) async {
    try {
      final uri = Uri.parse('$_calibrationApiUrl/$model');
      final response = await http.get(uri).timeout(const Duration(seconds: 10));
      
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        final profile = CalibrationProfile.fromMap(json);
        
        // Verify device match
        if (profile.model.equals(model, ignoreCase: true) &&
            profile.manufacturer.equals(manufacturer, ignoreCase: true) &&
            apiLevel >= profile.apiLevelMin &&
            apiLevel <= profile.apiLevelMax) {
          _cachedProfile = profile;
          return profile;
        }
      }
      
      // Try fallback list
      final listResponse = await http.get(Uri.parse(_calibrationApiUrl));
      if (listResponse.statusCode == 200) {
        final listJson = jsonDecode(listResponse.body);
        final response = RemoteCalibrationResponse.fromJson(listJson);
        
        for (final profile in response.profiles) {
          if (profile.model.equals(model, ignoreCase: true) &&
              profile.manufacturer.equals(manufacturer, ignoreCase: true) &&
              apiLevel >= profile.apiLevelMin &&
              apiLevel <= profile.apiLevelMax) {
            _cachedProfile = profile;
            return profile;
          }
        }
        
        // Use fallback from server
        _cachedProfile = CalibrationProfile(
          source: 'REMOTE_FALLBACK',
          isStale: false,
          model: model,
          marketingName: '${manufacturer} $model',
          manufacturer: manufacturer,
          zlo: response.fallbackZlo,
          sensorLinearSlope: response.fallbackSlope,
          apiLevelMin: 0,
          apiLevelMax: 999,
          calibrationVersion: response.version,
        );
        return _cachedProfile;
      }
      
      return null;
    } catch (e) {
      print("Remote calibration fetch failed: $e");
      return null;
    }
  }

  /// Initialize calibration at app startup - fetches remote, caches locally
  static Future<CalibrationProfile> initializeCalibration({
    required String model,
    required String manufacturer,
    required int apiLevel,
  }) async {
    // Try remote first
    final remote = await fetchRemoteCalibration(
      model: model,
      manufacturer: manufacturer,
      apiLevel: apiLevel,
    );
    
    if (remote != null) {
      // Start periodic refresh (every 7 days)
      _startPeriodicRefresh(model: model, manufacturer: manufacturer, apiLevel: apiLevel);
      return remote;
    }
    
    // Try local native cache
    final local = await getCurrentCalibration();
    if (local != null) {
      _startPeriodicRefresh(model: model, manufacturer: manufacturer, apiLevel: apiLevel);
      return local;
    }
    
    // Ultimate fallback
    return CalibrationProfile.fallback();
  }

  static void _startPeriodicRefresh({
    required String model,
    required String manufacturer,
    required int apiLevel,
  }) {
    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(const Duration(days: 7), (_) {
      fetchRemoteCalibration(model: model, manufacturer: manufacturer, apiLevel: apiLevel);
    });
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
        _cachedProfile = CalibrationProfile.fromMap(result);
        return _cachedProfile;
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

  static CalibrationProfile? get cachedProfile => _cachedProfile;

  static void dispose() {
    _refreshTimer?.cancel();
    _refreshTimer = null;
    _cachedProfile = null;
  }
}