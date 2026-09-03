import 'package:flutter/services.dart';

class NativeHardwareBridge {
  static const _hardwareChannel = MethodChannel('com.vitals.app/camera_hardware');
  static const _ppgChannel = MethodChannel('com.vitals.app/ppg_processor');

  static Future<void> lockHardwareISP() async {
    try {
      await _hardwareChannel.invokeMethod('applyStrictLocks');
    } on PlatformException catch (e) {
      print("ISP lock failure: ${e.message}");
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
}