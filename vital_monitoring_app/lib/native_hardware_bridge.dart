import 'package:flutter/services.dart';

class NativeHardwareBridge {
  static const _channel = MethodChannel('com.vitals.app/camera_hardware');

  static Future<void> lockHardwareISP() async {
    try {
      await _channel.invokeMethod('applyStrictLocks');
    } on PlatformException catch (e) {
      print("ISP lock failure: ${e.message}");
    }
  }
}