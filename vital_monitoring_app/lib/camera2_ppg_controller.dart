import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter/rendering.dart';

class Camera2PpgController {
  static const MethodChannel _channel = MethodChannel('com.vitals.app/camera2_ppg');

  static bool _isInitialized = false;
  static bool _isStreaming = false;
  static Function(int, int, Uint8List)? _onFrameCallback;

  static Future<Map<String, dynamic>?> initialize({
    required int facing, // 0 = front, 1 = back
    String resolutionPreset = 'medium',
  }) async {
    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>('initialize', {
        'facing': facing,
        'resolutionPreset': resolutionPreset,
      });

      if (result != null) {
        _isInitialized = true;
        return Map<String, dynamic>.from(result);
      }
      return null;
    } on PlatformException catch (e) {
      print('Camera2PpgController initialize error: ${e.message}');
      return null;
    }
  }

  static Future<bool> startStreaming({Function(int, int, Uint8List)? onFrame}) async {
    if (!_isInitialized) return false;
    if (_isStreaming) return true;

    _onFrameCallback = onFrame;

    if (onFrame != null) {
      _channel.setMethodCallHandler(_handleMethodCall);
    }

    try {
      await _channel.invokeMethod('startStreaming', {'withCallback': onFrame != null});
      _isStreaming = true;
      return true;
    } on PlatformException catch (e) {
      print('Camera2PpgController startStreaming error: ${e.message}');
      return false;
    }
  }

  static Future<void> stopStreaming() async {
    if (!_isStreaming) return;

    try {
      await _channel.invokeMethod('stopStreaming');
      _isStreaming = false;
    } on PlatformException catch (e) {
      print('Camera2PpgController stopStreaming error: ${e.message}');
    }
  }

  static Future<void> setTorch(bool enabled) async {
    try {
      await _channel.invokeMethod('setTorch', {'enabled': enabled});
    } on PlatformException catch (e) {
      print('Camera2PpgController setTorch error: ${e.message}');
    }
  }

  static Future<void> applyPpgSettings(Map<String, dynamic> settings) async {
    try {
      await _channel.invokeMethod('applyPpgSettings', settings);
    } on PlatformException catch (e) {
      print('Camera2PpgController applyPpgSettings error: ${e.message}');
    }
  }

  static Future<void> setZlo(double zlo) async {
    try {
      await _channel.invokeMethod('setZlo', {'zlo': zlo});
    } on PlatformException catch (e) {
      print('Camera2PpgController setZlo error: ${e.message}');
    }
  }

  static Future<void> dispose() async {
    if (_onFrameCallback != null) {
      _channel.setMethodCallHandler(null);
      _onFrameCallback = null;
    }
    try {
      await _channel.invokeMethod('dispose');
      _isInitialized = false;
      _isStreaming = false;
    } on PlatformException catch (e) {
      print('Camera2PpgController dispose error: ${e.message}');
    }
  }

  static Future<dynamic> _handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'onFrame':
        final args = call.arguments as Map<dynamic, dynamic>;
        final width = args['width'] as int;
        final height = args['height'] as int;
        final data = args['data'] as Uint8List;
        _onFrameCallback?.call(width, height, data);
        break;
    }
  }

  static bool get isInitialized => _isInitialized;
  static bool get isStreaming => _isStreaming;
}