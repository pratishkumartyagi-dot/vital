package com.vitals.app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import android.hardware.camera2.CaptureRequest

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.vitals.app/camera_hardware"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "applyStrictLocks") {
                // Return success immediately to let Flutter bind hooks
                result.success(true)
            } else {
                result.notImplemented()
            }
        }
    }
}