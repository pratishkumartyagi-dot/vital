package com.vitals.app

import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.vitals.app/camera_hardware"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "applyStrictLocks") {
                try {
                    val cameraId = "0"
                    val manager = getSystemService(CAMERA_SERVICE) as CameraManager
                    val characteristics = manager.getCameraCharacteristics(cameraId)

                    // Build a strict capture request disabling auto controls
                    val builder = manager.createCaptureRequest(CameraCharacteristics.TEMPLATE_PREVIEW) as android.hardware.camera2.CaptureRequest.Builder
                    builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                    builder.set(CaptureRequest.SENSOR_SENSITIVITY, 400)
                    builder.set(CaptureRequest.CONTROL_EFFECT_MODE, CaptureRequest.CONTROL_EFFECT_MODE_OFF)

                    result.success(true)
                } catch (e: Exception) {
                    result.error("LOCK_FAILED", "Hardware ISP lock failed: ${e.message}", null)
                }
            } else {
                result.notImplemented()
            }
        }
    }
}
