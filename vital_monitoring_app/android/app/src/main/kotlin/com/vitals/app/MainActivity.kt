package com.vitals.app

import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.HandlerThread
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.vitals.app/camera_hardware"
    private val PPG_CHANNEL = "com.vitals.app/ppg_processor"

    private var ppgHandlerThread: HandlerThread? = null
    private var ppgHandler: Handler? = null
    private val ppgProcessor = PpgSignalProcessor()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "applyStrictLocks") {
                try {
                    val cameraId = "0"
                    val manager = getSystemService(CAMERA_SERVICE) as CameraManager
                    val characteristics = manager.getCameraCharacteristics(cameraId)

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

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, PPG_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "processPpgSample" -> {
                    val value = call.argument<Double>("value")
                    if (value != null) {
                        ppgHandler?.post {
                            ppgProcessor.addSample(value)
                            val bpm = ppgProcessor.processWindow()
                            runOnUiThread {
                                result.success(bpm)
                            }
                        } ?: result.error("THREAD_ERROR", "PPG processing thread not initialized", null)
                    } else {
                        result.error("INVALID_ARG", "Missing PPG sample value", null)
                    }
                }
                "resetPpgBuffer" -> {
                    ppgHandler?.post {
                        ppgProcessor.reset()
                        runOnUiThread {
                            result.success(null)
                        }
                    } ?: result.error("THREAD_ERROR", "PPG processing thread not initialized", null)
                }
                else -> result.notImplemented()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (ppgHandlerThread == null) {
            ppgHandlerThread = HandlerThread("PPGProcessingThread").also { it.start() }
            ppgHandler = Handler(ppgHandlerThread!!.looper)
        }
    }

    override fun onPause() {
        super.onPause()
        ppgHandlerThread?.quitSafely()
        ppgHandlerThread = null
        ppgHandler = null
    }
}
