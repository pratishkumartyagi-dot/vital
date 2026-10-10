package com.vitals.app.camera2

import android.content.Context
import android.hardware.camera2.*
import android.hardware.camera2.params.TonemapCurve
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

class Camera2PpgController(
    private val context: Context,
    private val flutterEngine: FlutterEngine
) {
    private val TAG = "Camera2PpgController"
    private val CHANNEL = "com.vitals.app/camera2_ppg"
    
    private var cameraManager: CameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    
    private val methodChannel: MethodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
    private val settingsCache = ConcurrentHashMap<String, Any>()
    private val calibrationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    private var currentCameraId: String? = null
    private var currentFacing: Int = CameraCharacteristics.LENS_FACING_BACK
    private var isInitialized = false
    private var isStreaming = false
    private var torchEnabled = true
    private var appliedZlo = 16.0
    
    private var frameCallback: ((Int, Int, ByteArray) -> Unit)? = null
    
    init {
        setupMethodChannel()
    }
    
    private fun setupMethodChannel() {
        methodChannel.setMethodCallHandler { call, result ->
            when (call.method) {
                "initialize" -> {
                    val facing = call.argument<Int>("facing") ?: CameraCharacteristics.LENS_FACING_BACK
                    val resolutionPreset = call.argument<String>("resolutionPreset") ?: "medium"
                    initializeCamera(facing, resolutionPreset, result)
                }
                "startStreaming" -> {
                    val callback = call.argument<Boolean>("withCallback") ?: true
                    startStreaming(callback, result)
                }
                "stopStreaming" -> stopStreaming(result)
                "setTorch" -> {
                    val enabled = call.argument<Boolean>("enabled") ?: false
                    setTorch(enabled, result)
                }
                "applyPpgSettings" -> applyPpgSettings(call.arguments as Map<String, Any>?, result)
                "setZlo" -> {
                    val zlo = call.argument<Double>("zlo") ?: 16.0
                    appliedZlo = zlo
                    result.success(true)
                }
                "setFrameCallback" -> {
                    frameCallback = { width, height, data ->
                        methodChannel.invokeMethod("onFrame", mapOf(
                            "width" to width,
                            "height" to height,
                            "data" to data
                        ))
                    }
                    result.success(true)
                }
                "dispose" -> dispose(result)
                else -> result.notImplemented()
            }
        }
    }
    
    private fun initializeCamera(facing: Int, resolutionPreset: String, result: Result) {
        calibrationScope.launch {
            try {
                currentFacing = facing
                val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                    val characteristics = cameraManager.getCameraCharacteristics(id)
                    characteristics.get(CameraCharacteristics.LENS_FACING) == facing
                } ?: throw Exception("No camera found for facing: $facing")
                
                currentCameraId = cameraId
                
                // Query capabilities
                val caps = queryPpgCapabilities(cameraId)
                settingsCache.putAll(caps)
                
                // Apply optimized settings
                val appliedSettings = applyPpgOptimizedSettings(cameraId, facing)
                settingsCache.putAll(appliedSettings)
                
                // Get calibration
                val calibration = CameraCalibration.getCalibrationInfo(
                    Build.MODEL, Build.MANUFACTURER, Build.VERSION.SDK_INT
                )
                appliedZlo = calibration["zlo"] as Double
                settingsCache["calibration"] = calibration
                settingsCache["appliedZlo"] = appliedZlo
                
                // Open camera
                openCamera(cameraId, resolutionPreset)
                
                runOnUiThread { result.success(settingsCache) }
            } catch (e: Exception) {
                Log.e(TAG, "Camera init failed: ${e.message}", e)
                runOnUiThread { result.error("CAMERA_ERROR", e.message, null) }
            }
        }
    }
    
    private fun openCamera(cameraId: String, resolutionPreset: String) {
        val stateCallback = object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createCaptureSession()
            }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
            }
            override fun onError(camera: CameraDevice, error: Int) {
                Log.e(TAG, "Camera error: $error")
            }
        }
        
        startBackgroundThread()
        cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
    }
    
    private fun createCaptureSession() {
        cameraDevice?.createCaptureSession(
            listOf(imageReader?.surface).filterNotNull(),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    startRepeatingRequest()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(TAG, "Capture session configuration failed")
                }
            },
            backgroundHandler
        )
    }
    
    private fun startRepeatingRequest() {
        cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)?.also { builder ->
            captureRequestBuilder = builder
            builder.addTarget(imageReader?.surface!!)
            
            // Apply PPG-optimized settings
            applySettingsToBuilder(builder)
            
            captureSession?.setRepeatingRequest(builder.build(), null, backgroundHandler)
            isStreaming = true
        }
    }
    
    private fun applySettingsToBuilder(builder: CaptureRequest.Builder) {
        val caps = settingsCache
        
        // AE Lock
        if (caps["shouldLockAe"] as? Boolean == true) {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_LOCKED)
        }
        
        // AWB Lock
        if (caps["shouldLockAwb"] as? Boolean == true) {
            builder.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_LOCKED)
        }
        
        // AF Lock
        if (caps["shouldLockAf"] as? Boolean == true) {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_LOCKED)
        }
        
        // Manual exposure
        val targetExposureNs = caps["targetExposureNs"] as? Long
        val targetIso = caps["targetIso"] as? Int
        if (targetExposureNs != null && targetIso != null) {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, targetExposureNs)
            builder.set(CaptureRequest.SENSOR_SENSITIVITY, targetIso)
        }
        
        // Exposure compensation
        val aeCompMin = caps["aeCompMin"] as? Int ?: 0
        if (aeCompMin < 0) {
            builder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, aeCompMin)
        }
        
        // Tonemap curve (linear)
        val tonemapCurve = caps["tonemapCurveRed"] as? TonemapCurve
        if (tonemapCurve != null) {
            builder.set(CaptureRequest.TONEMAP_MODE, CaptureRequest.TONEMAP_MODE_CONTRAST_CURVE)
            builder.set(CaptureRequest.TONEMAP_CURVE_RED, tonemapCurve)
            builder.set(CaptureRequest.TONEMAP_CURVE_GREEN, caps["tonemapCurveGreen"] as TonemapCurve)
            builder.set(CaptureRequest.TONEMAP_CURVE_BLUE, caps["tonemapCurveBlue"] as TonemapCurve)
        }
        
        // Color correction
        val colorCorrectionMode = caps["colorCorrectionMode"] as? Int
        if (colorCorrectionMode != null) {
            builder.set(CaptureRequest.COLOR_CORRECTION_MODE, colorCorrectionMode)
            builder.set(CaptureRequest.COLOR_CORRECTION_GAINS, floatArrayOf(
                (caps["colorCorrectionGainsRed"] as? Float ?: 2.0f),
                (caps["colorCorrectionGainsGreenEven"] as? Float ?: 1.0f),
                (caps["colorCorrectionGainsGreenOdd"] as? Float ?: 1.0f),
                (caps["colorCorrectionGainsBlue"] as? Float ?: 2.0f)
            ))
            builder.set(CaptureRequest.COLOR_CORRECTION_TRANSFORM, intArrayOf(1, 0, 0, 0, 1, 0, 0, 0, 1))
        }
        
        // Flash/torch for rear camera
        if (currentFacing == CameraCharacteristics.LENS_FACING_BACK && torchEnabled) {
            builder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH)
        } else {
            builder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
        }
    }
    
    private fun startStreaming(withCallback: Boolean, result: Result) {
        if (isStreaming) {
            result.success(true)
            return
        }
        
        if (imageReader == null) {
            result.error("NOT_READY", "ImageReader not initialized", null)
            return
        }
        
        if (withCallback && frameCallback != null) {
            imageReader?.setOnImageAvailableListener({ reader ->
                reader.acquireLatestImage()?.use { image ->
                    val yPlane = image.planes[0]
                    val buffer = yPlane.buffer
                    val width = image.width
                    val height = image.height
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    frameCallback?.invoke(width, height, bytes)
                }
            }, backgroundHandler)
        }
        
        startRepeatingRequest()
        result.success(true)
    }
    
    private fun stopStreaming(result: Result) {
        captureSession?.stopRepeating()
        captureSession?.close()
        captureSession = null
        isStreaming = false
        result.success(true)
    }
    
    private fun setTorch(enabled: Boolean, result: Result) {
        torchEnabled = enabled
        captureRequestBuilder?.set(CaptureRequest.FLASH_MODE, 
            if (enabled && currentFacing == CameraCharacteristics.LENS_FACING_BACK) 
                CameraMetadata.FLASH_MODE_TORCH 
            else 
                CameraMetadata.FLASH_MODE_OFF
        )
        captureSession?.setRepeatingRequest(captureRequestBuilder!!.build(), null, backgroundHandler)
        result.success(true)
    }
    
    private fun applyPpgSettings(settings: Map<String, Any>?, result: Result) {
        settings?.let {
            settingsCache.putAll(it)
            captureRequestBuilder?.apply { applySettingsToBuilder(this) }
            captureSession?.setRepeatingRequest(captureRequestBuilder!!.build(), null, backgroundHandler)
        }
        result.success(true)
    }
    
    private fun dispose(result: Result) {
        stopStreaming { _ -> }
        imageReader?.close()
        imageReader = null
        cameraDevice?.close()
        cameraDevice = null
        stopBackgroundThread()
        methodChannel.setMethodCallHandler(null)
        result.success(true)
    }
    
    private fun queryPpgCapabilities(cameraId: String): Map<String, Any> {
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        
        val supportLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORT_LEVEL)
            ?: CameraMetadata.INFO_SUPPORT_LEVEL_LEGACY
        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val supportsManualControl = capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) ?: false
        
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val minIso = isoRange?.lower ?: 100
        val maxIso = isoRange?.upper ?: 800
        
        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val minExposure = exposureRange?.lower ?: 1000000L
        val maxExposure = exposureRange?.upper ?: 30000000L
        
        val aeCompRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val aeCompMin = aeCompRange?.lower ?: 0
        val aeCompMax = aeCompRange?.upper ?: 0
        
        val flashAvailable = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false
        } else false
        
        return mapOf(
            "cameraId" to cameraId,
            "supportLevel" to supportLevel,
            "supportsManualControl" to supportsManualControl,
            "minIso" to minIso,
            "maxIso" to maxIso,
            "minExposure" to minExposure,
            "maxExposure" to maxExposure,
            "aeCompMin" to aeCompMin,
            "aeCompMax" to aeCompMax,
            "flashAvailable" to flashAvailable,
            "facing" to currentFacing
        )
    }
    
    private fun applyPpgOptimizedSettings(cameraId: String, facing: Int): Map<String, Any> {
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        
        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val supportsManualControl = capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) ?: false
        
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val minIso = isoRange?.lower ?: 100
        val maxIso = isoRange?.upper ?: 800
        
        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val minExposure = exposureRange?.lower ?: 1000000L
        val maxExposure = exposureRange?.upper ?: 30000000L
        
        val aeCompRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val aeCompMin = aeCompRange?.lower ?: 0
        val aeCompMax = aeCompRange?.upper ?: 0
        
        val (targetExposure, targetIso) = when (facing) {
            CameraCharacteristics.LENS_FACING_FRONT -> {
                val exp = minExposure + (maxExposure - minExposure) / 3
                val iso = minIso + (maxIso - minIso) / 8
                exp to iso
            }
            else -> {
                val exp = (minExposure + maxExposure) / 2
                val iso = minIso + (maxIso - minIso) / 4
                exp to iso
            }
        }
        
        val targetExposureNs = targetExposure.coerceIn(minExposure, maxExposure)
        val targetIsoVal = targetIso.coerceIn(minIso, maxIso)
        
        val lensFocusDistance = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
        val hasFixedFocus = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
            ?.contains(CameraMetadata.CONTROL_AF_MODE_FIXED) ?: false
        
        val hasFaceDetection = if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
            characteristics.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES)
                ?.contains(CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE) ?: false
        } else false
        
        val shouldDisableFlash = facing == CameraCharacteristics.LENS_FACING_FRONT
        
        val tonemapCurve = createLinearTonemapCurve()
        val supportsTonemapCurve = characteristics.get(CameraCharacteristics.TONEMAP_MAX_CURVE_POINTS) != null
        
        return mapOf(
            "targetExposureNs" to targetExposureNs,
            "targetIso" to targetIsoVal,
            "supportsManualControl" to supportsManualControl,
            "lensFocusDistance" to (lensFocusDistance ?: 0f),
            "hasFixedFocus" to hasFixedFocus,
            "hasFaceDetection" to hasFaceDetection,
            "shouldLockAe" to true,
            "shouldLockAwb" to true,
            "shouldLockAf" to true,
            "shouldDisableFlash" to shouldDisableFlash,
            "aeCompMin" to aeCompMin,
            "aeCompMax" to aeCompMax,
            "tonemapMode" to CaptureRequest.TONEMAP_MODE_CONTRAST_CURVE,
            "tonemapCurveRed" to tonemapCurve,
            "tonemapCurveGreen" to tonemapCurve,
            "tonemapCurveBlue" to tonemapCurve,
            "supportsTonemapCurve" to supportsTonemapCurve,
            "colorCorrectionMode" to CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX,
            "colorCorrectionGainsRed" to 2.0f,
            "colorCorrectionGainsGreenEven" to 1.0f,
            "colorCorrectionGainsGreenOdd" to 1.0f,
            "colorCorrectionGainsBlue" to 2.0f,
            "colorCorrectionTransform" to intArrayOf(1, 0, 0, 0, 1, 0, 0, 0, 1)
        )
    }
    
    private fun createLinearTonemapCurve(): TonemapCurve {
        val points = floatArrayOf(0f, 0f, 1f, 1f)
        return TonemapCurve(points, points, points)
    }
    
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera2Background").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }
    
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        backgroundThread = null
        backgroundHandler = null
    }
    
    private fun runOnUiThread(action: () -> Unit) {
        Handler(context.mainLooper).post(action)
    }
}