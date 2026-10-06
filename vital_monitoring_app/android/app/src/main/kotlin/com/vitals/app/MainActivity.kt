package com.vitals.app

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.TonemapCurve
import android.os.Build
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlin.math.max

class MainActivity : FlutterActivity() {
    private val PPG_CHANNEL = "com.vitals.app/ppg_processor"
    private val CAMERA_CHANNEL = "com.vitals.app/camera_control"
    private val CALIBRATION_CHANNEL = "com.vitals.app/calibration"

    private var ppgHandlerThread: android.os.HandlerThread? = null
    private var ppgHandler: android.os.Handler? = null
    private var ppgProcessor = PpgSignalProcessor()
    
    private val calibrationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var remoteCalibrationManager: RemoteCalibrationManager? = null
    private var currentCalibrationResult: RemoteCalibrationManager.CalibrationResult? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        remoteCalibrationManager = RemoteCalibrationManager(this)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CAMERA_CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "configureSafePPG") {
                try {
                    val facing = call.argument<Int>("facing") ?: CameraCharacteristics.LENS_FACING_BACK
                    val caps = queryPpgCapabilities(facing)
                    if (caps != null) {
                        val appliedSettings = applyPpgOptimizedSettings(caps["cameraId"] as String, facing)
                        val calibration = CameraCalibration.getCalibrationInfo(
                            Build.MODEL, Build.MANUFACTURER, Build.VERSION.SDK_INT
                        )
                        // Create PPG processor with calibration
                        val zlo = calibration["zlo"] as Double
                        ppgProcessor = PpgSignalProcessor(30.0, 10, zlo)
                        
                        val response = caps.toMutableMap()
                        response["ppgOptimizedSettings"] = appliedSettings
                        response["calibration"] = calibration
                        result.success(response)
                    } else {
                        result.error("CAMERA_ERROR", "Could not query camera capabilities for facing: $facing", null)
                    }
                } catch (e: Exception) {
                    result.error("CAMERA_ERROR", "Exception: ${e.message}", null)
                }
            } else {
                result.notImplemented()
            }
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CALIBRATION_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "fetchCalibration" -> {
                    calibrationScope.launch {
                        val calibrationResult = remoteCalibrationManager?.fetchAndCacheRemote()
                        currentCalibrationResult = calibrationResult
                        if (calibrationResult != null) {
                            runOnUiThread {
                                val profile = calibrationResult.profile
                                val response = mapOf(
                                    "source" to calibrationResult.source.name,
                                    "isStale" to calibrationResult.isStale,
                                    "model" to profile.model,
                                    "marketingName" to profile.marketingName,
                                    "manufacturer" to profile.manufacturer,
                                    "zlo" to profile.zlo,
                                    "sensorLinearSlope" to profile.sensorLinearSlope,
                                    "apiLevelMin" to profile.apiLevelMin,
                                    "apiLevelMax" to profile.apiLevelMax,
                                    "calibrationVersion" to profile.calibrationVersion
                                )
                                result.success(response)
                            }
                        } else {
                            runOnUiThread {
                                result.error("CALIBRATION_ERROR", "Failed to fetch calibration", null)
                            }
                        }
                    }
                }
                "getCurrentCalibration" -> {
                    val calibrationResult = currentCalibrationResult ?: remoteCalibrationManager?.getCalibration()
                    if (calibrationResult != null) {
                        val profile = calibrationResult.profile
                        val response = mapOf(
                            "source" to calibrationResult.source.name,
                            "isStale" to calibrationResult.isStale,
                            "model" to profile.model,
                            "marketingName" to profile.marketingName,
                            "manufacturer" to profile.manufacturer,
                            "zlo" to profile.zlo,
                            "sensorLinearSlope" to profile.sensorLinearSlope,
                            "apiLevelMin" to profile.apiLevelMin,
                            "apiLevelMax" to profile.apiLevelMax,
                            "calibrationVersion" to profile.calibrationVersion
                        )
                        result.success(response)
                    } else {
                        result.error("CALIBRATION_ERROR", "No calibration available", null)
                    }
                }
                "applyZloCorrection" -> {
                    val value = call.argument<Double>("value")
                    if (value != null) {
                        val corrected = remoteCalibrationManager?.applyZloCorrection(value) ?: value
                        result.success(corrected)
                    } else {
                        result.error("INVALID_ARG", "Missing value", null)
                    }
                }
                "forceRefreshCalibration" -> {
                    calibrationScope.launch {
                        val calibrationResult = remoteCalibrationManager?.forceRefresh()
                        currentCalibrationResult = calibrationResult
                        if (calibrationResult != null) {
                            runOnUiThread {
                                val profile = calibrationResult.profile
                                val response = mapOf(
                                    "source" to calibrationResult.source.name,
                                    "isStale" to calibrationResult.isStale,
                                    "model" to profile.model,
                                    "marketingName" to profile.marketingName,
                                    "manufacturer" to profile.manufacturer,
                                    "zlo" to profile.zlo,
                                    "sensorLinearSlope" to profile.sensorLinearSlope,
                                    "calibrationVersion" to profile.calibrationVersion
                                )
                                result.success(response)
                            }
                        } else {
                            runOnUiThread {
                                result.error("CALIBRATION_ERROR", "Failed to refresh calibration", null)
                            }
                        }
                    }
                }
                else -> result.notImplemented()
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

    private fun queryPpgCapabilities(facing: Int = CameraCharacteristics.LENS_FACING_BACK): Map<String, Any>? {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = manager.cameraIdList.firstOrNull { id ->
            val characteristics = manager.getCameraCharacteristics(id)
            characteristics.get(CameraCharacteristics.LENS_FACING) == facing
        } ?: return null

        val characteristics = manager.getCameraCharacteristics(cameraId)

        val supportLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORT_LEVEL)
            ?: CameraMetadata.INFO_SUPPORT_LEVEL_LEGACY

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val supportsManualControl = capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
            ?: false

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
            "facing" to facing
        )
    }

    private fun applyPpgOptimizedSettings(cameraId: String, facing: Int = CameraCharacteristics.LENS_FACING_BACK): Map<String, Any> {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val supportsManualControl = capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
            ?: false

        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val minIso = isoRange?.lower ?: 100
        val maxIso = isoRange?.upper ?: 800

        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val minExposure = exposureRange?.lower ?: 1000000L
        val maxExposure = exposureRange?.upper ?: 30000000L

        val aeCompRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val aeCompMin = aeCompRange?.lower ?: 0
        val aeCompMax = aeCompRange?.upper ?: 0

        // Front camera (rPPG): lower ISO, longer exposure for skin-tone stability
        // Rear camera (PPG): slightly higher ISO, shorter exposure for fingertip
        val (targetExposure, targetIso) = when (facing) {
            CameraCharacteristics.LENS_FACING_FRONT -> {
                val exp = minExposure + (maxExposure - minExposure) / 3  // ~33% range
                val iso = minIso + (maxIso - minIso) / 8  // ~12% range (low ISO)
                exp to iso
            }
            else -> {
                val exp = (minExposure + maxExposure) / 2  // ~50% range
                val iso = minIso + (maxIso - minIso) / 4  // ~25% range
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

    override fun onResume() {
        super.onResume()
        if (ppgHandlerThread == null) {
            ppgHandlerThread = android.os.HandlerThread("PPGProcessingThread").also { it.start() }
            ppgHandler = android.os.Handler(ppgHandlerThread!!.looper)
        }
    }

    override fun onPause() {
        super.onPause()
        ppgHandlerThread?.quitSafely()
        ppgHandlerThread = null
        ppgHandler = null
    }

    override fun onDestroy() {
        super.onDestroy()
        calibrationScope.cancel()
        remoteCalibrationManager?.shutdown()
    }
}
