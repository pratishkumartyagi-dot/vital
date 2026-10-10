package com.vitals.app

import android.os.Build

data class CalibrationProfile(
    val model: String,
    val manufacturer: String,
    val zlo: Double,
    val apiLevel: Int,
    val sensorLinearSlope: Double = 1.0
) {
    fun matches(deviceModel: String, deviceManufacturer: String, deviceApiLevel: Int): Boolean {
        return this.model.equals(deviceModel, ignoreCase = true) &&
               this.manufacturer.equals(deviceManufacturer, ignoreCase = true) &&
               this.apiLevel <= deviceApiLevel
    }
}

object CameraCalibration {
    private val CALIBRATION_DATABASE = listOf(
        CalibrationProfile("Pixel 4", "Google", 22.5, Build.VERSION_CODES.Q, 1.0),
        CalibrationProfile("Pixel 7", "Google", 14.0, Build.VERSION_CODES.TIRAMISU, 1.02),
        CalibrationProfile("Pixel 7 Pro", "Google", 14.0, Build.VERSION_CODES.TIRAMISU, 1.02),
        CalibrationProfile("Pixel 7a", "Google", 14.0, Build.VERSION_CODES.TIRAMISU, 1.02),
        CalibrationProfile("Pixel 8", "Google", 14.0, Build.VERSION_CODES.UPSIDE_DOWN_CAKE, 1.02),
        CalibrationProfile("Pixel 8 Pro", "Google", 14.0, Build.VERSION_CODES.UPSIDE_DOWN_CAKE, 1.02),
        CalibrationProfile("Galaxy S22", "Samsung", 19.6, Build.VERSION_CODES.S, 0.98),
        CalibrationProfile("Galaxy S22+", "Samsung", 19.6, Build.VERSION_CODES.S, 0.98),
        CalibrationProfile("Galaxy S22 Ultra", "Samsung", 19.6, Build.VERSION_CODES.S, 0.98),
        CalibrationProfile("Galaxy S23", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU, 0.98),
        CalibrationProfile("Galaxy S23+", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU, 0.98),
        CalibrationProfile("Galaxy S23 Ultra", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU, 0.98),
        CalibrationProfile("Galaxy S24", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE, 0.98),
        CalibrationProfile("Galaxy S24+", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE, 0.98),
        CalibrationProfile("Galaxy S24 Ultra", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE, 0.98),
        CalibrationProfile("Moto G 2022", "Motorola", 14.9, Build.VERSION_CODES.S, 1.0),
        CalibrationProfile("Moto G Power 2022", "Motorola", 14.9, Build.VERSION_CODES.S, 1.0),
        CalibrationProfile("Moto G Stylus 2022", "Motorola", 14.9, Build.VERSION_CODES.S, 1.0),
        CalibrationProfile("OnePlus 10 Pro", "OnePlus", 18.0, Build.VERSION_CODES.S, 1.0),
        CalibrationProfile("OnePlus 11", "OnePlus", 18.0, Build.VERSION_CODES.TIRAMISU, 1.0),
        CalibrationProfile("Xiaomi 12", "Xiaomi", 16.5, Build.VERSION_CODES.S, 1.0),
        CalibrationProfile("Xiaomi 13", "Xiaomi", 16.5, Build.VERSION_CODES.TIRAMISU, 1.0),
        CalibrationProfile("iPhone 13", "Apple", 12.0, 0, 1.05),
        CalibrationProfile("iPhone 13 Pro", "Apple", 12.0, 0, 1.05),
        CalibrationProfile("iPhone 14", "Apple", 12.0, 0, 1.05),
        CalibrationProfile("iPhone 14 Pro", "Apple", 12.0, 0, 1.05),
        CalibrationProfile("iPhone 15", "Apple", 12.0, 0, 1.05),
        CalibrationProfile("iPhone 15 Pro", "Apple", 12.0, 0, 1.05),
    )

    private const val DEFAULT_ZLO = 16.0
    private const val DEFAULT_SLOPE = 1.0

    fun getZloForDevice(model: String, manufacturer: String, apiLevel: Int = Build.VERSION.SDK_INT): Double {
        for (profile in CALIBRATION_DATABASE) {
            if (profile.matches(model, manufacturer, apiLevel)) {
                return profile.zlo
            }
        }
        return DEFAULT_ZLO
    }

    fun getSlopeForDevice(model: String, manufacturer: String, apiLevel: Int = Build.VERSION.SDK_INT): Double {
        for (profile in CALIBRATION_DATABASE) {
            if (profile.matches(model, manufacturer, apiLevel)) {
                return profile.sensorLinearSlope
            }
        }
        return DEFAULT_SLOPE
    }

    fun getCalibrationInfo(model: String, manufacturer: String, apiLevel: Int = Build.VERSION.SDK_INT): Map<String, Any> {
        val zlo = getZloForDevice(model, manufacturer, apiLevel)
        val slope = getSlopeForDevice(model, manufacturer, apiLevel)
        val matched = CALIBRATION_DATABASE.find { it.matches(model, manufacturer, apiLevel) }
        return mapOf(
            "zlo" to zlo,
            "sensorLinearSlope" to slope,
            "model" to model,
            "manufacturer" to manufacturer,
            "matched" to (matched != null),
            "profile" to matched?.let { mapOf("model" to it.model, "manufacturer" to it.manufacturer, "zlo" to it.zlo, "sensorLinearSlope" to it.sensorLinearSlope) }
        )
    }

    fun applyZloCorrection(rawPixelValue: Double, zlo: Double): Double {
        val corrected = rawPixelValue - zlo
        return if (corrected > 0) corrected else 0.0
    }

    fun applyZloCorrection(signal: DoubleArray, zlo: Double): DoubleArray {
        return signal.map { applyZloCorrection(it, zlo) }.toDoubleArray()
    }

    fun applyZloCorrection(signal: List<Double>, zlo: Double): List<Double> {
        return signal.map { applyZloCorrection(it, zlo) }
    }
    
    fun applyZloCorrectionWithSlope(rawPixelValue: Double, zlo: Double, slope: Double): Double {
        val corrected = (rawPixelValue - zlo) / slope
        return if (corrected > 0) corrected else 0.0
    }
    
    fun applyZloCorrectionWithSlope(signal: DoubleArray, zlo: Double, slope: Double): DoubleArray {
        return signal.map { applyZloCorrectionWithSlope(it, zlo, slope) }.toDoubleArray()
    }
    
    fun applyZloCorrectionWithSlope(signal: List<Double>, zlo: Double, slope: Double): List<Double> {
        return signal.map { applyZloCorrectionWithSlope(it, zlo, slope) }
    }
}