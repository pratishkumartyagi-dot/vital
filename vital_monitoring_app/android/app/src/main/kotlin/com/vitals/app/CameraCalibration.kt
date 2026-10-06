package com.vitals.app

import android.os.Build

data class CalibrationProfile(
    val model: String,
    val manufacturer: String,
    val zlo: Double,
    val apiLevel: Int
) {
    fun matches(deviceModel: String, deviceManufacturer: String, deviceApiLevel: Int): Boolean {
        return this.model.equals(deviceModel, ignoreCase = true) &&
               this.manufacturer.equals(deviceManufacturer, ignoreCase = true) &&
               this.apiLevel <= deviceApiLevel
    }
}

object CameraCalibration {
    private val CALIBRATION_DATABASE = listOf(
        CalibrationProfile("Pixel 4", "Google", 22.5, Build.VERSION_CODES.Q),
        CalibrationProfile("Pixel 7", "Google", 14.0, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Pixel 7 Pro", "Google", 14.0, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Pixel 7a", "Google", 14.0, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Pixel 8", "Google", 14.0, Build.VERSION_CODES.UPSIDE_DOWN_CAKE),
        CalibrationProfile("Pixel 8 Pro", "Google", 14.0, Build.VERSION_CODES.UPSIDE_DOWN_CAKE),
        CalibrationProfile("Galaxy S22", "Samsung", 19.6, Build.VERSION_CODES.S),
        CalibrationProfile("Galaxy S22+", "Samsung", 19.6, Build.VERSION_CODES.S),
        CalibrationProfile("Galaxy S22 Ultra", "Samsung", 19.6, Build.VERSION_CODES.S),
        CalibrationProfile("Galaxy S23", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Galaxy S23+", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Galaxy S23 Ultra", "Samsung", 19.6, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Galaxy S24", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE),
        CalibrationProfile("Galaxy S24+", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE),
        CalibrationProfile("Galaxy S24 Ultra", "Samsung", 19.6, Build.VERSION_CODES.UPSIDE_DOWN_CAKE),
        CalibrationProfile("Moto G 2022", "Motorola", 14.9, Build.VERSION_CODES.S),
        CalibrationProfile("Moto G Power 2022", "Motorola", 14.9, Build.VERSION_CODES.S),
        CalibrationProfile("Moto G Stylus 2022", "Motorola", 14.9, Build.VERSION_CODES.S),
        CalibrationProfile("OnePlus 10 Pro", "OnePlus", 18.0, Build.VERSION_CODES.S),
        CalibrationProfile("OnePlus 11", "OnePlus", 18.0, Build.VERSION_CODES.TIRAMISU),
        CalibrationProfile("Xiaomi 12", "Xiaomi", 16.5, Build.VERSION_CODES.S),
        CalibrationProfile("Xiaomi 13", "Xiaomi", 16.5, Build.VERSION_CODES.TIRAMISU),
    )

    private const val DEFAULT_ZLO = 16.0

    fun getZloForDevice(model: String, manufacturer: String, apiLevel: Int = Build.VERSION.SDK_INT): Double {
        for (profile in CALIBRATION_DATABASE) {
            if (profile.matches(model, manufacturer, apiLevel)) {
                return profile.zlo
            }
        }
        return DEFAULT_ZLO
    }

    fun getCalibrationInfo(model: String, manufacturer: String, apiLevel: Int = Build.VERSION.SDK_INT): Map<String, Any> {
        val zlo = getZloForDevice(model, manufacturer, apiLevel)
        val matched = CALIBRATION_DATABASE.find { it.matches(model, manufacturer, apiLevel) }
        return mapOf(
            "zlo" to zlo,
            "model" to model,
            "manufacturer" to manufacturer,
            "matched" to (matched != null),
            "profile" to matched?.let { mapOf("model" to it.model, "manufacturer" to it.manufacturer, "zlo" to it.zlo) }
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
}