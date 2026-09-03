package com.vitals.app

import kotlin.math.sqrt

class PpgSignalProcessor(
    private val sampleRateHz: Double = 30.0,
    private val windowSizeSeconds: Int = 10
) {
    private val maxBufferSize = (sampleRateHz * windowSizeSeconds).toInt()
    private val rawSignalBuffer = ArrayList<Double>()

    private val bCoeffs = doubleArrayOf(0.0048, 0.0, -0.0096, 0.0, 0.0048)
    private val aCoeffs = doubleArrayOf(1.0, -3.2815, 4.0906, -2.3168, 0.5103)

    fun addSample(value: Double) {
        if (rawSignalBuffer.size >= maxBufferSize) {
            rawSignalBuffer.removeAt(0)
        }
        rawSignalBuffer.add(value)
    }

    fun reset() {
        rawSignalBuffer.clear()
    }

    fun processWindow(): Double? {
        if (rawSignalBuffer.size < maxBufferSize) return null

        val mean = rawSignalBuffer.average()
        val detrended = rawSignalBuffer.map { it - mean }.toDoubleArray()

        val filtered = applyIirFilter(detrended)

        return calculateHeartRate(filtered)
    }

    private fun applyIirFilter(input: DoubleArray): DoubleArray {
        val output = DoubleArray(input.size)
        val x = DoubleArray(5)
        val y = DoubleArray(5)

        for (i in input.indices) {
            for (j in 4 downTo 1) x[j] = x[j - 1]
            x[0] = input[i]

            var filteredValue = bCoeffs[0] * x[0] + bCoeffs[1] * x[1] + bCoeffs[2] * x[2] + bCoeffs[3] * x[3] + bCoeffs[4] * x[4]
            filteredValue -= aCoeffs[1] * y[1] + aCoeffs[2] * y[2] + aCoeffs[3] * y[3] + aCoeffs[4] * y[4]
            filteredValue /= aCoeffs[0]

            for (j in 4 downTo 1) y[j] = y[j - 1]
            y[0] = filteredValue

            output[i] = filteredValue
        }
        return output
    }

    private fun calculateHeartRate(signal: DoubleArray): Double? {
        val peaks = ArrayList<Int>()
        val mean = signal.average()
        val variance = signal.map { (it - mean) * (it - mean) }.sum() / signal.size
        val stdDev = sqrt(variance)
        val threshold = 0.5 * stdDev

        for (i in 1 until signal.size - 1) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                if (signal[i] > threshold) {
                    peaks.add(i)
                }
            }
        }

        if (peaks.size < 3) return null

        var totalIntervalFrames = 0
        for (k in 0 until peaks.size - 1) {
            totalIntervalFrames += (peaks[k + 1] - peaks[k])
        }

        val averageIntervalFrames = totalIntervalFrames.toDouble() / (peaks.size - 1)
        val averageIntervalSeconds = averageIntervalFrames / sampleRateHz
        val bpm = 60.0 / averageIntervalSeconds

        return if (bpm in 45.0..180.0) bpm else null
    }
}
