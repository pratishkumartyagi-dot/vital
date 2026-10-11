package com.vitals.app

import kotlin.math.sqrt

class PpgSignalProcessor(
    private val sampleRateHz: Double = 30.0,
    private val windowSizeSeconds: Int = 10
) {
    private val maxBufferSize = (sampleRateHz * windowSizeSeconds).toInt()
    private val rawSignalBuffer = ArrayList<Double>()

    // 4th-order Butterworth bandpass filter (0.5-3.5 Hz @ 30 Hz)
    // SOS coefficients from scipy.signal.butter(4, [0.5/15, 3.5/15], btype='band', output='sos')
    // Section 1: [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916]
    // Section 2: [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327]
    private val sos = doubleArrayOf(
        // Section 1: b0, b1, b2, a1, a2
        0.00295136, 0.0, -0.00295136, -1.891408, 0.897916,
        // Section 2: b0, b1, b2, a1, a2
        1.0, 2.0, 1.0, -1.882394, 0.906327
    )

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

        val filtered = applySosFilter(detrended)

        return calculateHeartRate(filtered)
    }

    private fun applySosFilter(input: DoubleArray): DoubleArray {
        var stage = input.copyOf()
        
        // Process two cascaded biquad sections
        for (sectionIdx in 0..1) {
            val offset = sectionIdx * 5
            val b0 = sos[offset]
            val b1 = sos[offset + 1]
            val b2 = sos[offset + 2]
            val a1 = sos[offset + 3]
            val a2 = sos[offset + 4]
            
            val output = DoubleArray(stage.size)
            var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
            
            for (i in stage.indices) {
                val x0 = stage[i]
                val y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                
                output[i] = y0
                x2 = x1; x1 = x0
                y2 = y1; y1 = y0
            }
            stage = output
        }
        
        return stage
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