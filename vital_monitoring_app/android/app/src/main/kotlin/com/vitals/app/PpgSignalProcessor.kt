package com.vitals.app

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.PI

class PpgSignalProcessor(
    private val sampleRateHz: Double = 30.0,
    private val windowSizeSeconds: Int = 10
) {
    private val maxBufferSize = (sampleRateHz * windowSizeSeconds).toInt()
    private val rawSignalBuffer = ArrayList<Double>()

    data class BiquadCoeffs(
        val b0: Double,
        val b1: Double,
        val b2: Double,
        val a1: Double,
        val a2: Double
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

        val filtered = applyBandpassFilter(detrended)

        return calculateHeartRate(filtered)
    }

    private fun applyBandpassFilter(input: DoubleArray): DoubleArray {
        val fs = sampleRateHz
        val low = 0.5
        val high = 3.5

        if (input.size < 10 || fs <= 0 || low <= 0 || high <= low || high >= fs / 2) {
            return input.copyOf()
        }

        val hp = highpassBiquad(low, fs)
        val lp = lowpassBiquad(high, fs)
        val dc = input.average()

        // Forward pass: highpass (seeded with DC steady-state) then lowpass
        var y = applyBiquad(input, hp, dc, 0.0)
        y = applyBiquad(y, lp, 0.0, 0.0)

        // Zero-phase (filtfilt): reverse and run forward/backward
        y = y.reversed().toDoubleArray()
        y = applyBiquad(y, hp, 0.0, 0.0)
        y = applyBiquad(y, lp, 0.0, 0.0)
        y = y.reversed().toDoubleArray()

        return y
    }

    private fun lowpassBiquad(fc: Double, fs: Double, q: Double = 1.0 / sqrt(2.0)): BiquadCoeffs {
        val w0 = (2 * PI * fc) / fs
        val cw = cos(w0)
        val sw = sin(w0)
        val alpha = sw / (2 * q)
        val a0 = 1 + alpha
        return BiquadCoeffs(
            b0 = (1 - cw) / 2 / a0,
            b1 = (1 - cw) / a0,
            b2 = (1 - cw) / 2 / a0,
            a1 = (-2 * cw) / a0,
            a2 = (1 - alpha) / a0
        )
    }

    private fun highpassBiquad(fc: Double, fs: Double, q: Double = 1.0 / sqrt(2.0)): BiquadCoeffs {
        val w0 = (2 * PI * fc) / fs
        val cw = cos(w0)
        val sw = sin(w0)
        val alpha = sw / (2 * q)
        val a0 = 1 + alpha
        return BiquadCoeffs(
            b0 = (1 + cw) / 2 / a0,
            b1 = -(1 + cw) / a0,
            b2 = (1 + cw) / 2 / a0,
            a1 = (-2 * cw) / a0,
            a2 = (1 - alpha) / a0
        )
    }

    private fun applyBiquad(input: DoubleArray, c: BiquadCoeffs, seedX: Double, seedY: Double): DoubleArray {
        val out = DoubleArray(input.size)
        var x1 = seedX
        var x2 = seedX
        var y1 = seedY
        var y2 = seedY
        for (i in input.indices) {
            val y = c.b0 * input[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2
            out[i] = y
            x2 = x1
            x1 = input[i]
            y2 = y1
            y1 = y
        }
        return out
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