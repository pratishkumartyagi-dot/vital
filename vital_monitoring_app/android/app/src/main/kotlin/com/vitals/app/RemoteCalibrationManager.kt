package com.vitals.app

import android.content.Context
import android.os.Build
import android.util.Log
import io.ktor.client.*
import io.ktor.client.engine.okhttp.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.util.concurrent.TimeUnit

@Serializable
data class RemoteCalibrationProfile(
    val model: String,
    val marketingName: String,
    val manufacturer: String,
    val zlo: Double,
    val sensorLinearSlope: Double,
    val apiLevelMin: Int,
    val apiLevelMax: Int,
    @SerialName("last_updated") val lastUpdated: String,
    @SerialName("calibration_version") val calibrationVersion: Int
)

@Serializable
data class RemoteCalibrationResponse(
    val profiles: List<RemoteCalibrationProfile>,
    val version: Int,
    @SerialName("min_app_version") val minAppVersion: Int,
    @SerialName("fallback_zlo") val fallbackZlo: Double,
    @SerialName("fallback_slope") val fallbackSlope: Double
)

class RemoteCalibrationManager(
    private val context: Context,
    private val baseUrl: String = "https://api.yourdomain.com/v1/calibration",
    private val timeoutMs: Long = 10000
) {
    private val TAG = "RemoteCalibrationManager"
    private val client = HttpClient(OkHttp) {
        engine {
            connectTimeout = timeoutMs.toLong()
            requestTimeout = timeoutMs.toLong()
        }
        install(ContentNegotiation) {
            json(json = KotlinxSerializer(Json { ignoreUnknownKeys = true }))
        }
        install(Logging) {
            level = LogLevel.HEADERS
        }
    }

    private val localCache = CalibrationCache(context)
    private var _currentProfile: RemoteCalibrationProfile? = null
    private val _updateJob = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.IO + _updateJob)

    data class CalibrationResult(
        val profile: RemoteCalibrationProfile,
        val source: Source,
        val isStale: Boolean
    ) {
        enum class Source { LOCAL_CACHE, REMOTE, FALLBACK }
    }

    fun getCalibration(): CalibrationResult {
        return scope.awaitOrNull {
            localCache.getCachedProfile().let { cached ->
                if (cached != null && cached.profile.matchesDevice()) {
                    val isStale = cached.isExpired()
                    if (!isStale) {
                        _currentProfile = cached.profile
                        CalibrationResult(cached.profile, CalibrationResult.Source.LOCAL_CACHE, false)
                    } else {
                        // Trigger background refresh
                        scope.launch { fetchAndCacheRemote() }
                        CalibrationResult(cached.profile, CalibrationResult.Source.LOCAL_CACHE, true)
                    }
                } else {
                    // No valid cache, try remote
                    fetchAndCacheRemote()
                }
            }
        } ?: getFallbackResult()
    }

    suspend fun fetchAndCacheRemote(): CalibrationResult {
        return try {
            val response = client.get<RemoteCalibrationResponse>(baseUrl)
            val profile = response.profiles.find { it.matchesDevice() }
                ?: run {
                    Log.w(TAG, "No matching profile for ${Build.MODEL} from remote")
                    return@try getFallbackResult()
                }

            localCache.cacheProfile(profile)
            _currentProfile = profile
            Log.i(TAG, "Remote calibration fetched for ${profile.marketingName}")
            CalibrationResult(profile, CalibrationResult.Source.REMOTE, false)
        } catch (e: Exception) {
            Log.e(TAG, "Remote fetch failed: ${e.message}", e)
            val cached = localCache.getCachedProfile()
            if (cached != null) {
                _currentProfile = cached.profile
                CalibrationResult(cached.profile, CalibrationResult.Source.LOCAL_CACHE, true)
            } else {
                getFallbackResult()
            }
        }
    }

    fun getCurrentProfile(): RemoteCalibrationProfile? = _currentProfile

    suspend fun forceRefresh(): CalibrationResult = fetchAndCacheRemote()

    fun shutdown() {
        _updateJob.cancel()
        client.close()
    }

    private fun getFallbackResult(): CalibrationResult {
        val fallback = RemoteCalibrationProfile(
            model = Build.MODEL,
            marketingName = "Unknown (${Build.MANUFACTURER} ${Build.MODEL})",
            manufacturer = Build.MANUFACTURER,
            zlo = 16.0,
            sensorLinearSlope = 1.0,
            apiLevelMin = 0,
            apiLevelMax = Int.MAX_VALUE,
            lastUpdated = "",
            calibrationVersion = 1
        )
        _currentProfile = fallback
        return CalibrationResult(fallback, CalibrationResult.Source.FALLBACK, false)
    }

    fun applyZloCorrection(rawPixelIntensity: Double): Double {
        val profile = _currentProfile
            ?: RemoteCalibrationProfile(
                model = Build.MODEL,
                marketingName = "Unknown",
                manufacturer = Build.MANUFACTURER,
                zlo = 16.0,
                sensorLinearSlope = 1.0,
                apiLevelMin = 0,
                apiLevelMax = Int.MAX_VALUE,
                lastUpdated = "",
                calibrationVersion = 1
            )
        
        val corrected = (rawPixelIntensity - profile.zlo) / profile.sensorLinearSlope
        return maxOf(0.0, corrected)
    }

    fun applyZloCorrection(signal: DoubleArray): DoubleArray {
        val profile = _currentProfile
            ?: return signal.map { maxOf(0.0, (it - 16.0) / 1.0) }.toDoubleArray()
        
        return signal.map { maxOf(0.0, (it - profile.zlo) / profile.sensorLinearSlope) }.toDoubleArray()
    }
}

private class CalibrationCache(private val context: Context) {
    private val prefs = context.getSharedPreferences("calibration_cache", Context.MODE_PRIVATE)
    private val json = Json { ignoreUnknownKeys = true }

    data class CachedProfile(
        val profile: RemoteCalibrationProfile,
        val timestamp: Long
    )

    private const val CACHE_DURATION_MS = 7 * 24 * 60 * 60 * 1000 // 7 days

    fun cacheProfile(profile: RemoteCalibrationProfile) {
        val cached = CachedProfile(profile, System.currentTimeMillis())
        val jsonString = json.encodeToString(cached)
        prefs.edit().putString("cached_profile", jsonString).apply()
    }

    fun getCachedProfile(): CachedProfile? {
        val jsonString = prefs.getString("cached_profile", null) ?: return null
        return try {
            json.decodeFromString(jsonString)
        } catch (e: Exception) {
            null
        }
    }

    inner class CachedProfile(
        val profile: RemoteCalibrationProfile,
        val timestamp: Long
    ) {
        fun isExpired(): Boolean = System.currentTimeMillis() - timestamp > CACHE_DURATION_MS

        fun matchesDevice(): Boolean = profile.matchesDevice()
    }
}

private fun CoroutineScope.awaitOrNull(block: suspend () -> RemoteCalibrationManager.CalibrationResult): RemoteCalibrationManager.CalibrationResult? {
    val deferred = async { block() }
    return try {
        deferred.await()
    } catch (e: Exception) {
        null
    }
}

extension RemoteCalibrationProfile {
    fun matchesDevice(): Boolean {
        return this.model.equals(Build.MODEL, ignoreCase = true) &&
               this.manufacturer.equals(Build.MANUFACTURER, ignoreCase = true) &&
               Build.VERSION.SDK_INT in apiLevelMin..apiLevelMax
    }
}