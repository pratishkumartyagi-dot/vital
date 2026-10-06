package com.vitals.calibration

import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.*
import io.ktor.server.plugins.logging.*
import io.ktor.server.plugins.statuspages.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.io.File

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

fun main() {
    val calibrationData = loadCalibrationDatabase()
    
    embeddedServer(Netty, port = 8080, host = "0.0.0.0") {
        install(ContentNegotiation) {
            json()
        }
        install(CORS) {
            anyHost()
        }
        install(CallLogging)
        install(StatusPages) {
            exception<Throwable> { call, cause ->
                call.respondText("Internal Server Error: ${cause.message}", status = io.ktor.http.HttpStatusCode.InternalServerError)
            }
        }
        
        routing {
            get("/v1/calibration") {
                call.respond(calibrationData)
            }
            
            get("/v1/calibration/{model}") {
                val model = call.parameters["model"] ?: return@get call.respondText("Model required", status = io.ktor.http.HttpStatusCode.BadRequest)
                val profile = calibrationData.profiles.find { it.model.equals(model, ignoreCase = true) }
                if (profile != null) {
                    call.respond(profile)
                } else {
                    call.respondText("Not found", status = io.ktor.http.HttpStatusCode.NotFound)
                }
            }
            
            get("/health") {
                call.respondText("OK")
            }
        }
    }.start(wait = true)
}

fun loadCalibrationDatabase(): RemoteCalibrationResponse {
    val file = File("calibration_database.json")
    if (!file.exists()) {
        throw IllegalStateException("calibration_database.json not found")
    }
    val json = file.readText()
    return Json { ignoreUnknownKeys = true }.decodeFromString(json)
}