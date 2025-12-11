package com.example.ekycsimulate.data

import com.example.ekycsimulate.zkp.SchnorrZKP
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import kotlinx.serialization.Serializable

@Serializable
data class EnrollmentResponse(
    val success: Boolean,
    val userId: Int? = null,
    val detail: String? = null
)

@Serializable
data class VerificationResponse(
    val success: Boolean,
    val userId: Int? = null,
    val sessionToken: String? = null,
    val detail: String? = null
)

@Serializable
data class ChallengeResponse(
    val sessionId: String
)

@Serializable
data class LoginPayload(
    val email: String,
    val password: String
)

object ApiService {
    private val client = NetworkModule.client

    suspend fun getChallenge(): Result<String> {
        return try {
            val response: ChallengeResponse = client.get("/api/challenge").body()
            Result.success(response.sessionId)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun login(payload: LoginPayload): Result<VerificationResponse> {
        return try {
            val response = client.post("/api/login") {
                contentType(ContentType.Application.Json)
                setBody(payload)
            }
            if (response.status.value in 200..299) {
                // Reuse VerificationResponse as it contains success + userId + detail
                Result.success(response.body())
            } else {
                 val errorBody = response.body<String>()
                val detail = try {
                    val match = "\"detail\"\\s*:\\s*\"([^\"]+)\"".toRegex().find(errorBody)
                    match?.groupValues?.get(1) ?: errorBody
                } catch (e: Exception) {
                    errorBody
                }
                Result.failure(Exception(detail))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun enroll(payload: SchnorrZKP.EnrollmentPayload): Result<EnrollmentResponse> {
        return try {
            val response = client.post("/api/enroll") {
                contentType(ContentType.Application.Json)
                setBody(payload)
            }
            
            if (response.status.value in 200..299) {
                Result.success(response.body())
            } else {
                // Try to parse error detail
                val errorBody = response.body<String>()
                val detail = try {
                    // Simple regex to extract "detail" from JSON {"detail": "..."}
                    val match = "\"detail\"\\s*:\\s*\"([^\"]+)\"".toRegex().find(errorBody)
                    match?.groupValues?.get(1) ?: errorBody
                } catch (e: Exception) {
                    errorBody
                }
                Result.failure(Exception(detail))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun verify(payload: SchnorrZKP.VerificationPayload, sessionId: String): Result<VerificationResponse> {
        return try {
            val response = client.post("/api/verify") {
                contentType(ContentType.Application.Json)
                url {
                    parameters.append("sessionId", sessionId)
                }
                setBody(payload)
            }
            
            if (response.status.value in 200..299) {
                Result.success(response.body())
            } else {
                val errorBody = response.body<String>()
                val detail = try {
                    val match = "\"detail\"\\s*:\\s*\"([^\"]+)\"".toRegex().find(errorBody)
                    match?.groupValues?.get(1) ?: errorBody
                } catch (e: Exception) {
                    errorBody
                }
                Result.failure(Exception(detail))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
