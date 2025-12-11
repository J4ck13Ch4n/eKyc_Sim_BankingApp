package com.example.ekycsimulate.zkp

import android.content.Context
import android.content.SharedPreferences
import com.example.ekycsimulate.ui.auth.IdCardInfo
import org.json.JSONObject
import java.math.BigInteger

/**
 * Manages ZKP enrollment process
 * Coordinates key generation, PII encryption, and proof generation
 */
class ZKPEnrollmentManager(private val context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences("zkp_keys", Context.MODE_PRIVATE)
    
    /**
     * Complete enrollment data
     */
    data class EnrollmentData(
        val payload: SchnorrZKP.EnrollmentPayload,
        val privateKey: BigInteger  // Keep this secure on device!
    )

    /**
     * Perform complete enrollment process
     * 
     * @param idCardInfo OCR extracted ID card information
     * @param fullName User's full name
     * @param phoneNumber User's phone number
     * @param address User's address
     * @param faceImageApproval Approval status from face scan (1 = approved)
     * @return Enrollment data ready to send to server
     */
    fun performEnrollment(
        idCardInfo: IdCardInfo,
        fullName: String,
        phoneNumber: String,
        address: String,
        faceImageApproval: Int
    ): EnrollmentData {
        
        // 1. Generate key pair
        val keyPair = SchnorrZKP.generateKeyPair()
        
        // 2. Create PII JSON
        val piiJson = JSONObject().apply {
            put("idNumber", idCardInfo.idNumber)
            put("fullName", fullName)
            put("phoneNumber", phoneNumber)
            put("address", address)
            put("dob", idCardInfo.dob)
            put("origin", idCardInfo.origin)
            put("faceApproval", faceImageApproval)
        }.toString()
        
        // 3. Encrypt PII
        val encryptedPII = PIIEncryption.encrypt(piiJson)
        
        // 4. Compute ID hash for uniqueness check (1 ID = 1 account)
        val idNumberHash = SchnorrZKP.sha256(idCardInfo.idNumber)
        
        // 5. Compute commitment (public identifier)
        // commitment = Hash(publicKey || idNumberHash)
        val commitmentData = SchnorrZKP.pointToHex(keyPair.publicKey) + idNumberHash
        val commitment = SchnorrZKP.sha256(commitmentData)
        
        // 6. Generate ZKP enrollment proof
        // IMPORTANT: Message binds OCR data and approval to the proof
        // This proves the data came from OCR module and Face Scan passed
        val currentTimestamp = System.currentTimeMillis()
        
        val enrollmentMessage = buildEnrollmentMessage(
            commitment = commitment,
            idNumberHash = idNumberHash,
            fullNameHash = SchnorrZKP.sha256(fullName),
            dobHash = SchnorrZKP.sha256(idCardInfo.dob),
            approval = faceImageApproval,
            timestamp = currentTimestamp
        )
        
        val proof = SchnorrZKP.generateProof(keyPair.privateKey, enrollmentMessage)
        
        // 7. Create payload
        val payload = SchnorrZKP.EnrollmentPayload(
            publicKey = SchnorrZKP.pointToHex(keyPair.publicKey),
            commitment = commitment,
            idNumberHash = idNumberHash,
            encryptedPII = PIIEncryption.encryptedDataToJson(encryptedPII),
            proof = SchnorrZKP.ProofData(
                commitmentR = SchnorrZKP.pointToHex(proof.commitment),
                challenge = SchnorrZKP.bigIntToHex(proof.challenge),
                response = SchnorrZKP.bigIntToHex(proof.response)
            ),
            timestamp = currentTimestamp,
            // Binding data proves OCR extraction and Face Scan approval
            fullNameHash = SchnorrZKP.sha256(fullName),
            dobHash = SchnorrZKP.sha256(idCardInfo.dob),
            approval = faceImageApproval
        )
        
        // 8. Store private key securely on device
        savePrivateKey(keyPair.privateKey)
        
        // 9. Store binding data for future verification
        saveBindingData(SchnorrZKP.sha256(idCardInfo.dob), idNumberHash, fullName)
        
        return EnrollmentData(payload, keyPair.privateKey)
    }

    /**
     * Build enrollment message that binds OCR data and approval
     * This message is used in ZKP proof generation
     */
    private fun buildEnrollmentMessage(
        commitment: String,
        idNumberHash: String,
        fullNameHash: String,
        dobHash: String,
        approval: Int,
        timestamp: Long
    ): String {
        return "ENROLL:" +
                "commitment:$commitment:" +
                "id:$idNumberHash:" +
                "name:$fullNameHash:" +
                "dob:$dobHash:" +
                "approval:$approval:" +
                "ts:$timestamp"
    }

    /**
     * Perform verification (subsequent login)
     * 
     * @param sessionId Server-provided session ID
     * @return Verification payload
     */
    fun performVerification(sessionId: String): SchnorrZKP.VerificationPayload? {
        // 1. Load private key
        val privateKey = loadPrivateKey() ?: return null
        
        // 2. Compute public key
        val publicKey = SchnorrZKP.computePublicKey(privateKey)
        
        // 3. Generate nullifier (prevents replay attacks)
        val nullifier = SchnorrZKP.generateNullifier(privateKey, sessionId)
        
        // 4. Generate verification proof
        val currentTimestamp = System.currentTimeMillis()
        val verificationMessage = "VERIFY:$sessionId:$currentTimestamp"
        val proof = SchnorrZKP.generateProof(privateKey, verificationMessage)
        
        // 5. Create payload
        return SchnorrZKP.VerificationPayload(
            publicKey = SchnorrZKP.pointToHex(publicKey),
            proof = SchnorrZKP.ProofData(
                commitmentR = SchnorrZKP.pointToHex(proof.commitment),
                challenge = SchnorrZKP.bigIntToHex(proof.challenge),
                response = SchnorrZKP.bigIntToHex(proof.response)
            ),
            nullifier = nullifier,
            timestamp = currentTimestamp
        )
    }

    /**
     * Save private key to encrypted SharedPreferences
     * Uses Android Keystore (Key Wrapping) for security
     */
    private fun savePrivateKey(privateKey: BigInteger) {
        val hexKey = SchnorrZKP.bigIntToHex(privateKey)
        val encryptedData = PIIEncryption.encrypt(hexKey, PIIEncryption.ZKP_KEY_ALIAS)
        val json = PIIEncryption.encryptedDataToJson(encryptedData)
        
        prefs.edit()
            .putString("private_key", json)
            .apply()
    }

    /**
     * Load private key from storage
     * Decrypts using Android Keystore
     */
    private fun loadPrivateKey(): BigInteger? {
        val storedValue = prefs.getString("private_key", null) ?: return null
        
        return try {
            // Check if it's the new encrypted format (JSON)
            if (storedValue.trim().startsWith("{")) {
                val encryptedData = PIIEncryption.jsonToEncryptedData(storedValue)
                val hexKey = PIIEncryption.decrypt(encryptedData, PIIEncryption.ZKP_KEY_ALIAS)
                SchnorrZKP.hexToBigInt(hexKey)
            } else {
                // Legacy support or invalid data
                // For security, we might want to force re-enrollment, but for now return null
                null
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Check if user is already enrolled
     */
    fun isEnrolled(): Boolean {
        return prefs.contains("private_key")
    }

    /**
     * Clear enrollment data (logout)
     */
    fun clearEnrollment() {
        prefs.edit().clear().apply()
    }

    /**
     * Save binding data (id hash, name, dob) for verification
     */
    private fun saveBindingData(faceImageHash: String, idNumberHash: String, fullName: String) {
        prefs.edit()
            .putString("id_hash", idNumberHash)
            .putString("name_hash", SchnorrZKP.sha256(fullName))
            .putString("dob_hash", faceImageHash) // Reuse parameter name but store dobHash
            .apply()
    }

    /**
     * Get binding data for verification
     */
    fun getBindingData(): Map<String, String>? {
        val faceHash = prefs.getString("face_hash", null)
        val idHash = prefs.getString("id_hash", null)
        val nameHash = prefs.getString("name_hash", null)
        
        return if (faceHash != null && idHash != null && nameHash != null) {
            mapOf(
                "faceHash" to faceHash,
                "idHash" to idHash,
                "nameHash" to nameHash
            )
        } else null
    }

    /**
     * Convert enrollment payload to JSON for API
     */
    fun enrollmentPayloadToJson(payload: SchnorrZKP.EnrollmentPayload): String {
        return JSONObject().apply {
            put("publicKey", payload.publicKey)
            put("commitment", payload.commitment)
            put("idNumberHash", payload.idNumberHash)
            put("encryptedPII", payload.encryptedPII)
            put("proof", JSONObject().apply {
                put("commitmentR", payload.proof.commitmentR)
                put("challenge", payload.proof.challenge)
                put("response", payload.proof.response)
            })
            put("timestamp", payload.timestamp)
            // Binding data (OCR + Face Scan approval)
            put("fullNameHash", payload.fullNameHash)
            put("dobHash", payload.dobHash)
            put("approval", payload.approval)
            // Auth fields
            put("email", payload.email)
            put("password", payload.password)
        }.toString(2) // Pretty print with indent
    }

    /**
     * Convert verification payload to JSON for API
     */
    fun verificationPayloadToJson(payload: SchnorrZKP.VerificationPayload): String {
        return JSONObject().apply {
            put("publicKey", payload.publicKey)
            put("proof", JSONObject().apply {
                put("commitmentR", payload.proof.commitmentR)
                put("challenge", payload.proof.challenge)
                put("response", payload.proof.response)
            })
            put("nullifier", payload.nullifier)
            put("timestamp", payload.timestamp)
        }.toString(2)
    }
    /**
     * Send enrollment payload to server
     */
    suspend fun sendEnrollment(payload: SchnorrZKP.EnrollmentPayload): Result<com.example.ekycsimulate.data.EnrollmentResponse> {
        val jsonPayload = enrollmentPayloadToJson(payload)
        android.util.Log.d("ZKP_PAYLOAD", "Client Sending Payload: $jsonPayload")
        return com.example.ekycsimulate.data.ApiService.enroll(payload)
    }
    /**
     * Send verification payload to server
     */
    suspend fun sendVerification(payload: SchnorrZKP.VerificationPayload, sessionId: String): Result<com.example.ekycsimulate.data.VerificationResponse> {
        return com.example.ekycsimulate.data.ApiService.verify(payload, sessionId)
    }

    /**
     * Get challenge (session ID) from server
     */
    suspend fun getChallenge(): Result<String> {
        return com.example.ekycsimulate.data.ApiService.getChallenge()
    }
}
