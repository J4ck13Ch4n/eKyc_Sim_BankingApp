package com.example.ekycsimulate.zkp

import org.bouncycastle.jce.ECNamedCurveTable
import org.bouncycastle.jce.spec.ECNamedCurveParameterSpec
import org.bouncycastle.math.ec.ECPoint
import java.math.BigInteger
import java.security.MessageDigest
import java.security.SecureRandom
import kotlinx.serialization.Serializable

/**
 * True Zero-Knowledge Proof implementation using Schnorr Protocol
 * Based on Elliptic Curve Cryptography (secp256k1)
 * 
 * This is a REAL ZKP where:
 * - Prover proves knowledge of private key without revealing it
 * - Verifier only needs public key to verify
 * - Server NEVER knows the private key
 */
object SchnorrZKP {
    
    private val curveParams: ECNamedCurveParameterSpec = ECNamedCurveTable.getParameterSpec("secp256k1")
    private val G: ECPoint = curveParams.g
    private val n: BigInteger = curveParams.n
    private val secureRandom = SecureRandom()

    /**
     * Key Pair for identity
     */
    data class KeyPair(
        val privateKey: BigInteger,  // Secret (never leaves device)
        val publicKey: ECPoint       // Public (sent to server)
    )

    /**
     * ZKP Proof structure
     */
    data class SchnorrProof(
        val commitment: ECPoint,     // R = k*G
        val challenge: BigInteger,   // c = Hash(R || P || message)
        val response: BigInteger     // s = k + c*x (mod n)
    )

    /**
     * Enrollment payload to send to server
     */
    @Serializable
    data class EnrollmentPayload(
        val publicKey: String,           // Hex encoded public key
        val commitment: String,          // Hex encoded commitment
        val idNumberHash: String,        // SHA256(ID number) for uniqueness check
        val encryptedPII: String,        // AES-256-GCM encrypted PII
        val proof: ProofData,            // ZKP proof
        val timestamp: Long,
        // Data binding fields (proves data came from OCR and Face Scan)
        val fullNameHash: String,        // SHA256 of full name (from OCR)
        val dobHash: String,             // SHA256 of date of birth (from OCR)
        val approval: Int,               // Face scan approval status (from AI model)
        val email: String = "",          // [NEW] Email for login
        val password: String = ""        // [NEW] Password for login
    )

    @Serializable
    data class ProofData(
        val commitmentR: String,         // Hex encoded R
        val challenge: String,           // Hex encoded c
        val response: String             // Hex encoded s
    )

    /**
     * Verification payload for subsequent logins
     */
    @Serializable
    data class VerificationPayload(
        val publicKey: String,
        val proof: ProofData,
        val nullifier: String,           // Prevents replay attacks
        val timestamp: Long
    )

    /**
     * Generate a new key pair for the user
     * This should be called once during enrollment
     */
    fun generateKeyPair(): KeyPair {
        val privateKey = BigInteger(256, secureRandom).mod(n)
        val publicKey = G.multiply(privateKey).normalize()
        return KeyPair(privateKey, publicKey)
    }

    /**
     * Compute public key from private key
     * publicKey = privateKey * G
     */
    fun computePublicKey(privateKey: BigInteger): ECPoint {
        return G.multiply(privateKey).normalize()
    }

    /**
     * Generate ZKP proof that proves knowledge of private key
     * without revealing it
     * 
     * @param privateKey The secret private key
     * @param message Additional message to bind to proof (e.g., session ID)
     * @return Schnorr proof
     */
    fun generateProof(privateKey: BigInteger, message: String = ""): SchnorrProof {
        // 1. Generate random k (ephemeral key)
        val k = BigInteger(256, secureRandom).mod(n)
        
        // 2. Compute commitment R = k*G
        val R = G.multiply(k).normalize()
        
        // 3. Compute public key P = x*G
        val P = G.multiply(privateKey).normalize()
        
        // 4. Compute challenge c = Hash(R || P || message)
        val challenge = computeChallenge(R, P, message)
        
        // 5. Compute response s = k + c*x (mod n)
        val response = k.add(challenge.multiply(privateKey)).mod(n)
        
        return SchnorrProof(R, challenge, response)
    }

    /**
     * Verify ZKP proof
     * This can be done by anyone with just the public key
     * NO NEED TO KNOW THE PRIVATE KEY
     * 
     * @param publicKey The public key
     * @param proof The Schnorr proof
     * @param message The message that was bound to the proof
     * @return true if proof is valid
     */
    fun verifyProof(publicKey: ECPoint, proof: SchnorrProof, message: String = ""): Boolean {
        try {
            // 1. Recompute challenge c' = Hash(R || P || message)
            val expectedChallenge = computeChallenge(proof.commitment, publicKey, message)
            
            // 2. Check if challenges match
            if (proof.challenge != expectedChallenge) {
                return false
            }
            
            // 3. Verify equation: s*G = R + c*P
            val leftSide = G.multiply(proof.response).normalize()
            val rightSide = proof.commitment.add(publicKey.multiply(proof.challenge)).normalize()
            
            return leftSide == rightSide
        } catch (e: Exception) {
            return false
        }
    }

    /**
     * Compute challenge using Fiat-Shamir heuristic
     * c = Hash(R || P || message)
     */
    private fun computeChallenge(R: ECPoint, P: ECPoint, message: String): BigInteger {
        val digest = MessageDigest.getInstance("SHA-256")
        
        // Encode R
        digest.update(R.getEncoded(true))
        
        // Encode P
        digest.update(P.getEncoded(true))
        
        // Encode message
        if (message.isNotEmpty()) {
            digest.update(message.toByteArray())
        }
        
        val hash = digest.digest()
        return BigInteger(1, hash).mod(n)
    }

    /**
     * Serialize ECPoint to hex string
     */
    fun pointToHex(point: ECPoint): String {
        return point.getEncoded(true).joinToString("") { "%02x".format(it) }
    }

    /**
     * Deserialize hex string to ECPoint
     */
    fun hexToPoint(hex: String): ECPoint {
        val bytes = hex.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
        return curveParams.curve.decodePoint(bytes)
    }

    /**
     * Serialize BigInteger to hex string
     */
    fun bigIntToHex(value: BigInteger): String {
        return value.toString(16).padStart(64, '0')
    }

    /**
     * Deserialize hex string to BigInteger
     */
    fun hexToBigInt(hex: String): BigInteger {
        return BigInteger(hex, 16)
    }

    /**
     * Hash data using SHA-256
     */
    fun sha256(data: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(data.toByteArray())
        return hash.joinToString("") { "%02x".format(it) }
    }

    /**
     * Generate a nullifier for replay attack prevention
     * nullifier = Hash(privateKey || sessionId)
     */
    fun generateNullifier(privateKey: BigInteger, sessionId: String): String {
        return sha256(bigIntToHex(privateKey) + sessionId)
    }
}
