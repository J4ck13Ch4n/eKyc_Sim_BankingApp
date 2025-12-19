package com.example.ekycsimulate.model

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.sqrt

class FaceRecognitionManager(private val context: Context) {

    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null

    init {
        initModel()
    }

    private fun initModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelFile = loadModelFromAssets("face_recognition_s.onnx")
            if (modelFile != null) {
                val opts = OrtSession.SessionOptions()
                ortSession = ortEnv?.createSession(modelFile.absolutePath, opts)
                Log.d("FaceRecognitionManager", "Face recognition model loaded successfully")
            } else {
                Log.e("FaceRecognitionManager", "Failed to load face recognition model file")
            }
        } catch (e: Exception) {
            Log.e("FaceRecognitionManager", "Error initializing model", e)
        }
    }

    private fun loadModelFromAssets(assetName: String): File? {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            try {
                context.assets.open(assetName).use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.e("FaceRecognitionManager", "Error copying asset: $assetName", e)
                return null
            }
        }
        return file
    }

    fun getEmbedding(bitmap: Bitmap): FloatArray? {
        if (ortSession == null) {
            Log.e("FaceRecognitionManager", "Model not initialized")
            return null
        }

        try {
            // 1. Preprocess: Resize to 112x112 and Normalize
            val floatBuffer = preprocess(bitmap)

            // 2. Create Input Tensor
            val inputName = ortSession!!.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, 112, 112)
            val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)

            // 3. Run Inference
            val results = ortSession!!.run(Collections.singletonMap(inputName, inputTensor))
            
            // 4. Get Output
            // Assuming the first output is the embedding
            val outputTensor = results[0] as OnnxTensor
            val floatArray = outputTensor.floatBuffer.array()
            
            results.close()
            inputTensor.close()

            return floatArray
        } catch (e: Exception) {
            Log.e("FaceRecognitionManager", "Error running inference", e)
            return null
        }
    }

    private fun preprocess(bitmap: Bitmap): FloatBuffer {
        val width = 112
        val height = 112
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatBuffer = FloatBuffer.allocate(1 * 3 * width * height)
        
        // NCHW Layout
        // InsightFace expectation: (pixel - 127.5) / 128.0  (Standard for some ArcFace)
        // OR Standard ImageNet: (pixel/255 - mean) / std.
        // Usually, InsightFace uses (pixel - 127.5) / 128.0 which scales to [-1, 1]
        
        for (i in 0 until width * height) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF)
            val g = ((pixel shr 8) and 0xFF)
            val b = (pixel and 0xFF)

            // Layout: RRR... GGG... BBB...
            // Indexing for NCHW
        }
        
        // Let's do a second pass for cleanliness or use calculation
        
        for (c in 0..2) { // 0=R, 1=G, 2=B
            for (i in 0 until width * height) {
                val pixel = pixels[i]
                val p = when(c) {
                    0 -> ((pixel shr 16) and 0xFF)
                    1 -> ((pixel shr 8) and 0xFF)
                    else -> (pixel and 0xFF)
                }
                
                // Normalization: (x - 127.5) / 128.0
                val normalized = (p - 127.5f) / 128.0f
                floatBuffer.put(normalized)
            }
        }
        
        floatBuffer.rewind()
        return floatBuffer
    }

    fun calculateCosineSimilarity(emb1: FloatArray, emb2: FloatArray): Float {
        if (emb1.size != emb2.size) return 0f
        
        var dot = 0f
        var norm1 = 0f
        var norm2 = 0f
        
        for (i in emb1.indices) {
            dot += emb1[i] * emb2[i]
            norm1 += emb1[i] * emb1[i]
            norm2 += emb2[i] * emb2[i]
        }
        
        return if (norm1 > 0 && norm2 > 0) {
            dot / (sqrt(norm1) * sqrt(norm2))
        } else {
            0f
        }
    }
}
