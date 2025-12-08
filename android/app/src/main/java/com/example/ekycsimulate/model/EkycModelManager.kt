package com.example.ekycsimulate.model

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.LiteModuleLoader
import java.io.File
import java.io.FileOutputStream

data class EkycResult(val livenessProb: Float, val matchingScore: Float) {
    fun isPassed(threshold: Float = 0.7f): Boolean {
        return livenessProb > threshold && matchingScore > threshold
    }
}

class EkycModelManager(
    private val context: Context,
    private val faceDetector: com.example.ekycsimulate.domain.FaceDetector
) {
    private var module: Module? = null

    init {
        try {
            module = loadModuleFromAssets("ekyc_model.ptl")
        } catch (e: Exception) {
            Log.e("EkycModelManager", "Failed to load model: ${e.message}")
        }
    }

    private fun loadModuleFromAssets(assetName: String): Module? {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            try {
                context.assets.open(assetName).use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.e("EkycModelManager", "Error copying asset: $assetName", e)
                return null
            }
        }
        return try {
            LiteModuleLoader.load(file.absolutePath)
        } catch (e: Exception) {
            Log.e("EkycModelManager", "Error loading module: ${file.absolutePath}", e)
            null
        }
    }

    private fun preprocessId(idBitmap: Bitmap): FloatArray {
        return bitmapToFloatArrayCHW(idBitmap, 224, 224)
    }

    private fun preprocessFrames(frames: List<Bitmap>): FloatArray {
        // Model expects: [1, T, C, H, W]
        // We return FloatArray in T,C,H,W layout (no batch dim, will add in runInference)
        val T = frames.size
        val H = 224
        val W = 224
        val HW = H * W
        val CHW = 3 * HW
        val out = FloatArray(T * CHW)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val pixels = IntArray(HW)

        for (t in 0 until T) {
            val bmp = frames[t]
            // Ensure bitmap is RGB and resized
            val resized = Bitmap.createScaledBitmap(bmp, W, H, true)
            resized.getPixels(pixels, 0, W, 0, 0, W, H)
            
            // Extract pixel data into CHW layout
            var rIdx = t * CHW + 0 * HW
            var gIdx = t * CHW + 1 * HW
            var bIdx = t * CHW + 2 * HW
            
            for (i in 0 until HW) {
                val pixel = pixels[i]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                out[rIdx++] = (r - mean[0]) / std[0]
                out[gIdx++] = (g - mean[1]) / std[1]
                out[bIdx++] = (b - mean[2]) / std[2]
            }
        }
        return out
    }

    private fun bitmapToFloatArrayCHW(bitmap: Bitmap, width: Int, height: Int): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val out = FloatArray(3 * width * height)
        val wh = width * height

        var rIndex = 0
        var gIndex = wh
        var bIndex = 2 * wh

        for (i in 0 until wh) {
            val c = pixels[i]
            val r = ((c shr 16) and 0xFF) / 255.0f
            val g = ((c shr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f

            out[rIndex++] = (r - mean[0]) / std[0]
            out[gIndex++] = (g - mean[1]) / std[1]
            out[bIndex++] = (b - mean[2]) / std[2]
        }

        return out
    }

    suspend fun runInference(frames: List<Bitmap>, idBitmap: Bitmap): Result<EkycResult> {
        Log.d("EkycModelManager", "runInference called with ${frames.size} frames")
        
        // 0. Detect and Crop Face from ID Card
        Log.d("EkycModelManager", "Detecting face in ID card...")
        val faceResults = faceDetector.detect(idBitmap)
        if (faceResults.isEmpty()) {
            Log.e("EkycModelManager", "No face detected in ID card")
            return Result.failure(Exception("No face detected in ID card"))
        }
        
        // Use the largest face
        val face = faceResults.first()
        val bounds = face.bounds
        
        // --- SQUARE CROP LOGIC ---
        val centerX = bounds.centerX()
        val centerY = bounds.centerY()
        val sideLength = kotlin.math.max(bounds.width(), bounds.height())
        // Add some padding (e.g., 20%)
        val paddedSide = (sideLength * 1.2f).toInt()
        val halfSide = paddedSide / 2
        
        val x1 = (centerX - halfSide).coerceAtLeast(0)
        val y1 = (centerY - halfSide).coerceAtLeast(0)
        val x2 = (centerX + halfSide).coerceAtMost(idBitmap.width)
        val y2 = (centerY + halfSide).coerceAtMost(idBitmap.height)
        
        val w = x2 - x1
        val h = y2 - y1
        
        if (w <= 0 || h <= 0) {
             return Result.failure(Exception("Invalid face bounds"))
        }

        val croppedIdFace = Bitmap.createBitmap(idBitmap, x1, y1, w, h)
        Log.d("EkycModelManager", "Face cropped square: ${w}x${h}")
        
        // DEBUG: Save images to inspect
        saveBitmapToCache(croppedIdFace, "debug_id_face.jpg")
        if (frames.isNotEmpty()) {
            saveBitmapToCache(frames[0], "debug_video_frame_0.jpg")
        }


        if (module == null) {
            Log.e("EkycModelManager", "Module is null. Attempting to reload...")
            module = loadModuleFromAssets("ekyc_model_mobile.ptl")
            if (module == null) {
                return Result.failure(Exception("Failed to load model ekyc_model_mobile.ptl"))
            }
        }
        val mod = module!!
        
        try {
            val T = frames.size
            if (T <= 0) {
                return Result.failure(Exception("No frames provided for inference"))
            }
            
            // 1. Preprocess ID Image (Use Cropped Face)
            Log.d("EkycModelManager", "Preprocessing ID bitmap...")
            val idArr = preprocessId(croppedIdFace) 
            // Shape: [1, 3, 224, 224]
            val idTensor = Tensor.fromBlob(idArr, longArrayOf(1, 3, 224, 224))

            // 2. Preprocess Video Frames
            Log.d("EkycModelManager", "Preprocessing ${frames.size} frames...")
            val framesArr = preprocessFrames(frames)
            // Shape: [1, T, 3, 224, 224]
            val framesTensor = Tensor.fromBlob(framesArr, longArrayOf(1, T.toLong(), 3, 224, 224))

            // DEBUG: Check tensor values
            val idMean = idArr.average()
            val framesMean = framesArr.average()
            val idStd = kotlin.math.sqrt(idArr.map { (it - idMean) * (it - idMean) }.average())
            val framesStd = kotlin.math.sqrt(framesArr.map { (it - framesMean) * (it - framesMean) }.average())
            
            Log.d("EkycModelManager", "ID Tensor: size=${idArr.size}, mean=$idMean, std=$idStd, first5=${idArr.take(5)}")
            Log.d("EkycModelManager", "Frames Tensor: size=${framesArr.size}, mean=$framesMean, std=$framesStd, first5=${framesArr.take(5)}")

            // 3. Prepare Inputs - Order: (ID, Video)
            // Model forward: def forward(self, id_img, video_frames):
            val inputs = arrayOf(IValue.from(idTensor), IValue.from(framesTensor))
            
            Log.d("EkycModelManager", "Running forward pass with ID shape [1,3,224,224] and Video shape [1,$T,3,224,224]...")
            val outputs = mod.forward(*inputs)
            
            if (outputs.isTuple) {
                val tuple = outputs.toTuple()
                Log.d("EkycModelManager", "Output tuple size: ${tuple.size}")
                
                if (tuple.size < 2) {
                     return Result.failure(Exception("Model output tuple size mismatch. Expected >= 2, got ${tuple.size}"))
                }
                
                // --- SWAP OUTPUTS ---
                val matchingTensor = tuple[0].toTensor() 
                val livenessTensor = tuple[1].toTensor()

                val matchingData = matchingTensor.dataAsFloatArray
                val livenessData = livenessTensor.dataAsFloatArray
                
                Log.d("EkycModelManager", "Raw Matching Tensor: ${matchingData.joinToString()}")
                Log.d("EkycModelManager", "Raw Liveness Tensor: ${livenessData.joinToString()}")

                val matchingScore = matchingData[0]
                val livenessProb = livenessData[0]
                
                Log.d("EkycModelManager", "✅ Inference success: Liveness=$livenessProb, Matching=$matchingScore")
                return Result.success(EkycResult(livenessProb, matchingScore))
            } else {
                Log.e("EkycModelManager", "❌ Unexpected model output type: $outputs")
                return Result.failure(Exception("Unexpected model output type"))
            }
        } catch (e: Exception) {
            Log.e("EkycModelManager", "❌ Model inference failed", e)
            return Result.failure(e)
        }
    }

    private fun saveBitmapToCache(bitmap: Bitmap, filename: String) {
        try {
            val file = File(context.cacheDir, filename)
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            Log.d("EkycModelManager", "Saved debug image to: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e("EkycModelManager", "Failed to save debug image", e)
        }
    }
}
