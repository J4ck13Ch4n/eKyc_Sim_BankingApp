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

class EkycModelManager(private val context: Context) {
    private var module: Module? = null

    init {
        try {
            module = loadModuleFromAssets("ekyc_model_mobile.ptl")
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
        // Layout: [T, C, H, W]
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
            val resized = Bitmap.createScaledBitmap(bmp, W, H, true)
            resized.getPixels(pixels, 0, W, 0, 0, W, H)
            
            for (i in 0 until HW) {
                val c = pixels[i]
                val r = ((c shr 16) and 0xFF) / 255.0f
                val g = ((c shr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f

                // Frame t, Channel 0 (R), Pixel i
                out[t * CHW + 0 * HW + i] = (r - mean[0]) / std[0]
                
                // Frame t, Channel 1 (G), Pixel i
                out[t * CHW + 1 * HW + i] = (g - mean[1]) / std[1]
                
                // Frame t, Channel 2 (B), Pixel i
                out[t * CHW + 2 * HW + i] = (b - mean[2]) / std[2]
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

    fun runInference(frames: List<Bitmap>, idBitmap: Bitmap): Result<EkycResult> {
        Log.d("EkycModelManager", "runInference called with ${frames.size} frames")
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
            
            // 1. Preprocess ID Image
            Log.d("EkycModelManager", "Preprocessing ID bitmap...")
            val idArr = preprocessId(idBitmap) 
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
            Log.d("EkycModelManager", "ID Tensor Stats: Size=${idArr.size}, Mean=$idMean, First=${idArr.take(5)}")
            Log.d("EkycModelManager", "Frames Tensor Stats: Size=${framesArr.size}, Mean=$framesMean, First=${framesArr.take(5)}")


            // 3. Prepare Inputs - SWAPPED ORDER: (ID, Video)
            // Model forward: def forward(self, id_img, video_frames):
            val inputs = arrayOf(IValue.from(idTensor), IValue.from(framesTensor))
            
            Log.d("EkycModelManager", "Running forward pass...")
            val outputs = mod.forward(*inputs)
            
            if (outputs.isTuple) {
                val tuple = outputs.toTuple()
                // Model returns: (live_score, match_score)
                if (tuple.size < 2) {
                     return Result.failure(Exception("Model output tuple size mismatch. Expected >= 2, got ${tuple.size}"))
                }
                
                val livenessTensor = tuple[0].toTensor()
                val matchingTensor = tuple[1].toTensor()

                val livenessProb = livenessTensor.dataAsFloatArray[0]
                val matchingScore = matchingTensor.dataAsFloatArray[0]
                
                Log.d("EkycModelManager", "Inference success: Liveness=$livenessProb, Matching=$matchingScore")
                return Result.success(EkycResult(livenessProb, matchingScore))
            } else {
                Log.e("EkycModelManager", "Unexpected model output type: $outputs")
                return Result.failure(Exception("Unexpected model output type"))
            }
        } catch (e: Exception) {
            Log.e("EkycModelManager", "Model inference failed", e)
            return Result.failure(e)
        }
    }
}
