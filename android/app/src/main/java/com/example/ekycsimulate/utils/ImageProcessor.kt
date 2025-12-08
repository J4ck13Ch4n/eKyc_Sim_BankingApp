package com.example.ekycsimulate.utils

import android.graphics.*
import android.util.Log
import androidx.core.graphics.applyCanvas
import kotlin.math.roundToInt

data class IlluminationDiagnostics(
    val brightRatio: Float,
    val darkRatio: Float,
    val meanLuma: Float,
    val sampleCount: Int
) {
    val isOverexposed: Boolean get() = brightRatio > 0.18f && meanLuma > 0.55f
    val isSeverelyOverexposed: Boolean get() = brightRatio > 0.32f
    val isUnderexposed: Boolean get() = darkRatio > 0.28f && meanLuma < 0.42f
    val shouldRequestRecapture: Boolean get() = isSeverelyOverexposed || (isOverexposed && meanLuma > 0.62f)
}

data class OcrPreparationResult(
    val variants: List<Bitmap>,
    val diagnostics: IlluminationDiagnostics
)

object ImageProcessor {
    private const val TAG = "ImageProcessor"

    /**
     * Higher-level API: return a cleaned single image for quick OCR, and also
     * generateVariants for multi-pass OCR.
     */
    fun runAdvancedPreprocessing(bitmap: Bitmap): Bitmap {
        // Try OpenCV path if available
        return try {
            preprocessWithOpenCv(bitmap)
        } catch (t: Throwable) {
            Log.w(TAG, "OpenCV not available or preprocessWithOpenCv failed: ${t.message}")
            // fallback: grayscale -> contrast -> sharpen
            val g = toGrayscale(bitmap)
            val c = changeContrastAndBrightness(g, 1.5f, -40f)
            sharpen(c)
        }
    }

    /**
     * Generate multiple image variants used for multi-pass voting.
     */
    fun generateVariants(original: Bitmap): List<Bitmap> {
        return prepareForOcr(original).variants
    }

    /** Prepare variants plus diagnostics so caller can detect exposure problems. */
    fun prepareForOcr(original: Bitmap, maxVariants: Int = 8): OcrPreparationResult {
        val diagnostics = analyzeIllumination(original)
        val variants = mutableListOf<Bitmap>()
        fun addVariant(bitmap: Bitmap) {
            if (variants.none { it === bitmap }) variants.add(bitmap)
        }

        addVariant(original)

        if (diagnostics.isOverexposed || diagnostics.isSeverelyOverexposed) {
            addVariant(reduceSpecularHighlights(original))
            addVariant(changeContrastAndBrightness(original, 1.2f, -60f))
        }

        if (diagnostics.isUnderexposed) {
            addVariant(boostShadows(original))
            addVariant(changeContrastAndBrightness(original, 1.5f, 35f))
        }

        val cleaned = runAdvancedPreprocessing(original)
        addVariant(cleaned)
        addVariant(changeContrastAndBrightness(cleaned, 1.4f, -20f))
        addVariant(sharpen(toGrayscale(original)))
        addVariant(binaryThreshold(toGrayscale(original)))

        if (variants.size > maxVariants) {
            variants.subList(maxVariants, variants.size).clear()
        }

        Log.d(TAG, "Generated ${variants.size} variants for OCR (brightRatio=${diagnostics.brightRatio}, darkRatio=${diagnostics.darkRatio})")
        return OcrPreparationResult(variants, diagnostics)
    }

    // ---------- IMAGE OPS (Kotlin fallback) ----------

    private fun toGrayscale(src: Bitmap): Bitmap {
        val bmpGrayscale = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmpGrayscale)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(src, 0f, 0f, paint)
        return bmpGrayscale
    }

    private fun binaryThreshold(src: Bitmap, threshold: Int = 130): Bitmap {
        val width = src.width
        val height = src.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = src.getPixel(x, y)
                val gray = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                val value = if (gray > threshold) 255 else 0
                result.setPixel(x, y, Color.rgb(value, value, value))
            }
        }
        return result
    }

    private fun changeContrastAndBrightness(bitmap: Bitmap, contrast: Float, brightness: Float): Bitmap {
        val colorMatrix = ColorMatrix(floatArrayOf(
            contrast, 0f, 0f, 0f, brightness,
            0f, contrast, 0f, 0f, brightness,
            0f, 0f, contrast, 0f, brightness,
            0f, 0f, 0f, 1f, 0f
        ))
        val grayscaleMatrix = ColorMatrix().apply { setSaturation(0f) }
        grayscaleMatrix.postConcat(colorMatrix)
        val paint = Paint().apply { colorFilter = ColorMatrixColorFilter(grayscaleMatrix) }
        val resultBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        resultBitmap.applyCanvas { drawBitmap(bitmap, 0f, 0f, paint) }
        return resultBitmap
    }

    private fun sharpen(src: Bitmap): Bitmap {
        val kernel = floatArrayOf(
            0f, -1f, 0f,
            -1f, 5f, -1f,
            0f, -1f, 0f
        )
        val width = src.width
        val height = src.height
        val pixels = IntArray(width * height)
        src.getPixels(pixels, 0, width, 0, 0, width, height)
        val resultPixels = IntArray(width * height)
        // copy border pixels
        for (x in 0 until width) {
            resultPixels[x] = pixels[x]
            resultPixels[(height - 1) * width + x] = pixels[(height - 1) * width + x]
        }
        for (y in 0 until height) {
            resultPixels[y * width] = pixels[y * width]
            resultPixels[y * width + width - 1] = pixels[y * width + width - 1]
        }
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var sumR = 0f
                var sumG = 0f
                var sumB = 0f
                var kernelIndex = 0
                for (ky in -1..1) {
                    for (kx in -1..1) {
                        val pixel = pixels[(y + ky) * width + (x + kx)]
                        val kv = kernel[kernelIndex++]
                        sumR += Color.red(pixel) * kv
                        sumG += Color.green(pixel) * kv
                        sumB += Color.blue(pixel) * kv
                    }
                }
                val r = sumR.coerceIn(0f, 255f).roundToInt()
                val g = sumG.coerceIn(0f, 255f).roundToInt()
                val b = sumB.coerceIn(0f, 255f).roundToInt()
                resultPixels[y * width + x] = Color.rgb(r, g, b)
            }
        }
        val resultBitmap = Bitmap.createBitmap(width, height, src.config ?: Bitmap.Config.ARGB_8888)
        resultBitmap.setPixels(resultPixels, 0, width, 0, 0, width, height)
        return resultBitmap
    }

    fun analyzeIllumination(bitmap: Bitmap, sampleStep: Int = 12): IlluminationDiagnostics {
        val width = bitmap.width
        val height = bitmap.height
        var bright = 0
        var dark = 0
        var total = 0
        var sumLuma = 0f
        for (y in 0 until height step sampleStep) {
            for (x in 0 until width step sampleStep) {
                val pixel = bitmap.getPixel(x, y)
                val luma = (Color.red(pixel) * 0.299f + Color.green(pixel) * 0.587f + Color.blue(pixel) * 0.114f) / 255f
                if (luma > 0.85f) bright++
                if (luma < 0.12f) dark++
                sumLuma += luma
                total++
            }
        }
        val mean = if (total == 0) 0f else sumLuma / total
        val brightRatio = if (total == 0) 0f else bright.toFloat() / total
        val darkRatio = if (total == 0) 0f else dark.toFloat() / total
        return IlluminationDiagnostics(brightRatio, darkRatio, mean, total)
    }

    private fun reduceSpecularHighlights(src: Bitmap): Bitmap {
        val width = src.width
        val height = src.height
        val result = Bitmap.createBitmap(width, height, src.config ?: Bitmap.Config.ARGB_8888)
        val hsv = FloatArray(3)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = src.getPixel(x, y)
                Color.colorToHSV(pixel, hsv)
                if (hsv[2] > 0.78f) {
                    val excess = hsv[2] - 0.78f
                    hsv[2] = 0.78f + excess * 0.25f
                    hsv[1] = (hsv[1] * 0.85f).coerceAtLeast(0f)
                }
                result.setPixel(x, y, Color.HSVToColor(Color.alpha(pixel), hsv))
            }
        }
        return result
    }

    private fun boostShadows(src: Bitmap): Bitmap {
        val width = src.width
        val height = src.height
        val result = Bitmap.createBitmap(width, height, src.config ?: Bitmap.Config.ARGB_8888)
        val hsv = FloatArray(3)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = src.getPixel(x, y)
                Color.colorToHSV(pixel, hsv)
                if (hsv[2] < 0.35f) {
                    val deficit = 0.35f - hsv[2]
                    hsv[2] = (hsv[2] + deficit * 0.6f).coerceAtMost(0.55f)
                }
                result.setPixel(x, y, Color.HSVToColor(Color.alpha(pixel), hsv))
            }
        }
        return result
    }

    // ---------- OPTIONAL: OpenCV-based preprocess (if OpenCV available) ----------
    private fun preprocessWithOpenCv(bitmap: Bitmap): Bitmap {
        // NOTE: This method requires OpenCV android libs and setup; if not present, it will throw.
        // We'll implement a simple deskew + adaptive threshold-like flow using OpenCV APIs.
        // To keep this code compile-safe even if OpenCV not present, we call it inside try/catch in runAdvancedPreprocessing.
        throw UnsupportedOperationException("OpenCV not initialized in this build. If you want OpenCV path, implement preprocessWithOpenCv using org.opencv.* APIs.")
    }
    /**
     * Crops the face from the bitmap given the bounding box, with optional padding.
     */
    fun cropFace(bitmap: Bitmap, bounds: Rect, paddingPercent: Float = 0.20f): Bitmap {
        val width = bounds.width()
        val height = bounds.height()
        
        // Add padding
        val paddingX = (width * paddingPercent).toInt()
        val paddingY = (height * paddingPercent).toInt()

        val left = (bounds.left - paddingX).coerceAtLeast(0)
        val top = (bounds.top - paddingY).coerceAtLeast(0)
        val right = (bounds.right + paddingX).coerceAtMost(bitmap.width)
        val bottom = (bounds.bottom + paddingY).coerceAtMost(bitmap.height)

        val cropWidth = right - left
        val cropHeight = bottom - top

        // If invalid crop, return original or throwing error? Let's return original with warning log for safety
        if (cropWidth <= 0 || cropHeight <= 0) {
            Log.w(TAG, "Invalid crop dimensions: w=$cropWidth, h=$cropHeight. Returning original.")
            return bitmap
        }

        return Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)
    }
}
