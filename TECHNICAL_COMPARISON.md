# 🔬 Technical Comparison: Python Model vs Android Implementation

## 1️⃣ Model Architecture

### Python (debug_model.py)
```python
# Model Input Specification
id_img:       torch.Tensor[1, 3, 224, 224]  # Single ID image
video_frames: torch.Tensor[1, 8, 3, 224, 224]  # 8 video frames

# Model Output
return (liveness_score: float, matching_score: float)
```

### Android (EkycModelManager.kt)
```kotlin
// Model Input Specification
idTensor:      Tensor[1, 3, 224, 224]  // org.pytorch.Tensor
framesTensor:  Tensor[1, T, 3, 224, 224]  // T = frame count

// Model Output
outputs: IValue (isTuple = true)
tuple[0].toTensor() -> liveness_score
tuple[1].toTensor() -> matching_score
```

---

## 2️⃣ Image Preprocessing

### Common Normalization
```
ImageNet Mean: [0.485, 0.456, 0.406]
ImageNet Std:  [0.229, 0.224, 0.225]

Normalized_Value = (Original_Value / 255.0 - Mean) / Std
```

### Python Path
```python
# From debug_model.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts [0, 255] -> [0, 1] & HxWxC -> CxHxW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Result shape: [1, 3, 224, 224] (batch, channels, height, width)
```

### Android Path
```kotlin
// From EkycModelManager.kt
private fun bitmapToFloatArrayCHW(bitmap: Bitmap, width: Int, height: Int): FloatArray {
    val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val pixels = IntArray(224 * 224)
    resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
    
    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)
    
    val out = FloatArray(3 * 224 * 224)
    
    for (i in 0 until 224*224) {
        val pixel = pixels[i]
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f
        
        out[0*224*224 + i] = (r - mean[0]) / std[0]  // Channel R
        out[1*224*224 + i] = (g - mean[1]) / std[1]  // Channel G
        out[2*224*224 + i] = (b - mean[2]) / std[2]  // Channel B
    }
    
    return out  // Shape: [3, 224, 224] in memory as FloatArray
}

// Create Tensor: Tensor.fromBlob(out, longArrayOf(1, 3, 224, 224))
// Result shape: [1, 3, 224, 224]
```

---

## 3️⃣ Video Frame Processing

### Python Path
```python
# From debug_model.py
face_tensor_single = preprocess_image(face_img)  # [3, 224, 224]
face_tensor = face_tensor_single.unsqueeze(0).repeat(8, 1, 1, 1).unsqueeze(0)
# Result: [1, 8, 3, 224, 224]
```

### Android Path
```kotlin
// From EkycModelManager.kt
fun preprocessFrames(frames: List<Bitmap>): FloatArray {
    val T = frames.size  // Usually 8
    val out = FloatArray(T * 3 * 224 * 224)
    
    for (t in 0 until T) {
        val bmp = frames[t]
        val resized = Bitmap.createScaledBitmap(bmp, 224, 224, true)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Layout: [T, C, H, W] in memory
        // out[t * (3*224*224) + c * (224*224) + pixel_index]
        var rIdx = t * CHW + 0 * HW
        var gIdx = t * CHW + 1 * HW
        var bIdx = t * CHW + 2 * HW
        
        for (i in 0 until 224*224) {
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

// Create Tensor: Tensor.fromBlob(out, longArrayOf(1, T, 3, 224, 224))
// Result shape: [1, 8, 3, 224, 224]
```

---

## 4️⃣ Model Forward Pass

### Python
```python
with torch.no_grad():
    outputs = model(id_tensor, face_tensor)
    # outputs is a tuple: (liveness_score, matching_score)
    
    liveness_score = outputs[0].item()      # Convert tensor -> Python float
    matching_score = outputs[1].item()
```

### Android
```kotlin
val inputs = arrayOf(IValue.from(idTensor), IValue.from(framesTensor))
val outputs = mod.forward(*inputs)

if (outputs.isTuple) {
    val tuple = outputs.toTuple()
    
    val livenessTensor = tuple[0].toTensor()
    val matchingTensor = tuple[1].toTensor()
    
    val livenessProb = livenessTensor.dataAsFloatArray[0]
    val matchingScore = matchingTensor.dataAsFloatArray[0]
}
```

---

## 5️⃣ Output Validation

### Expected Ranges
```
Liveness Score:  [0.0, 1.0]  (probability)
Matching Score:  [0.0, 1.0]  (similarity)
```

### Thresholds (Android)
```kotlin
val livenessThreshold = 0.5f
val matchingThreshold = 0.5f

if (ekycResult.livenessProb > livenessThreshold &&
    ekycResult.matchingScore > matchingThreshold) {
    // PASS: Proceed to ZKP enrollment
} else {
    // FAIL: Request retake
}
```

---

## 6️⃣ Frame Extraction from Video

### Android
```kotlin
fun extractFramesFromVideo(context: Context, videoUri: Uri, targetFrameCount: Int): List<Bitmap> {
    val retriever = MediaMetadataRetriever()
    retriever.setDataSource(context, videoUri)
    
    val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
    val step = durationMs / targetFrameCount
    
    val frames = mutableListOf<Bitmap>()
    for (i in 0 until targetFrameCount) {
        val timeUs = (i * step) * 1000L
        val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        
        if (bitmap != null) {
            // Resize immediately to save memory
            val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            frames.add(resized)
            bitmap.recycle()
        }
    }
    
    return frames
}
```

**Key Points:**
- Uniformly samples frames across entire video duration
- Resizes to 224x224 immediately
- Returns List<Bitmap>, not ByteArray
- Memory-conscious: recycles original bitmap after resize

---

## 7️⃣ Consistency Checks

### ✅ Matching Python
- [x] ImageNet normalization identical
- [x] CHW memory layout consistent
- [x] Model input shapes [1,3,224,224] and [1,T,3,224,224]
- [x] Model output: tuple of (liveness, matching)
- [x] Output extraction: dataAsFloatArray[0]

### 🔍 To Verify on Device
```
LogCat filter: "EkycModelManager"
Expected logs:
- ID Tensor: size=150528, mean=~0.0, std=~1.0, first5=[...]
- Frames Tensor: size=1204224, mean=~0.0, std=~1.0, first5=[...]
- Running forward pass with ID shape [1,3,224,224] and Video shape [1,8,3,224,224]...
- ✅ Inference success: Liveness=0.XX, Matching=0.XX
```

---

## 8️⃣ Potential Issues & Fixes

### ❌ Issue: Tensor Shape Mismatch
```
Error: "Expected 5D tensor, got 4D tensor"
Fix: Ensure Tensor.fromBlob(..., longArrayOf(1, T, 3, 224, 224))
     Don't forget the batch dimension [1, ...]
```

### ❌ Issue: NaN or Invalid Scores
```
Cause: Uninitialized tensor, pixel byte overflow, or precision loss
Fix: Check mean/std in debug logs
     Verify pixels are in [0, 255] range
     Use Float instead of Int for normalization
```

### ❌ Issue: Model Loads but Returns Wrong Type
```
Error: "Unexpected model output type"
Fix: Model must return a tuple
     Check forward() method returns (liveness, matching)
```

### ❌ Issue: Out of Memory
```
Cause: Creating large bitmaps without recycling
Fix: Use createScaledBitmap(bitmap, 224, 224, true)
     Recycle immediately after processing
     Don't keep all frames in memory
```

---

## 📊 Memory Footprint

```
ID Image:      1 × 3 × 224 × 224 × 4 bytes = 0.6 MB
Video Frames:  8 × 3 × 224 × 224 × 4 bytes = 4.8 MB
Total per inference: ~5.4 MB
```

**Optimization:** Frames are resized to 224×224 immediately after extraction to minimize memory usage.

---

## 🎯 Summary

| Aspect | Python | Android |
|--------|--------|---------|
| **ID Input** | [1,3,224,224] | [1,3,224,224] ✅ |
| **Video Input** | [1,8,3,224,224] | [1,T,3,224,224] ✅ |
| **Normalization** | ImageNet | ImageNet ✅ |
| **Model Output** | Tuple | IValue(Tuple) ✅ |
| **Liveness Range** | [0,1] | [0,1] ✅ |
| **Matching Range** | [0,1] | [0,1] ✅ |

**Status:** ✅ Python and Android implementations are synchronized!
