package com.example.ekycsimulate.ui.auth

import com.example.ekycsimulate.utils.ImageProcessor

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.content.ContentValues
import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.ekycsimulate.ui.viewmodel.EkycViewModel
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.ekycsimulate.zkp.ZKPEnrollmentManager
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@Composable
fun FaceScanScreen(
    idCardInfo: IdCardInfo,
    croppedImage: Bitmap?, // Passed from shared ViewModel
    onEnrollmentComplete: (String) -> Unit,  // Callback with JSON payload
    onBack: () -> Unit // Callback for Cancel button
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    // val ekycViewModel: EkycViewModel = viewModel() // Removed: Use passed parameter
    
    var hasCameraPermission by remember { 
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        ) 
    }
    
    var capturedImage by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }
    var isRecording by remember { mutableStateOf(false) }

    var videoUri by remember { mutableStateOf<Uri?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var videoCapture by remember { mutableStateOf<VideoCapture<Recorder>?>(null) }

    var randomDigits by remember { mutableStateOf(generateRandomDigits()) }
    var approvalStatus by remember { mutableStateOf(0) } // 0: None, 1: Approved
    var enrollmentPayload by remember { mutableStateOf<String?>(null) }
    var enrollmentDataObj by remember { mutableStateOf<ZKPEnrollmentManager.EnrollmentData?>(null) }
    var zkpDetails by remember { mutableStateOf<Map<String, String>?>(null) }
    var isSending by remember { mutableStateOf(false) }
    var sendError by remember { mutableStateOf<String?>(null) }
    var inferenceResult by remember { mutableStateOf<com.example.ekycsimulate.model.EkycResult?>(null) }
    var debugLog by remember { mutableStateOf("") }


    
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    // Toggle this for testing on Emulator vs Real Device
    val useFakeDetector = false 
    val faceDetector = remember { 
        if (useFakeDetector) com.example.ekycsimulate.data.FakeFaceDetector() 
        else com.example.ekycsimulate.data.MLKitFaceDetector() 
    }

    val modelManager = remember { com.example.ekycsimulate.model.EkycModelManager(context, faceDetector) }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Xác thực khuôn mặt", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))

        when {
            !hasCameraPermission -> {
                Text("Cần quyền truy cập Camera để tiếp tục")
                Button(onClick = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Cấp quyền Camera")
                }
            }
            
            // FAILURE STATE: Show Error and Retry/Cancel options
            sendError != null -> {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = androidx.compose.material.icons.Icons.Default.Warning,
                            contentDescription = "Error",
                            tint = MaterialTheme.colorScheme.error
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Xác thực thất bại",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = sendError ?: "Lỗi không xác định",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(24.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    OutlinedButton(
                        onClick = onBack,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Hủy bỏ")
                    }
                    
                    Spacer(modifier = Modifier.width(16.dp))
                    
                    Button(
                        onClick = {
                            // Retry Logic
                            sendError = null
                            approvalStatus = 0
                            capturedImage = null
                            videoUri = null
                            isProcessing = false
                            inferenceResult = null // Reset this too
                            randomDigits = generateRandomDigits()
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Thử lại")
                    }
                }
            }

            isProcessing -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Đang xử lý...")
            }

            enrollmentPayload != null -> {
                Text("✅ ZKP Enrollment Complete!", color = Color(0xFF4CAF50), fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(16.dp))
                
                // Display ZKP Details
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                        zkpDetails?.forEach { (key, value) ->
                            Text(key, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                            SelectionContainer {
                                Text(value, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, fontSize = 9.sp)
                            }
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Show payload preview
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Server Payload (Ready to Send):", style = MaterialTheme.typography.titleSmall)
                        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                        SelectionContainer {
                            Text(
                                enrollmentPayload!!.take(200) + "\n...",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 8.sp
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(24.dp))
                
                if (sendError != null) {
                    Text("Lỗi: $sendError", color = MaterialTheme.colorScheme.error)
                    Spacer(modifier = Modifier.height(8.dp))
                }

                Button(
                    onClick = { 
                        if (enrollmentDataObj != null && !isSending) {
                            isSending = true
                            sendError = null
                            scope.launch {
                                val manager = ZKPEnrollmentManager(context)
                                val result = manager.sendEnrollment(enrollmentDataObj!!.payload)
                                result.onSuccess {
                                    isSending = false
                                    onEnrollmentComplete(enrollmentPayload!!)
                                }.onFailure { e ->
                                    isSending = false
                                    sendError = e.message ?: "Unknown error"
                                }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isSending
                ) {
                    if (isSending) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp), color = MaterialTheme.colorScheme.onPrimary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Đang gửi...")
                    } else {
                        Text("Hoàn tất & Gửi lên Server")
                    }
                }
                
                OutlinedButton(
                    onClick = { 
                        capturedImage = null
                        approvalStatus = 0
                        enrollmentPayload = null
                        zkpDetails = null
                        videoUri = null

                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Chụp lại")
                }
                Spacer(modifier = Modifier.height(16.dp))
                
                // Run simulation on frames
                Button(onClick = {
                    if (videoUri == null) { sendError = "Chưa có video"; return@Button }
                    isProcessing = true
                    scope.launch {
                        delay(5000) // Simulate work
                        isProcessing = false
                        
                        approvalStatus = 1
                        enrollmentPayload = null // Reset to trigger ZKP flow
                        zkpDetails = null
                        sendError = null
                        randomDigits = generateRandomDigits()
                    }
                }, modifier = Modifier.fillMaxWidth()) {
                    Text("Chạy lại (Model Inference)")
                }
            }

            approvalStatus == 1 && enrollmentPayload == null -> {
                Text("✅ Xác thực thành công!", color = Color(0xFF4CAF50), fontWeight = FontWeight.Bold)
                Text("Approval Status: 1", style = MaterialTheme.typography.bodySmall)
                Spacer(modifier = Modifier.height(16.dp))
                
                Text("Đang tạo Zero-Knowledge Proof...", style = MaterialTheme.typography.bodyMedium)
                
                LaunchedEffect(Unit) {
                    delay(1000)
                    
                    // Generate ZKP enrollment (binds OCR data + approval)
                    val enrollmentManager = ZKPEnrollmentManager(context)
                    val enrollmentData = enrollmentManager.performEnrollment(
                        idCardInfo = idCardInfo,
                        fullName = idCardInfo.fullName,
                        phoneNumber = "", // You can add input fields for these
                        address = idCardInfo.address,
                        faceImageApproval = approvalStatus
                    )
                    
                    enrollmentDataObj = enrollmentData
                    val payload = enrollmentManager.enrollmentPayloadToJson(enrollmentData.payload)
                    enrollmentPayload = payload
                    
                    // Extract details for display
                    zkpDetails = mapOf(
                        "Public Key" to enrollmentData.payload.publicKey.take(40) + "...",
                        "Commitment" to enrollmentData.payload.commitment,
                        "ID Hash" to enrollmentData.payload.idNumberHash,
                        "Proof R" to enrollmentData.payload.proof.commitmentR.take(40) + "...",
                        "Proof Challenge" to enrollmentData.payload.proof.challenge.take(40) + "...",
                        "Proof Response" to enrollmentData.payload.proof.response.take(40) + "..."
                    )
                }
                
                CircularProgressIndicator(modifier = Modifier.size(24.dp))
            }
            
            capturedImage == null -> {
                // Camera Preview
                Text("Đặt khuôn mặt vào khung hình", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(8.dp))
                
                CameraPreview(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(400.dp),
                    onImageCaptured = { bitmap ->
                        capturedImage = bitmap
                        // Start processing
                        isProcessing = true
                        scope.launch {
                            val results = faceDetector.detect(bitmap)
                            
                            if (results.isNotEmpty()) {
                                val face = results.first()
                                // Simple Liveness Check: Ensure face is looking somewhat straight
                                val isLookingStraight = kotlin.math.abs(face.headEulerAngleY) < 15 && kotlin.math.abs(face.headEulerAngleZ) < 15
                                
                                if (isLookingStraight) {
                                    approvalStatus = 1 // Approved
                                } else {
                                    // In a real app, you'd show a specific error
                                    approvalStatus = 0
                                    capturedImage = null // Reset to try again
                                }
                            } else {
                                approvalStatus = 0
                                capturedImage = null // Reset
                            }
                            isProcessing = false
                        }
                    },
                    onVideoCaptureReady = { vc ->
                        videoCapture = vc
                    }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Random Digits Display (Large and Clear)
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Vui lòng đọc to dãy số sau:", style = MaterialTheme.typography.labelLarge)
                        Text(
                            text = randomDigits,
                            style = MaterialTheme.typography.displayMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            letterSpacing = 4.sp
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = {
                            if (videoCapture == null) {

                                return@Button
                            }
                            
                            val name = "ekyc_rec_${System.currentTimeMillis()}.mp4"
                            val contentValues = ContentValues().apply {
                                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                                if (android.os.Build.VERSION.SDK_INT > android.os.Build.VERSION_CODES.P) {
                                    put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/EkycSim")
                                }
                            }
                            val mediaStoreOutputOptions = MediaStoreOutputOptions
                                .Builder(context.contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                                .setContentValues(contentValues)
                                .build()
                            
                            try {
                                val activeRecording = videoCapture!!.output
                                    .prepareRecording(context, mediaStoreOutputOptions)
                                    .apply {
                                        // if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                                        //     withAudioEnabled()
                                        // }
                                    }
                                    .start(ContextCompat.getMainExecutor(context)) { recordEvent ->
                                        when(recordEvent) {
                                            is VideoRecordEvent.Start -> {
                                                isRecording = true

                                            }
                                            is VideoRecordEvent.Finalize -> {
                                                isRecording = false
                                                if (!recordEvent.hasError()) {
                                                    val uri = recordEvent.outputResults.outputUri
                                                    videoUri = uri

                                                    
                                                    // Process video
                                                    scope.launch {
                                                        isProcessing = true
                                                        try {
                                                            // Move heavy work to IO thread
                                                            withContext(kotlinx.coroutines.Dispatchers.IO) {
                                                                // Small delay to ensure file is ready
                                                                delay(500)
                                                                
                                                                // Extract 8 frames as requested
                                                                val frames = extractFramesFromVideo(context, uri, 8)
                                                                
                                                                withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                    debugLog = "Đã trích xuất ${frames.size} frames. Đang chạy Face Detection...\n$debugLog"
                                                                }
                                                                
                                                                if (frames.isEmpty()) {
                                                                    withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                        sendError = "Không thể trích xuất frames từ video"

                                                                    }
                                                                    return@withContext
                                                                }

                                                                // --- FACE DETECTION & CROPPING (VIDEO) ---
                                                                val croppedFrames = mutableListOf<Bitmap>()
                                                                var framesWithFace = 0
                                                                for (frame in frames) {
                                                                    val faces = faceDetector.detect(frame)
                                                                    if (faces.isNotEmpty()) {
                                                                        val face = faces.first() // Assume largest face/first detected is user
                                                                        val cropped = ImageProcessor.cropFace(frame, face.bounds)
                                                                        croppedFrames.add(cropped)
                                                                        framesWithFace++
                                                                    } else {
                                                                        // Fallback: Use center crop or original? 
                                                                        // Let's use original for now but log it. 
                                                                        // Or better, skip? If we skip, we might have too few frames.
                                                                        // Let's keep original but scaled.
                                                                        croppedFrames.add(frame)
                                                                    }
                                                                }
                                                                
                                                                withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                    debugLog = "Đã crop $framesWithFace/${frames.size} frames video.\n$debugLog"
                                                                }

                                                                // Prefer ID card bitmap passed from shared ViewModel, fallback to capturedImage if needed
                                                                val idBmp = croppedImage ?: capturedImage
                                                                if (idBmp == null) {
                                                                    withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                        sendError = "Không có ảnh CCCD để ghép với video"

                                                                    }
                                                                    return@withContext
                                                                }
                                                                
                                                                // --- FACE DETECTION & CROPPING (ID CARD) ---
                                                                var finalIdBmp = idBmp
                                                                val idFaces = faceDetector.detect(idBmp)
                                                                if (idFaces.isNotEmpty()) {
                                                                    val face = idFaces.first()
                                                                    finalIdBmp = ImageProcessor.cropFace(idBmp, face.bounds)
                                                                    withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                        debugLog = "Đã detect & crop mặt từ ảnh CCCD.\n$debugLog"
                                                                    }
                                                                } else {
                                                                    withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                        debugLog = "CẢNH BÁO: Không tìm thấy mặt trong ảnh CCCD. Dùng ảnh gốc.\n$debugLog"
                                                                    }
                                                                }
                                                                
                                                                // ✅ RUN ACTUAL MODEL INFERENCE
                                                                Log.d("FaceScanScreen", "🔄 Running model inference with ${croppedFrames.size} frames and ID bitmap")
                                                                val result = modelManager.runInference(croppedFrames, finalIdBmp)
                                                                
                                                                withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                    result.onSuccess { ekycResult ->
                                                                        Log.d("FaceScanScreen", "✅ Model inference success: $ekycResult")
                                                                        inferenceResult = ekycResult
                                                                        
                                                                        val livenessThreshold = 0.9f
                                                                        val matchingThreshold = 0.7f
                                                                        
                                                                        if (ekycResult.livenessProb > livenessThreshold && 
                                                                            ekycResult.matchingScore > matchingThreshold) {
                                                                            approvalStatus = 1 // Approved
                                                                            sendError = null
                                                                            debugLog = "✅ Xác thực thành công!\nLiveness: ${ekycResult.livenessProb}\nMatching: ${ekycResult.matchingScore}\n"
                                                                        } else {
                                                                            approvalStatus = 0 // Failed
                                                                            sendError = "❌ Xác thực thất bại:\nLiveness: ${ekycResult.livenessProb} (threshold: $livenessThreshold)\nMatching: ${ekycResult.matchingScore} (threshold: $matchingThreshold)"
                                                                            debugLog = sendError + "\n"
                                                                        }
                                                                    }.onFailure { e ->
                                                                        approvalStatus = 0
                                                                        sendError = "❌ Model Error: ${e.message}"
                                                                        Log.e("FaceScanScreen", "❌ Model inference failed: ${e.message}", e)
                                                                        debugLog = "LỖI Xử lý model: ${e.message}\n$debugLog"
                                                                    }
                                                                    randomDigits = generateRandomDigits()
                                                                }
                                                            }
                                                        } catch (e: Exception) {
                                                            withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                                sendError = e.message

                                                            }
                                                        } finally {
                                                            isProcessing = false
                                                        }
                                                    }
                                                } else {
                                                    recording?.close()
                                                    recording = null

                                                }
                                            }
                                        }
                                    }
                                recording = activeRecording
                            } catch (e: Exception) {

                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = !isRecording && !isProcessing,
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text("Bắt đầu quay")
                    }
                    
                    Spacer(modifier = Modifier.width(16.dp))
                    
                    Button(
                        onClick = {
                            recording?.stop()
                            recording = null
                        },
                        modifier = Modifier.weight(1f),
                        enabled = isRecording,
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                    ) {
                        Text("Dừng quay")
                    }
                }

            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))

    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onImageCaptured: (Bitmap) -> Unit,
    onVideoCaptureReady: (VideoCapture<Recorder>) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context) }
    var imageCapture: ImageCapture? by remember { mutableStateOf(null) }

    LaunchedEffect(cameraProviderFuture) {
        val cameraProvider = cameraProviderFuture.get()
        
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
            
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.SD)) // Use SD for faster processing
            .build()
        val videoCapture = VideoCapture.withOutput(recorder)
        onVideoCaptureReady(videoCapture)
        
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
        
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageCapture,
                videoCapture
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    Column(modifier = modifier) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )
        
        Spacer(modifier = Modifier.height(16.dp))
    }
}

private fun extractFramesFromVideo(context: Context, videoUri: Uri, targetFrameCount: Int): List<Bitmap> {
    val retriever = MediaMetadataRetriever()
    try {
        retriever.setDataSource(context, videoUri)
        val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val durationMs = durationStr?.toLongOrNull() ?: 0L
        
        if (durationMs == 0L) return emptyList()
        
        val frames = mutableListOf<Bitmap>()
        // Use a slightly smaller step to ensure we fit in duration, or handle last frame
        val step = if (targetFrameCount > 1) durationMs / targetFrameCount else durationMs
        
        for (i in 0 until targetFrameCount) {
            val timeUs = (i * step) * 1000L
            // OPTION_CLOSEST is more accurate than OPTION_CLOSEST_SYNC but might be slower. 
            // For eKYC, accuracy is preferred.
            val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
            if (bitmap != null) {
                // Resize to 640px width (maintain aspect ratio) to ensure face detection works well
                val targetWidth = 640
                val targetHeight = (bitmap.height * (targetWidth.toFloat() / bitmap.width)).toInt()
                val resized = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
                frames.add(resized)
                if (bitmap != resized) bitmap.recycle()
            } else {
                // If extraction fails, duplicate the last frame if available
                if (frames.isNotEmpty()) {
                     val last = frames.last()
                     val dup = Bitmap.createBitmap(last)
                     frames.add(dup)
                }
            }
        }
        
        // Final padding if we still don't have enough frames
        while (frames.size < targetFrameCount && frames.isNotEmpty()) {
             val last = frames.last()
             val dup = Bitmap.createBitmap(last)
             frames.add(dup)
        }
        
        return frames
    } catch (e: Exception) {
        e.printStackTrace()
        return emptyList()
    } finally {
        retriever.release()
    }
}



private fun generateRandomDigits(): String {
    val rnd = java.util.Random()
    val sb = StringBuilder()
    repeat(5) { sb.append(rnd.nextInt(10)) }
    return sb.toString()
}

private fun sampleOrPadFrames(frames: List<Bitmap>, count: Int): List<Bitmap> {
    if (frames.size == count) return frames
    if (frames.isEmpty()) return List(count) { Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888) }
    if (frames.size > count) {
        // Uniformly sample 'count' frames
        val step = frames.size.toDouble() / count
        val out = mutableListOf<Bitmap>()
        var idx = 0.0
        repeat(count) {
            out.add(frames[(idx).toInt()])
            idx += step
        }
        return out
    }
    // frames.size < count -> pad by repeating last frame
    val out = frames.toMutableList()
    while (out.size < count) out.add(frames.last())
    return out
}
