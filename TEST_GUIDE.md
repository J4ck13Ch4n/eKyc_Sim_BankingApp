# 🚀 Quick Start Testing Guide

## 📋 Kiểm Tra Trước Test

### 1. Xác Nhận Model File
```bash
# Check model exists in assets
android/app/src/main/assets/ekyc_model_mobile.ptl
```
- [ ] File tồn tại
- [ ] Size > 0 MB
- [ ] Format: TorchScript (.ptl)

### 2. Kiểm Tra Dependencies
```gradle
// In android/app/build.gradle.kts
implementation("org.pytorch:pytorch_android_lite:2.1.0")
implementation("org.pytorch:pytorch_android_torchvision_lite:2.1.0")
```
- [ ] PyTorch dependencies added
- [ ] Build succeeds without errors

---

## 🧪 Test Scenarios

### Scenario 1: Happy Path (Success)
**Mục tiêu:** Verify model gives high scores

**Steps:**
1. Chạy app, choose "Đăng ký"
2. Chụp ảnh CCCD rõ ràng, chất lượng tốt
3. Confirm thông tin CCCD
4. Quay video khuôn mặt (5-10 giây)
   - Đọc các số được hiển thị
   - Nhìn thẳng vào camera
5. Nhấn "Bắt đầu quay"

**Expected Output:**
```
[Logs]
🔄 Running model inference with 8 frames and ID bitmap
ID Tensor: size=150528, mean≈0.0, std≈1.0
Frames Tensor: size=1204224, mean≈0.0, std≈1.0
Running forward pass with ID shape [1,3,224,224] and Video shape [1,8,3,224,224]...
✅ Model inference success: EkycResult(livenessProb=0.8X, matchingScore=0.8X)
✅ Xác thực thành công!
Liveness: 0.8X
Matching: 0.8X

[UI]
→ Proceed to ZKP enrollment screen
→ "Đang tạo Zero-Knowledge Proof..."
→ Show enrollment payload
```

**Verification Checklist:**
- [ ] No exceptions in logcat
- [ ] Both scores > 0.5
- [ ] ZKP enrollment triggered
- [ ] No "LỖI" (Error) messages

---

### Scenario 2: Low Liveness (Failure)
**Mục tiêu:** Verify threshold detection

**Steps:**
1. Quay video but **hướng khuôn mặt sang một bên** (>30 degrees)
2. Hoặc quay video **không nhìn vào camera**

**Expected Output:**
```
[Logs]
✅ Model inference success: EkycResult(livenessProb=0.2X, matchingScore=0.7X)
❌ Xác thực thất bại:
Liveness: 0.2X (threshold: 0.5)
Matching: 0.7X (threshold: 0.5)

[UI]
→ Red error message: "❌ Xác thực thất bại: Liveness: 0.2X (threshold: 0.5)"
→ Button "Chạy lại" available
→ NO ZKP enrollment
```

**Verification Checklist:**
- [ ] Liveness score is low (< 0.5)
- [ ] Matching score may be high
- [ ] Error message displayed clearly
- [ ] Can retry recording

---

### Scenario 3: Low Matching (ID Mismatch)
**Mục tiêu:** Verify ID-Face matching detection

**Steps:**
1. Chụp ID card of person A
2. Quay video of person B (different person)

**Expected Output:**
```
✅ Model inference success: EkycResult(livenessProb=0.85, matchingScore=0.2X)
❌ Xác thực thất bại:
Liveness: 0.85 (threshold: 0.5) ✓
Matching: 0.2X (threshold: 0.5) ✗

[UI]
→ Matching score is too low
→ Error message displayed
```

---

### Scenario 4: Model Failure
**Mục tiêu:** Verify error handling

**Steps:**
1. Delete or corrupt model file temporarily
2. Run inference

**Expected Output:**
```
[Logs]
Error loading model: Failed to load model ekyc_model_mobile.ptl

-- or --

[Logs]
❌ Model inference failed: Tensor shape mismatch
❌ Model Error: Tensor shape mismatch

[UI]
→ sendError = "Model Error: ..."
→ User sees error in red
→ Can tap "Chụp lại"
```

---

## 🔍 Debugging with Logcat

### Filter by Tags
```bash
# Android Studio Terminal
adb logcat EkycModelManager:V FaceScanScreen:V *:S
```

### What to Look For
```
✅ Good Logs:
D/EkycModelManager: ID Tensor: size=150528, mean≈0.0
D/EkycModelManager: Frames Tensor: size=1204224, mean≈0.0
D/EkycModelManager: Running forward pass...
D/EkycModelManager: ✅ Inference success: EkycResult(...)

❌ Bad Logs:
E/EkycModelManager: Module is null
E/EkycModelManager: ❌ Model inference failed
E/FaceScanScreen: ❌ Model Error:
D/EkycModelManager: Unexpected model output type
```

---

## 📊 Metrics to Monitor

### Tensor Statistics
```
Expected:
- ID Tensor mean ≈ 0.0 (normalized)
- ID Tensor std ≈ 1.0 (normalized)
- Frames Tensor mean ≈ 0.0
- Frames Tensor std ≈ 1.0

Bad Signs:
- mean >> 1.0 (pixels not normalized)
- std ≈ 0 (all same values)
- negative values > -5 (normalization off)
- positive values > 5 (normalization off)
```

### Score Distribution
```
Good Pass:
- Liveness: 0.7-0.99
- Matching: 0.7-0.99

Borderline:
- Liveness: 0.4-0.7
- Matching: 0.4-0.7

Clear Fail:
- Liveness: 0.0-0.4
- Matching: 0.0-0.4
```

---

## 🎯 Checklist Before Production

- [ ] Model file exists and is readable
- [ ] PyTorch dependencies resolve correctly
- [ ] Scenario 1 (Happy Path) passes
- [ ] Scenario 2 (Low Liveness) fails gracefully
- [ ] Scenario 3 (Low Matching) fails gracefully
- [ ] Scenario 4 (Model Error) handled with user message
- [ ] Logcat shows detailed tensor info
- [ ] Thresholds (0.5, 0.5) appropriate for your use case
- [ ] ZKP enrollment flows correctly after approval
- [ ] No unhandled exceptions

---

## 🔧 Troubleshooting

### Problem: Model crashes immediately
**Solution:**
1. Check model file path is correct
2. Ensure PyTorch libraries are loaded
3. Check Android version >= 24 (minSdk requirement)

### Problem: Tensor shape mismatch
**Solution:**
1. Verify input shapes in logs match [1,3,224,224] and [1,T,3,224,224]
2. Check frame count T > 0
3. Verify bitmap is RGB (not grayscale)

### Problem: Scores are NaN or Inf
**Solution:**
1. Check tensor mean/std in logs are reasonable
2. Verify pixel normalization is correct
3. Check for division by zero in preprocessing

### Problem: Scores are always 0 or 1
**Solution:**
1. Check if model is actually running (not simulation)
2. Verify model output extraction with dataAsFloatArray[0]
3. Check tuple indexing [0] and [1] for correct scores

---

## 📞 Support

When reporting issues, include:
1. Full logcat output (filter: EkycModelManager)
2. Which scenario failed
3. Expected vs actual output
4. Device model and Android version
5. Exact error message shown to user

---

## ✅ Sign-Off

After all tests pass:
```
✅ Model implementation verified
✅ Android <-> Python consistency confirmed
✅ Error handling works correctly
✅ Production ready
```
