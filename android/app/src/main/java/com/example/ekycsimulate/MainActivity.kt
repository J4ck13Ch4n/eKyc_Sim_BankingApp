// In file: app/src/main/java/com/example/ekycsimulate/MainActivity.kt
package com.example.ekycsimulate

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.ekycsimulate.ui.auth.ConfirmInfoScreen
import com.example.ekycsimulate.ui.auth.LandingScreen
import com.example.ekycsimulate.ui.theme.EkycSimulateTheme
// Sửa lại đường dẫn import cho đúng
import com.example.ekycsimulate.ui.viewmodel.EkycViewModel
import com.example.ekycsimulate.ui.auth.FaceScanScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EkycSimulateTheme {
                // Gọi hàm điều hướng chính của ứng dụng
                AppNavigation()
            }
        }
    }
}

// Định nghĩa các "đường dẫn" đến màn hình để tránh gõ sai
object AppRoutes {
    const val LANDING = "landing"
    const val EKYC_CAMERA = "ekyc_camera"
    const val CONFIRM_INFO = "confirm_info"
    const val FACE_SCAN = "face_scan"
    const val LOGIN = "login"
    const val HOME = "home"
}

@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    // Tạo một instance của ViewModel. Nó sẽ được chia sẻ cho tất cả các màn hình trong NavHost này.
    val ekycViewModel: EkycViewModel = viewModel()

    NavHost(
        navController = navController,
        startDestination = AppRoutes.LANDING // Màn hình bắt đầu
    ) {
        // 1. Định nghĩa màn hình Chào mừng
        composable(route = AppRoutes.LANDING) {
            LandingScreen(
                onLoginClicked = {
                    navController.navigate(AppRoutes.LOGIN)
                },
                onRegisterClicked = {
                    // Khi nhấn nút Đăng ký, chuyển đến màn hình camera
                    navController.navigate(AppRoutes.EKYC_CAMERA)
                }
            )
        }

        // 2. Định nghĩa màn hình Camera (CHỈ CÓ MỘT ĐỊNH NGHĨA DUY NHẤT)
        composable(route = AppRoutes.EKYC_CAMERA) {
            EkycCameraScreen(
                // Callback này được gọi sau khi ảnh đã được chụp và cắt thành công
                onImageCropped = { croppedBitmap ->
                    // 1. Lưu ảnh Bitmap đã cắt vào instance của SharedViewModel
                    ekycViewModel.croppedImage = croppedBitmap
                    Log.d("AppNavigation", "Đã nhận và lưu Bitmap đã cắt.")

                    // 2. Điều hướng sang màn hình xác nhận thông tin
                    navController.navigate(AppRoutes.CONFIRM_INFO)
                }
            )
        }

        // 3. Định nghĩa màn hình xác nhận thông tin (CHỈ CÓ MỘT ĐỊNH NGHĨA DUY NHẤT)
        composable(route = AppRoutes.CONFIRM_INFO) {
            // Lấy ảnh đã cắt từ SharedViewModel
            val bitmap = ekycViewModel.croppedImage

            if (bitmap != null) {
                ConfirmInfoScreen(
                    croppedBitmap = bitmap,
                    onNextStep = { idCardInfo ->
                        Log.d("EKYC_Flow", "Xác nhận thông tin thành công: $idCardInfo")
                        // Lưu thông tin vào ViewModel để dùng ở bước sau
                        ekycViewModel.idCardInfo = idCardInfo
                        // Điều hướng đến bước Face Scan
                        navController.navigate(AppRoutes.FACE_SCAN)
                    },
                    onRetake = {
                        ekycViewModel.croppedImage = null
                        navController.popBackStack()
                    }
                )
            } else {
                // Xử lý trường hợp người dùng vào màn hình này mà không có ảnh (ví dụ: do process death).
                // An toàn nhất là quay lại màn hình trước đó.
                Log.e("AppNavigation", "Bitmap bị null, quay lại màn hình trước.")
                navController.popBackStack()
            }
        }

        // 4. Định nghĩa màn hình Face Scan
        composable(route = AppRoutes.FACE_SCAN) {
            val idInfo = ekycViewModel.idCardInfo
            if (idInfo != null) {
                FaceScanScreen(
                    idCardInfo = idInfo,
                    croppedImage = ekycViewModel.croppedImage,
                    onEnrollmentComplete = { jsonPayload ->
                        // Payload is ready to send to server
                        Log.d("EKYC_ZKP", "Enrollment Payload:\n$jsonPayload")
                        
                        // TODO: Send to server API
                        // Example: apiService.enrollUser(jsonPayload)
                        
                        // For now, just complete the flow
                        ekycViewModel.croppedImage = null
                        ekycViewModel.idCardInfo = null
                        
                        // Parse userId from response if possible, or default to 0
                        // Since onEnrollmentComplete currently returns JSON string, we might need to parse it
                        // For simplicity, let's assume success and navigate to Home with ID 0 or parse it
                        // Ideally onEnrollmentComplete should pass the EnrollmentResponse object
                        
                    // Navigate to Home
                    navController.navigate("${AppRoutes.HOME}/0") {
                        popUpTo(AppRoutes.LANDING) { inclusive = false }
                    }
                    }
                )
            } else {
                // Missing info, go back
                navController.popBackStack()
            }
        }

        // 5. Định nghĩa màn hình Đăng nhập
        composable(route = AppRoutes.LOGIN) {
            com.example.ekycsimulate.ui.auth.LoginScreen(
                onLoginSuccess = { userId ->
                    Log.d("EKYC_ZKP", "Login Success: $userId")
                    // Navigate to Home
                    navController.navigate("${AppRoutes.HOME}/$userId") {
                        popUpTo(AppRoutes.LANDING) { inclusive = false }
                    }
                }
            )
        }

        // 6. Định nghĩa màn hình Trang chủ
        composable(route = "${AppRoutes.HOME}/{userId}") { backStackEntry ->
            val userId = backStackEntry.arguments?.getString("userId")?.toIntOrNull() ?: 0
            com.example.ekycsimulate.ui.home.HomeScreen(
                userId = userId,
                onLogout = {
                    // Logout logic
                    val enrollmentManager = com.example.ekycsimulate.zkp.ZKPEnrollmentManager(navController.context)
                    // Optional: enrollmentManager.clearEnrollment() // Only if you want to clear keys
                    navController.navigate(AppRoutes.LANDING) {
                        popUpTo(0) // Clear back stack
                    }
                }
            )
        }
    }
}
