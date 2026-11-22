package com.reno14.freshcam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorMatrix

enum class CameraProfile(
    val id: String,
    val displayName: String,
    val accentColor: Color,
    val contrast: Float,
    val saturation: Float,
    val warmth: Float,
    val shadowLift: Float
) {
    NATURAL("naturale", "Naturale", Color(0xFF4FC3F7), 1.02f, 1.01f, 0.0f, 0.005f),
    VIVID("vivido", "Vivido", Color(0xFFFFA726), 1.11f, 1.24f, 0.02f, 0.01f),
    BW("bianco_nero", "Bianco e Nero", Color(0xFFE0E0E0), 1.18f, 0.0f, 0.0f, -0.01f),
    LEICA("leica", "Leica", Color(0xFFE53935), 1.08f, 0.99f, 0.015f, 0.0f),
    ZEISS("zeiss", "Zeiss", Color(0xFF1E88E5), 1.05f, 1.02f, -0.01f, 0.01f),
    HASSELBLAD("hasselblad", "Hasselblad", Color(0xFFFF7043), 1.04f, 1.04f, 0.02f, 0.01f),
    PIXEL("pixel", "Pixel", Color(0xFF66BB6A), 1.06f, 1.05f, 0.0f, 0.02f),
    KODAK("kodak", "Kodak", Color(0xFFFFCA28), 1.03f, 1.06f, 0.035f, 0.015f);

    fun toColorMatrix(strength: Float = 1.0f, brightnessOffset: Float = 0.0f): ColorMatrix {
        val c = 1f + (contrast - 1f) * strength
        val s = if (this == BW) 0f else (1f + (saturation - 1f) * strength).coerceAtLeast(0f)
        val w = if (this == BW) 0f else warmth * strength
        val lift = (shadowLift * strength + brightnessOffset) * 255f

        val invSat = 1f - s
        val satMatrix = ColorMatrix(
            floatArrayOf(
                invSat * 0.2126f + s, invSat * 0.7152f, invSat * 0.0722f, 0f, 0f,
                invSat * 0.2126f, invSat * 0.7152f + s, invSat * 0.0722f, 0f, 0f,
                invSat * 0.2126f, invSat * 0.7152f, invSat * 0.0722f + s, 0f, 0f,
                0f, 0f, 0f, 1f, 0f
            )
        )

        val translate = (1f - c) * 128f + lift
        val warmthR = (w * 14f).coerceIn(-18f, 18f)
        val warmthB = (-w * 14f).coerceIn(-18f, 18f)

        val cwMatrix = ColorMatrix(
            floatArrayOf(
                c, 0f, 0f, 0f, translate + warmthR,
                0f, c, 0f, 0f, translate,
                0f, 0f, c, 0f, translate + warmthB,
                0f, 0f, 0f, 1f, 0f
            )
        )
        satMatrix.timesAssign(cwMatrix)
        return satMatrix
    }

    fun applyToBitmap(source: Bitmap, strength: Float = 1.0f, brightnessBoost: Float = 0.0f): Bitmap {
        val cm = toColorMatrix(strength, brightnessBoost)
        val filter = ColorMatrixColorFilter(android.graphics.ColorMatrix(cm.values))
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { colorFilter = filter }
        canvas.drawBitmap(source, 0f, 0f, paint)
        return out
    }
}

/**
 * Unico selettore AI unificato (7 stati):
 * Off · Locale (Bassa / Media / Alta) · Cloud (Bassa / Media / Alta)
 */
enum class UnifiedAiPreset(
    val shortBadge: String,
    val fullLabel: String,
    val isEnabled: Boolean,
    val isCloud: Boolean,
    val level: Int // 0 = off, 1 = bassa, 2 = media, 3 = alta
) {
    OFF("AI Off", "AI Disattivata", false, false, 0),
    LOCAL_LOW("AI Bassa", "Locale · Bassa", true, false, 1),
    LOCAL_MED("AI Media", "Locale · Media", true, false, 2),
    LOCAL_HIGH("AI Alta", "Locale · Alta", true, false, 3),
    CLOUD_LOW("Cloud Bassa", "Cloud · Bassa", true, true, 1),
    CLOUD_MED("Cloud Media", "Cloud · Media", true, true, 2),
    CLOUD_HIGH("Cloud Alta", "Cloud · Alta", true, true, 3)
}

enum class TimerMode(val seconds: Int, val label: String) {
    OFF(0, "Off"),
    SEC_3(3, "3s"),
    SEC_5(5, "5s"),
    SEC_10(10, "10s")
}

enum class VideoResolution(val label: String) {
    UHD_4K("4K UHD"),
    FHD_1080P("1080p Full HD"),
    HD_720P("720p HD")
}

enum class CaptureFormat(val label: String, val badge: String) {
    NORMAL("Normale", "JPG"),
    RAW("RAW", "RAW"),
    PRO("Pro", "PRO")
}

enum class LensType(val label: String) {
    ULTRAWIDE("Ultra-Grandangolo 8MP"),
    MAIN_50MP("Principale 50MP OIS"),
    TELE_50MP("Tele 50MP 3.5x OIS"),
    FRONT("Frontale 50MP")
}

enum class FocusMode(val label: String, val shortLabel: String) {
    AUTO("Autofocus", "AF"),
    MANUAL("Manuale", "MF"),
    AI_LOCK("AI Lock Soggetto", "AI Lock")
}

enum class ShootingMode(val label: String) {
    PHOTO("Foto"),
    VIDEO("Video"),
    PORTRAIT("Ritratto"),
    NIGHT("Notte")
}

enum class FlashMode(val label: String) {
    OFF("Off"),
    AUTO("Auto"),
    ON("On"),
    TORCH("Torcia")
}

enum class HdrMode(val label: String) {
    OFF("No HDR"),
    NATURAL("HDR Naturale"),
    AUTO("HDR Auto")
}

data class PhysicalLensInfo(
    val cameraId: String,
    val lensType: LensType,
    val focalLengthMm: Float
)

data class FreshCamState(
    val profile: CameraProfile = CameraProfile.NATURAL,
    val captureFormat: CaptureFormat = CaptureFormat.NORMAL,
    val aiPreset: UnifiedAiPreset = UnifiedAiPreset.LOCAL_MED,
    val timerMode: TimerMode = TimerMode.OFF,
    val countdownRemaining: Int = 0,
    val videoResolution: VideoResolution = VideoResolution.UHD_4K,
    val jpegQuality: Int = 99,
    val activeLens: LensType = LensType.MAIN_50MP,
    val detectedLenses: List<PhysicalLensInfo> = emptyList(),
    val focusMode: FocusMode = FocusMode.AI_LOCK,
    val isSubjectLocked: Boolean = false,
    val manualFocusDistance: Float = 0.5f,
    val shootingMode: ShootingMode = ShootingMode.PHOTO,
    val flashMode: FlashMode = FlashMode.AUTO,
    val hdrMode: HdrMode = HdrMode.NATURAL,
    val displayZoom: Float = 1.0f,
    val lensZoomRatio: Float = 1.0f,
    val maxLensZoomRatio: Float = 34.3f,
    val brightnessBoost: Float = 0.0f,
    val evCompensation: Int = 0,
    val profileIntensity: Float = 1.0f,
    val isoValue: Int = 160,
    val shutterSpeedDenom: Int = 125,
    val stabilizationEnabled: Boolean = true,
    val cloudApiKey: String = "",
    val cloudEndpointUrl: String = "https://router.huggingface.co/hf-inference/models/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    val isNeuralModelReady: Boolean = false,
    val loadedModelName: String = "Nessuno",
    val activeAccelerator: String = "CPU",
    val postSavePolishEnabled: Boolean = true,
    val gridEnabled: Boolean = true,
    val locationTagEnabled: Boolean = false,
    val hasLocationPermission: Boolean = false,
    val isRecordingVideo: Boolean = false,
    val lastCapturedUri: String? = null,
    val bannerMessage: String? = null,
    val persistentStatus: String? = null
)
