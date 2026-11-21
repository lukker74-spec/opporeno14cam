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
    NORMAL(
        id = "normal",
        displayName = "Normale",
        accentColor = Color(0xFF4FC3F7),
        contrast = 1.06f,
        saturation = 1.06f,
        warmth = 0.01f,
        shadowLift = 0.05f
    ),
    LEICA(
        id = "leica",
        displayName = "Leica",
        accentColor = Color(0xFFE53935),
        contrast = 1.15f,
        saturation = 0.95f,
        warmth = 0.04f,
        shadowLift = 0.01f
    ),
    ZEISS(
        id = "zeiss",
        displayName = "Zeiss",
        accentColor = Color(0xFF1E88E5),
        contrast = 1.10f,
        saturation = 1.04f,
        warmth = -0.02f,
        shadowLift = 0.04f
    ),
    HASSELBLAD(
        id = "hasselblad",
        displayName = "Hasselblad",
        accentColor = Color(0xFFFF7043),
        contrast = 1.07f,
        saturation = 1.06f,
        warmth = 0.05f,
        shadowLift = 0.05f
    ),
    PIXEL(
        id = "pixel",
        displayName = "Pixel",
        accentColor = Color(0xFF66BB6A),
        contrast = 1.12f,
        saturation = 1.07f,
        warmth = 0.0f,
        shadowLift = 0.07f
    ),
    KODAK(
        id = "kodak",
        displayName = "Kodak",
        accentColor = Color(0xFFFFCA28),
        contrast = 1.06f,
        saturation = 1.11f,
        warmth = 0.10f,
        shadowLift = 0.06f
    ),
    IPHONE(
        id = "iphone",
        displayName = "iPhone",
        accentColor = Color(0xFFECEFF1),
        contrast = 1.05f,
        saturation = 1.05f,
        warmth = 0.02f,
        shadowLift = 0.05f
    ),
    VIVO_V70(
        id = "vivo_v70",
        displayName = "Vivo V70",
        accentColor = Color(0xFF26C6DA),
        contrast = 1.09f,
        saturation = 1.08f,
        warmth = 0.03f,
        shadowLift = 0.04f
    );

    fun toColorMatrix(strength: Float = 1.0f, brightnessOffset: Float = 0.06f): ColorMatrix {
        val c = 1f + (contrast - 1f) * strength
        val s = 1f + (saturation - 1f) * strength
        val w = warmth * strength
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
        val warmthR = (w * 24f).coerceIn(-30f, 30f)
        val warmthB = (-w * 24f).coerceIn(-30f, 30f)

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

    fun applyToBitmap(source: Bitmap, strength: Float = 1.0f, brightnessBoost: Float = 0.06f): Bitmap {
        val cm = toColorMatrix(strength, brightnessBoost)
        val filter = ColorMatrixColorFilter(android.graphics.ColorMatrix(cm.values))
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { colorFilter = filter }
        canvas.drawBitmap(source, 0f, 0f, paint)
        return out
    }
}

enum class AiMode(val label: String, val shortLabel: String) {
    LOW("AI Basso", "Basso"),
    MEDIUM("AI Media", "Medio"),
    AGGRESSIVE("AI Alto", "Alto")
}

enum class CaptureFormat(val label: String, val badge: String) {
    NORMAL("Normale", "JPG"),
    RAW("RAW", "RAW"),
    PRO("Pro", "PRO")
}

enum class LensType(val label: String) {
    ULTRAWIDE("Ultra-Grandangolo 8MP"),
    MAIN_50MP("Principale 50MP OIS"),
    TELE_50MP("Teleobiettivo 50MP 3.5x OIS"),
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
    TELE_50MP("Tele 50MP"),
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
    val profile: CameraProfile = CameraProfile.NORMAL,
    val captureFormat: CaptureFormat = CaptureFormat.NORMAL,
    val aiMode: AiMode = AiMode.MEDIUM,
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
    val maxLensZoomRatio: Float = 17.2f,
    val brightnessBoost: Float = 0.08f,
    val evCompensation: Int = 1,
    val profileIntensity: Float = 1.0f,
    val isoValue: Int = 160,
    val shutterSpeedDenom: Int = 125,
    val stabilizationEnabled: Boolean = true,
    val neuralSuperResEnabled: Boolean = true,
    val cloudAiEnabled: Boolean = false,
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
