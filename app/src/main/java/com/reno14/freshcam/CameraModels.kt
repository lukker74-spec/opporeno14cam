package com.reno14.freshcam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorMatrix

val PRO_ISO_STOPS = listOf(50, 100, 160, 200, 320, 400, 640, 800, 1250, 1600, 3200)
val PRO_SHUTTER_DENOMS = listOf(4, 8, 15, 25, 30, 50, 60, 100, 125, 250, 500, 1000, 2000)

enum class CameraProfile(
    val id: String,
    val displayName: String,
    val accentColor: Color,
    val contrast: Float,
    val saturation: Float,
    val warmth: Float,
    val shadowLift: Float
) {
    NATURAL("naturale", "Naturale", Color(0xFF4FC3F7), 1.01f, 1.01f, 0.0f, 0.0f),
    VIVID("vivido", "Vivido", Color(0xFFFFA726), 1.10f, 1.22f, 0.015f, 0.005f),
    BW("bianco_nero", "Bianco e Nero", Color(0xFFE0E0E0), 1.18f, 0.0f, 0.0f, -0.01f),
    LEICA("leica", "Leica", Color(0xFFE53935), 1.08f, 0.99f, 0.015f, 0.0f),
    ZEISS("zeiss", "Zeiss", Color(0xFF1E88E5), 1.05f, 1.02f, -0.01f, 0.005f),
    HASSELBLAD("hasselblad", "Hasselblad", Color(0xFFFF7043), 1.04f, 1.04f, 0.02f, 0.005f),
    PIXEL("pixel", "Pixel", Color(0xFF66BB6A), 1.06f, 1.05f, 0.0f, 0.015f),
    KODAK("kodak", "Kodak", Color(0xFFFFCA28), 1.03f, 1.06f, 0.03f, 0.01f);

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
        val warmthR = (w * 12f).coerceIn(-16f, 16f)
        val warmthB = (-w * 12f).coerceIn(-16f, 16f)

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

enum class LocalNeuralModel(
    val shortName: String,
    val displayName: String,
    val assetPath: String
) {
    CLEAR_REALITY_SPAN(
        shortName = "SPAN",
        displayName = "ClearReality SPAN 4x (Zoom 7x-120x)",
        assetPath = "models/4x-ClearRealityV1.onnx"
    )
}

enum class UnifiedAiPreset(
    val shortBadge: String,
    val fullLabel: String,
    val isEnabled: Boolean,
    val isCloud: Boolean,
    val level: Int
) {
    OFF("AI Off", "AI Spenta (Scatto Diretto Veloce)", false, false, 0),
    LOCAL_LOW("AI Bassa", "AI Locale · Bassa (HDR + Dettaglio Naturale)", true, false, 1),
    LOCAL_MED("AI Media", "AI Locale · Media (HDR + Ombre + Nitidezza)", true, false, 2),
    LOCAL_HIGH("AI Alta", "AI Locale · Alta (Massimo Contrasto & Dettaglio)", true, false, 3),
    CLOUD_FREE_MED("Cloud Gratis", "Cloud Gratis HF · Media", true, true, 2),
    CLOUD_FREE_HIGH("Cloud Max", "Cloud Gratis HF · Alta", true, true, 3)
}

enum class FreeCloudPreset(val label: String, val url: String) {
    HF_SWIN2SR_FREE(
        "Hugging Face Gratis (Swin2SR · Token hf_...)",
        "https://router.huggingface.co/hf-inference/models/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
    ),
    CUSTOM_FREE(
        "URL Personalizzato (es. Space ZeroGPU / Server gratuito)",
        ""
    )
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

data class PhysicalLensInfo(
    val cameraId: String,
    val lensType: LensType,
    val focalLengthMm: Float
)

data class FreshCamState(
    val profile: CameraProfile = CameraProfile.NATURAL,
    val captureFormat: CaptureFormat = CaptureFormat.NORMAL,
    // v19: AI Locale e Cloud SPENTE all'avvio per scatti immediati e zero overhead
    val aiPreset: UnifiedAiPreset = UnifiedAiPreset.OFF,
    val localNeuralModel: LocalNeuralModel = LocalNeuralModel.CLEAR_REALITY_SPAN,
    val preferNnapiNpu: Boolean = true,
    val localAiBackendStatus: String = "Standby (si carica solo su Zoom >= 7x)",
    // v19: RAW-in-RAM spento all'avvio (attivabile opzionalmente dalle Impostazioni)
    val rawInRamEnabled: Boolean = false,
    val backgroundProcessingEnabled: Boolean = true,
    val backgroundJobsCount: Int = 0,
    val freeCloudPreset: FreeCloudPreset = FreeCloudPreset.HF_SWIN2SR_FREE,
    val cloudApiKey: String = "",
    val cloudEndpointUrl: String = FreeCloudPreset.HF_SWIN2SR_FREE.url,
    val showCloudSetupDialog: Boolean = false,
    val timerMode: TimerMode = TimerMode.OFF,
    val countdownRemaining: Int = 0,
    val videoResolution: VideoResolution = VideoResolution.UHD_4K,
    val jpegQuality: Int = 97,
    val activeLens: LensType = LensType.MAIN_50MP,
    val detectedLenses: List<PhysicalLensInfo> = emptyList(),
    val forceLogicalHwExtensions: Boolean = true,
    val activeHardwareExtension: String? = null,
    val hwExtensionsReport: String = "Scansione estensioni HW...",
    val sensorCapabilitiesReport: String = "Scansione sensori...",
    // v19: Parte in Autofocus (AF), mantenendo scelta tra AF, Manuale e AI Lock
    val focusMode: FocusMode = FocusMode.AUTO,
    val isSubjectLocked: Boolean = false,
    val manualFocusDistance: Float = 0.5f,
    val shootingMode: ShootingMode = ShootingMode.PHOTO,
    // v19: Flash OFF di default per evitare pre-capture lenti in interno
    val flashMode: FlashMode = FlashMode.OFF,
    val displayZoom: Float = 1.0f,
    val lensZoomRatio: Float = 1.0f,
    val maxLensZoomRatio: Float = 34.3f,
    val brightnessBoost: Float = 0.0f,
    val evCompensation: Int = 0,
    val profileIntensity: Float = 1.0f,
    val isoValue: Int = 400,
    val shutterSpeedDenom: Int = 60,
    val stabilizationEnabled: Boolean = true,
    val postSavePolishEnabled: Boolean = false,
    val gridEnabled: Boolean = true,
    val locationTagEnabled: Boolean = false,
    val hasLocationPermission: Boolean = false,
    val isRecordingVideo: Boolean = false,
    val isCapturing: Boolean = false,
    val lastCapturedUri: String? = null,
    val bannerMessage: String? = null,
    val persistentStatus: String? = null
)
