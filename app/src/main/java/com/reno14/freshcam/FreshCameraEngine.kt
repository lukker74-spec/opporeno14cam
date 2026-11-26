package com.reno14.freshcam

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraExtensionCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.location.Location
import android.location.LocationManager
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraFilter
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.extensions.ExtensionMode
import androidx.camera.extensions.ExtensionsManager
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max

class FreshCameraEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private var extensionsManager: ExtensionsManager? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var camera: Camera? = null

    private val saveWorker = Executors.newSingleThreadExecutor()
    private val neuralEngine = NeuralSuperResEngine(context)

    var minFocusDistanceDiopters: Float = 0f
        private set

    var currentMaxHwZoomRatio: Float = 10.0f
        private set

    private var discoveredLenses: List<PhysicalLensInfo> = emptyList()

    fun discoverLenses(): List<PhysicalLensInfo> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val list = mutableListOf<PhysicalLensInfo>()
        try {
            val backCandidates = mutableListOf<Pair<String, Float>>()
            for (id in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
                val focal = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 4.8f
                if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    if (list.none { it.lensType == LensType.FRONT }) {
                        list.add(PhysicalLensInfo(id, LensType.FRONT, focal))
                    }
                } else if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    backCandidates.add(id to focal)
                }
            }

            val mainCam = backCandidates.firstOrNull { it.first == "0" && it.second in 3.5f..7.0f }
                ?: backCandidates.firstOrNull { it.second in 3.8f..6.8f }
                ?: backCandidates.minByOrNull { kotlin.math.abs(it.second - 5.0f) }

            val ultraCam = backCandidates
                .filter { it.first != mainCam?.first && it.second < (mainCam?.second ?: 4.5f) - 0.8f }
                .minByOrNull { it.second }

            val teleCam = backCandidates
                .filter { it.first != mainCam?.first && it.second > (mainCam?.second ?: 5.0f) + 2.0f }
                .maxByOrNull { it.second }

            mainCam?.let { list.add(PhysicalLensInfo(it.first, LensType.MAIN_50MP, it.second)) }
            ultraCam?.let { list.add(PhysicalLensInfo(it.first, LensType.ULTRAWIDE, it.second)) }
            teleCam?.let { list.add(PhysicalLensInfo(it.first, LensType.TELE_50MP, it.second)) }
        } catch (e: Exception) {
            Log.e("Reno14FreshCam", "Errore rilevamento fotocamere", e)
        }
        discoveredLenses = list
        return list
    }

    @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
    private fun selectorForLens(lensType: LensType): CameraSelector {
        if (lensType == LensType.FRONT) return CameraSelector.DEFAULT_FRONT_CAMERA
        val target = discoveredLenses.firstOrNull { it.lensType == lensType }
        if (target != null) {
            return CameraSelector.Builder()
                .addCameraFilter(CameraFilter { infos ->
                    val filtered = infos.filter { Camera2CameraInfo.from(it).cameraId == target.cameraId }
                    filtered.ifEmpty { infos }
                })
                .build()
        }
        return CameraSelector.DEFAULT_BACK_CAMERA
    }

    /**
     * Scansione profonda v15: verifica sia CameraX Extensions su DEFAULT_BACK_CAMERA (logica)
     * sia le Camera2 CameraExtensionCharacteristics (Android 12+) su ogni ID fisico!
     */
    private fun buildDeepExtensionsReport(extMgr: ExtensionsManager): String {
        fun probeCx(label: String, sel: CameraSelector): String {
            val found = mutableListOf<String>()
            if (extMgr.isExtensionAvailable(sel, ExtensionMode.HDR)) found.add("HDR")
            if (extMgr.isExtensionAvailable(sel, ExtensionMode.NIGHT)) found.add("Notte")
            if (extMgr.isExtensionAvailable(sel, ExtensionMode.AUTO)) found.add("Auto")
            if (extMgr.isExtensionAvailable(sel, ExtensionMode.BOKEH)) found.add("Bokeh")
            return "$label: " + if (found.isEmpty()) "Nessuna" else found.joinToString("/")
        }

        val logicalStr = probeCx("Logica", CameraSelector.DEFAULT_BACK_CAMERA)
        val teleStr = probeCx("Tele", selectorForLens(LensType.TELE_50MP))

        var c2Extra = ""
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                val mgr = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val c2Found = mutableListOf<String>()
                for (id in mgr.cameraIdList) {
                    val chars = mgr.getCameraCharacteristics(id)
                    if (chars.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) continue
                    val extChars = mgr.getCameraExtensionCharacteristics(id)
                    val modes = extChars.supportedExtensions
                    if (modes.isNotEmpty()) {
                        val names = modes.map { m ->
                            when (m) {
                                CameraExtensionCharacteristics.EXTENSION_AUTOMATIC -> "Auto"
                                CameraExtensionCharacteristics.EXTENSION_HDR -> "HDR"
                                CameraExtensionCharacteristics.EXTENSION_NIGHT -> "Notte"
                                CameraExtensionCharacteristics.EXTENSION_BOKEH -> "Bokeh"
                                CameraExtensionCharacteristics.EXTENSION_FACE_RETOUCH -> "Face"
                                else -> "#$m"
                            }
                        }
                        c2Found.add("ID$id[${names.joinToString(",")}]")
                    }
                }
                if (c2Found.isNotEmpty()) {
                    c2Extra = " | C2: ${c2Found.joinToString(" ")}"
                }
            } catch (_: Throwable) {
            }
        }
        return "$logicalStr · $teleStr$c2Extra"
    }

    private fun pickDesiredExtensionMode(state: FreshCamState, extMgr: ExtensionsManager, targetSelector: CameraSelector): Pair<Int, String>? {
        if (state.shootingMode == ShootingMode.VIDEO) return null
        if (state.captureFormat != CaptureFormat.NORMAL) return null
        if (!state.aiPreset.isEnabled) return null

        val candidates: List<Pair<Int, String>> = when (state.shootingMode) {
            ShootingMode.NIGHT -> listOf(
                ExtensionMode.NIGHT to "HW Notte",
                ExtensionMode.HDR to "HW HDR",
                ExtensionMode.AUTO to "HW Auto"
            )
            ShootingMode.PORTRAIT -> listOf(
                ExtensionMode.BOKEH to "HW Bokeh",
                ExtensionMode.HDR to "HW HDR",
                ExtensionMode.AUTO to "HW Auto"
            )
            else -> when (state.aiPreset.level) {
                // Su ColorOS 1x espone Notte/Bokeh: su AI Alta usiamo HW Notte come multi-frame hardware!
                3 -> listOf(
                    ExtensionMode.HDR to "HW HDR",
                    ExtensionMode.NIGHT to "HW Notte",
                    ExtensionMode.AUTO to "HW Auto"
                )
                2 -> listOf(
                    ExtensionMode.HDR to "HW HDR",
                    ExtensionMode.AUTO to "HW Auto"
                )
                else -> listOf(
                    ExtensionMode.AUTO to "HW Auto",
                    ExtensionMode.HDR to "HW HDR"
                )
            }
        }

        for (pair in candidates) {
            try {
                if (extMgr.isExtensionAvailable(targetSelector, pair.first)) {
                    return pair
                }
            } catch (_: Throwable) {
            }
        }
        return null
    }

    @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
    fun bind(
        previewView: PreviewView,
        state: FreshCamState,
        onZoomBounds: (minZ: Float, maxZ: Float) -> Unit,
        onExtensionUpdate: (activeExt: String?, report: String) -> Unit,
        onBanner: (String) -> Unit
    ) {
        if (discoveredLenses.isEmpty()) discoverLenses()

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider

                val extMgr = extensionsManager
                if (extMgr == null) {
                    val extFuture = ExtensionsManager.getInstanceAsync(context, provider)
                    extFuture.addListener({
                        val mgr = try { extFuture.get() } catch (_: Throwable) { null }
                        extensionsManager = mgr
                        bindInternal(provider, mgr, previewView, state, onZoomBounds, onExtensionUpdate, onBanner)
                    }, ContextCompat.getMainExecutor(context))
                } else {
                    bindInternal(provider, extMgr, previewView, state, onZoomBounds, onExtensionUpdate, onBanner)
                }
            } catch (e: Exception) {
                Log.e("Reno14FreshCam", "Errore inizializzazione CameraProvider", e)
                onBanner("Attivazione ${state.activeLens.label}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
    private fun bindInternal(
        provider: ProcessCameraProvider,
        extMgr: ExtensionsManager?,
        previewView: PreviewView,
        state: FreshCamState,
        onZoomBounds: (minZ: Float, maxZ: Float) -> Unit,
        onExtensionUpdate: (activeExt: String?, report: String) -> Unit,
        onBanner: (String) -> Unit
    ) {
        try {
            // Se forceLogicalHwExtensions è attivo e stiamo tentando una HW Extension,
            // proviamo sul selettore logico DEFAULT_BACK_CAMERA (dove ColorOS aggancia le estensioni!)
            val physicalSelector = selectorForLens(state.activeLens)
            val extCandidateSelector = if (state.activeLens != LensType.FRONT && state.forceLogicalHwExtensions) {
                CameraSelector.DEFAULT_BACK_CAMERA
            } else {
                physicalSelector
            }

            val report = if (extMgr != null) buildDeepExtensionsReport(extMgr) else "Estensioni HW non rilevate"
            val desiredExt = if (extMgr != null) pickDesiredExtensionMode(state, extMgr, extCandidateSelector) else null
            var activeExtLabel: String? = null
            var bound: Camera? = null

            if (extMgr != null && desiredExt != null) {
                try {
                    val extSelector = extMgr.getExtensionEnabledCameraSelector(extCandidateSelector, desiredExt.first)
                    val previewExt = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    val capExtBuilder = ImageCapture.Builder()
                        .setJpegQuality(state.jpegQuality.coerceIn(85, 100))
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    applyFlashBuilder(capExtBuilder, state.flashMode)
                    val capExt = capExtBuilder.build()

                    provider.unbindAll()
                    bound = provider.bindToLifecycle(lifecycleOwner, extSelector, previewExt, capExt)
                    imageCapture = capExt
                    activeExtLabel = desiredExt.second
                } catch (e: Throwable) {
                    Log.w("Reno14FreshCam", "Estensione HW ${desiredExt.second} rifiutata: ${e.message}")
                    activeExtLabel = null
                    bound = null
                }
            }

            if (bound == null) {
                val previewBuilder = Preview.Builder()
                val resSelector = ResolutionSelector.Builder()
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .build()

                val captureBuilder = ImageCapture.Builder()
                    .setResolutionSelector(resSelector)
                    .setJpegQuality(state.jpegQuality.coerceIn(85, 100))
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)

                val pExt = Camera2Interop.Extender(previewBuilder)
                val cExt = Camera2Interop.Extender(captureBuilder)

                val isTele = state.activeLens == LensType.TELE_50MP
                val nrMode = if (state.captureFormat == CaptureFormat.RAW) {
                    CaptureRequest.NOISE_REDUCTION_MODE_FAST
                } else {
                    CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY
                }
                val edgeMode = if (isTele) {
                    CaptureRequest.EDGE_MODE_HIGH_QUALITY
                } else {
                    CaptureRequest.EDGE_MODE_FAST
                }

                pExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, nrMode)
                pExt.setCaptureRequestOption(CaptureRequest.EDGE_MODE, edgeMode)
                cExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, nrMode)
                cExt.setCaptureRequestOption(CaptureRequest.EDGE_MODE, edgeMode)
                cExt.setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_HIGH_QUALITY)
                cExt.setCaptureRequestOption(CaptureRequest.SHADING_MODE, CaptureRequest.SHADING_MODE_HIGH_QUALITY)
                cExt.setCaptureRequestOption(CaptureRequest.TONEMAP_MODE, CaptureRequest.TONEMAP_MODE_HIGH_QUALITY)

                val oisMode = if (state.stabilizationEnabled) {
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                } else {
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                }
                val videoStabMode = if (state.stabilizationEnabled) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_PREVIEW_STABILIZATION
                    } else {
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                    }
                } else {
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF
                }

                pExt.setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, oisMode)
                cExt.setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, oisMode)
                pExt.setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, videoStabMode)

                if (state.captureFormat == CaptureFormat.PRO) {
                    val expNs = 1_000_000_000L / state.shutterSpeedDenom.coerceAtLeast(1)
                    pExt.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    pExt.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, state.isoValue)
                    pExt.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, expNs)
                    cExt.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    cExt.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, state.isoValue)
                    cExt.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, expNs)
                }

                when (state.focusMode) {
                    FocusMode.AUTO, FocusMode.AI_LOCK -> {
                        val afMode = if (state.shootingMode == ShootingMode.VIDEO) {
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                        } else {
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                        }
                        pExt.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, afMode)
                    }
                    FocusMode.MANUAL -> {
                        pExt.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        val diopters = (1.0f - state.manualFocusDistance) * minFocusDistanceDiopters
                        pExt.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, diopters)
                        cExt.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        cExt.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, diopters)
                    }
                }

                applyFlashBuilder(captureBuilder, state.flashMode)

                val preview = previewBuilder.build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                imageCapture = captureBuilder.build()

                val targetQuality = when (state.videoResolution) {
                    VideoResolution.UHD_4K -> Quality.UHD
                    VideoResolution.FHD_1080P -> Quality.FHD
                    VideoResolution.HD_720P -> Quality.HD
                }
                val qualitySelector = QualitySelector.from(
                    targetQuality,
                    FallbackStrategy.lowerQualityOrHigherThan(targetQuality)
                )
                val recorder = Recorder.Builder().setQualitySelector(qualitySelector).build()
                val vBuilder = VideoCapture.Builder(recorder)
                val vExt = Camera2Interop.Extender(vBuilder)
                vExt.setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, oisMode)
                vExt.setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, videoStabMode)
                vExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY)
                videoCapture = vBuilder.build()

                provider.unbindAll()
                bound = if (state.shootingMode == ShootingMode.VIDEO) {
                    provider.bindToLifecycle(lifecycleOwner, physicalSelector, preview, videoCapture)
                } else {
                    provider.bindToLifecycle(lifecycleOwner, physicalSelector, preview, imageCapture)
                }
            }

            camera = bound
            onExtensionUpdate(activeExtLabel, report)

            bound.cameraControl.enableTorch(state.flashMode == FlashMode.TORCH)

            if (state.captureFormat != CaptureFormat.PRO) {
                applyExposureCompensation(state.evCompensation)
            }

            try {
                val chars = Camera2CameraInfo.extractCameraCharacteristics(bound.cameraInfo)
                minFocusDistanceDiopters = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            } catch (_: Throwable) {
            }

            bound.cameraInfo.zoomState.value?.let { zs ->
                currentMaxHwZoomRatio = zs.maxZoomRatio
                onZoomBounds(zs.minZoomRatio, zs.maxZoomRatio)
                val targetRatio = if (activeExtLabel != null && state.forceLogicalHwExtensions) {
                    state.displayZoom.coerceIn(zs.minZoomRatio, zs.maxZoomRatio)
                } else if (state.activeLens == LensType.MAIN_50MP && state.displayZoom <= 1.05f) {
                    1.0f.coerceIn(zs.minZoomRatio, zs.maxZoomRatio)
                } else {
                    state.lensZoomRatio.coerceIn(zs.minZoomRatio, zs.maxZoomRatio)
                }
                bound.cameraControl.setZoomRatio(targetRatio)
            }
        } catch (e: Exception) {
            Log.e("Reno14FreshCam", "Errore avvio fotocamera", e)
            onBanner("Attivazione ${state.activeLens.label}")
        }
    }

    private fun applyFlashBuilder(builder: ImageCapture.Builder, flashMode: FlashMode) {
        when (flashMode) {
            FlashMode.OFF -> builder.setFlashMode(ImageCapture.FLASH_MODE_OFF)
            FlashMode.AUTO -> builder.setFlashMode(ImageCapture.FLASH_MODE_AUTO)
            FlashMode.ON -> builder.setFlashMode(ImageCapture.FLASH_MODE_ON)
            FlashMode.TORCH -> builder.setFlashMode(ImageCapture.FLASH_MODE_OFF)
        }
    }

    fun applyExposureCompensation(ev: Int) {
        val cam = camera ?: return
        val range = cam.cameraInfo.exposureState.exposureCompensationRange
        if (range.lower < range.upper) {
            cam.cameraControl.setExposureCompensationIndex(ev.coerceIn(range.lower, range.upper))
        }
    }

    fun applyFlashLive(flashMode: FlashMode) {
        val cap = imageCapture
        when (flashMode) {
            FlashMode.OFF -> {
                cap?.flashMode = ImageCapture.FLASH_MODE_OFF
                camera?.cameraControl?.enableTorch(false)
            }
            FlashMode.AUTO -> {
                cap?.flashMode = ImageCapture.FLASH_MODE_AUTO
                camera?.cameraControl?.enableTorch(false)
            }
            FlashMode.ON -> {
                cap?.flashMode = ImageCapture.FLASH_MODE_ON
                camera?.cameraControl?.enableTorch(false)
            }
            FlashMode.TORCH -> {
                cap?.flashMode = ImageCapture.FLASH_MODE_OFF
                camera?.cameraControl?.enableTorch(true)
            }
        }
    }

    fun setZoom(ratio: Float) {
        val cam = camera ?: return
        val zs = cam.cameraInfo.zoomState.value ?: return
        currentMaxHwZoomRatio = zs.maxZoomRatio
        cam.cameraControl.setZoomRatio(ratio.coerceIn(zs.minZoomRatio, zs.maxZoomRatio))
    }

    fun applyFocusPoint(x: Float, y: Float, w: Float, h: Float, keepLocked: Boolean) {
        val cam = camera ?: return
        val pt = SurfaceOrientedMeteringPointFactory(w, h).createPoint(x, y)
        val builder = FocusMeteringAction.Builder(pt, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
        if (keepLocked) builder.disableAutoCancel() else builder.setAutoCancelDuration(3, TimeUnit.SECONDS)
        cam.cameraControl.startFocusAndMetering(builder.build())
    }

    fun clearFocusLock() {
        camera?.cameraControl?.cancelFocusAndMetering()
    }

    @SuppressLint("MissingPermission")
    private fun getLastKnownLocation(): Location? {
        val fine = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        if (!fine && !coarse) return null

        val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return null
        val providers = lm.getProviders(true)
        var best: Location? = null
        for (p in providers) {
            val loc = try { lm.getLastKnownLocation(p) } catch (_: Exception) { null }
            if (loc != null && (best == null || loc.accuracy < best.accuracy)) {
                best = loc
            }
        }
        return best
    }

    fun shootInstant(
        snapshot: FreshCamState,
        onStatus: (String) -> Unit,
        onResult: (uriString: String?, message: String) -> Unit
    ) {
        val cap = imageCapture ?: run {
            onResult(null, "Fotocamera non pronta")
            return
        }

        val capturedLocation = if (snapshot.locationTagEnabled) getLastKnownLocation() else null
        val isTele = snapshot.activeLens == LensType.TELE_50MP || snapshot.displayZoom >= 3.45f

        // v15: Burst multi-frame ATTIVO GIA' DA 3.5x sulla Tele, e anche su 1x con AI Media/Alta se non c'è HW Extension
        val needBurst = snapshot.activeHardwareExtension == null &&
            snapshot.aiPreset.isEnabled &&
            snapshot.captureFormat != CaptureFormat.RAW &&
            (isTele || snapshot.aiPreset.level >= 2)

        val burstCount = when {
            !needBurst -> 1
            snapshot.aiPreset.level >= 3 -> 4
            else -> 3
        }
        val maxSide = if (snapshot.aiPreset.isEnabled && snapshot.activeHardwareExtension == null) 3400 else 4400
        val collectedFrames = mutableListOf<Bitmap>()

        val requestedRatio = if (snapshot.activeHardwareExtension != null && snapshot.forceLogicalHwExtensions) {
            snapshot.displayZoom
        } else {
            snapshot.lensZoomRatio
        }
        val digitalExtraZoom = if (requestedRatio > currentMaxHwZoomRatio && snapshot.displayZoom >= 3.45f) {
            (requestedRatio / currentMaxHwZoomRatio).coerceIn(1.0f, 4.0f)
        } else {
            1.0f
        }

        if (snapshot.activeHardwareExtension != null) {
            onStatus("Scatto ${snapshot.activeHardwareExtension}...")
        } else if (snapshot.aiPreset.isEnabled) {
            onStatus(if (burstCount > 1) "Elaborazione ${snapshot.aiPreset.shortBadge} + Burst ${burstCount}x..." else "Elaborazione ${snapshot.aiPreset.shortBadge}...")
        }

        fun captureNextFrame(remaining: Int) {
            cap.takePicture(
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(imageProxy: ImageProxy) {
                        try {
                            val deg = imageProxy.imageInfo.rotationDegrees
                            val rawBmp = proxyToBitmap(imageProxy)
                            val rotated = rotate(rawBmp, deg)
                            if (rotated !== rawBmp) rawBmp.recycle()
                            val capped = capSize(rotated, maxSide)
                            if (capped !== rotated) rotated.recycle()
                            collectedFrames.add(capped)
                        } catch (t: Throwable) {
                            Log.e("Reno14FreshCam", "Errore acquisizione frame", t)
                        } finally {
                            imageProxy.close()
                        }

                        if (remaining > 1 && collectedFrames.isNotEmpty()) {
                            captureNextFrame(remaining - 1)
                        } else {
                            processAndSaveFrames(collectedFrames, snapshot, digitalExtraZoom, capturedLocation, onResult)
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        if (collectedFrames.isNotEmpty()) {
                            processAndSaveFrames(collectedFrames, snapshot, digitalExtraZoom, capturedLocation, onResult)
                        } else {
                            onResult(null, "Errore scatto: ${exception.message}")
                        }
                    }
                }
            )
        }

        captureNextFrame(burstCount)
    }

    private fun processAndSaveFrames(
        frames: List<Bitmap>,
        snapshot: FreshCamState,
        digitalExtraZoom: Float,
        capturedLocation: Location?,
        onResult: (uriString: String?, message: String) -> Unit
    ) {
        if (frames.isEmpty()) {
            onResult(null, "Nessun fotogramma acquisito")
            return
        }

        saveWorker.execute {
            try {
                val outcome = if (snapshot.captureFormat == CaptureFormat.RAW) {
                    NeuralProcessResult(
                        bitmap = snapshot.profile.applyToBitmap(
                            source = frames.first(),
                            strength = 0.0f,
                            brightnessBoost = snapshot.brightnessBoost
                        ),
                        modeApplied = "RAW"
                    )
                } else {
                    neuralEngine.processBurst(
                        frames = frames,
                        profile = snapshot.profile,
                        profileIntensity = snapshot.profileIntensity,
                        brightnessBoost = snapshot.brightnessBoost,
                        aiPreset = snapshot.aiPreset,
                        activeHardwareExtension = snapshot.activeHardwareExtension,
                        activeLens = snapshot.activeLens,
                        displayZoom = snapshot.displayZoom,
                        digitalExtraZoom = digitalExtraZoom,
                        cloudModelPreset = snapshot.cloudModelPreset,
                        cloudEndpointUrl = snapshot.cloudEndpointUrl,
                        cloudApiKey = snapshot.cloudApiKey,
                        postSavePolish = snapshot.postSavePolishEnabled
                    )
                }

                frames.forEach { if (!it.isRecycled && it !== outcome.bitmap) it.recycle() }

                val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.ITALY).format(Date())
                val zoomTag = "${String.format(Locale.US, "%.1f", snapshot.displayZoom)}x"
                val name = "FreshCam_${snapshot.profile.id}_${zoomTag}_$stamp.jpg"

                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Reno14FreshCam")
                    }
                }

                val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    context.contentResolver.openOutputStream(uri)?.use { out ->
                        outcome.bitmap.compress(Bitmap.CompressFormat.JPEG, snapshot.jpegQuality.coerceIn(85, 100), out)
                    }
                    if (!outcome.bitmap.isRecycled) outcome.bitmap.recycle()

                    if (capturedLocation != null) {
                        try {
                            context.contentResolver.openFileDescriptor(uri, "rw")?.use { pfd ->
                                val exif = ExifInterface(pfd.fileDescriptor)
                                exif.setGpsInfo(capturedLocation)
                                exif.saveAttributes()
                            }
                        } catch (e: Exception) {
                            Log.w("Reno14FreshCam", "Impossibile scrivere EXIF GPS", e)
                        }
                    }
                    val gpsNote = if (capturedLocation != null) " · GPS" else ""
                    ContextCompat.getMainExecutor(context).execute {
                        onResult(uri.toString(), "Salvata: ${outcome.modeApplied}$gpsNote")
                    }
                } else {
                    if (!outcome.bitmap.isRecycled) outcome.bitmap.recycle()
                    ContextCompat.getMainExecutor(context).execute {
                        onResult(null, "Errore salvataggio in Galleria")
                    }
                }
            } catch (t: Throwable) {
                Log.e("Reno14FreshCam", "Errore elaborazione foto", t)
                frames.forEach { if (!it.isRecycled) it.recycle() }
                ContextCompat.getMainExecutor(context).execute {
                    onResult(null, "Errore foto: ${t.localizedMessage}")
                }
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun toggleVideo(
        state: FreshCamState,
        onRecording: (Boolean) -> Unit,
        onBanner: (String) -> Unit
    ) {
        val rec = activeRecording
        if (rec != null) {
            rec.stop()
            activeRecording = null
            onRecording(false)
            return
        }

        val vc = videoCapture ?: run {
            onBanner("Video non pronto")
            return
        }

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ITALY).format(Date())
        val name = "FreshCam_VID_${state.videoResolution.name}_$stamp.mp4"
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Reno14FreshCam")
            }
        }

        val outOptBuilder = MediaStoreOutputOptions.Builder(
            context.contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        ).setContentValues(values)

        if (state.locationTagEnabled) {
            getLastKnownLocation()?.let { loc ->
                outOptBuilder.setLocation(loc)
            }
        }

        activeRecording = vc.output.prepareRecording(context, outOptBuilder.build())
            .withAudioEnabled()
            .start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        onRecording(true)
                        onBanner("Video ${state.videoResolution.label} avviato (OIS+EIS)")
                    }
                    is VideoRecordEvent.Finalize -> {
                        onRecording(false)
                        if (event.hasError()) onBanner("Errore video") else onBanner("Video salvato: $name")
                    }
                }
            }
    }

    private fun proxyToBitmap(image: ImageProxy): Bitmap {
        if (image.format == ImageFormat.JPEG) {
            val buf = image.planes[0].buffer
            val arr = ByteArray(buf.remaining())
            buf.get(arr)
            return BitmapFactory.decodeByteArray(arr, 0, arr.size)
        }
        val yBuf = image.planes[0].buffer
        val uBuf = image.planes[1].buffer
        val vBuf = image.planes[2].buffer

        val ySz = yBuf.remaining()
        val uSz = uBuf.remaining()
        val vSz = vBuf.remaining()

        val nv21 = ByteArray(ySz + uSz + vSz)
        yBuf.get(nv21, 0, ySz)
        vBuf.get(nv21, ySz, vSz)
        uBuf.get(nv21, ySz + vSz, uSz)

        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 98, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun rotate(bmp: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bmp
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun capSize(bmp: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bmp.width, bmp.height)
        if (longest <= maxSide) return bmp
        val scale = maxSide.toFloat() / longest.toFloat()
        return Bitmap.createScaledBitmap(bmp, (bmp.width * scale).toInt().coerceAtLeast(1), (bmp.height * scale).toInt().coerceAtLeast(1), true)
    }
}
