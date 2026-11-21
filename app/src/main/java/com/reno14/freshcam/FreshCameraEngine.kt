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
import androidx.camera.lifecycle.ProcessCameraProvider
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
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var camera: Camera? = null

    private val saveWorker = Executors.newSingleThreadExecutor()
    private val neuralEngine = NeuralSuperResEngine(context)

    var minFocusDistanceDiopters: Float = 0f
        private set

    private var discoveredLenses: List<PhysicalLensInfo> = emptyList()

    fun checkNeuralModelReady(): Boolean = neuralEngine.isModelAvailable()
    fun getLoadedModelName(): String = neuralEngine.loadedAssetName

    fun discoverLenses(): List<PhysicalLensInfo> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val list = mutableListOf<PhysicalLensInfo>()
        try {
            val backList = mutableListOf<Pair<String, Float>>()
            for (id in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
                val focal = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 4.5f
                if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    if (list.none { it.lensType == LensType.FRONT }) {
                        list.add(PhysicalLensInfo(id, LensType.FRONT, focal))
                    }
                } else if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    backList.add(id to focal)
                }
            }
            val sortedBack = backList.sortedBy { it.second }
            when {
                sortedBack.size >= 3 -> {
                    list.add(PhysicalLensInfo(sortedBack[0].first, LensType.ULTRAWIDE, sortedBack[0].second))
                    list.add(PhysicalLensInfo(sortedBack[1].first, LensType.MAIN_50MP, sortedBack[1].second))
                    list.add(PhysicalLensInfo(sortedBack.last().first, LensType.TELE_50MP, sortedBack.last().second))
                }
                sortedBack.size == 2 -> {
                    list.add(PhysicalLensInfo(sortedBack[0].first, LensType.MAIN_50MP, sortedBack[0].second))
                    list.add(PhysicalLensInfo(sortedBack[1].first, LensType.TELE_50MP, sortedBack[1].second))
                }
                sortedBack.size == 1 -> {
                    list.add(PhysicalLensInfo(sortedBack[0].first, LensType.MAIN_50MP, sortedBack[0].second))
                }
            }
        } catch (e: Exception) {
            Log.e("Reno14FreshCam", "Errore rilevamento fotocamere", e)
        }
        discoveredLenses = list
        return list
    }

    @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
    private fun selectorForLens(lensType: LensType): CameraSelector {
        val target = discoveredLenses.firstOrNull { it.lensType == lensType }
        if (target != null) {
            return CameraSelector.Builder()
                .addCameraFilter(CameraFilter { infos ->
                    val filtered = infos.filter { Camera2CameraInfo.from(it).cameraId == target.cameraId }
                    filtered.ifEmpty { infos }
                })
                .build()
        }
        return if (lensType == LensType.FRONT) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
    }

    @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
    fun bind(
        previewView: PreviewView,
        state: FreshCamState,
        onZoomBounds: (minZ: Float, maxZ: Float) -> Unit,
        onBanner: (String) -> Unit
    ) {
        if (discoveredLenses.isEmpty()) discoverLenses()

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider

                val previewBuilder = Preview.Builder()
                val resSelector = ResolutionSelector.Builder()
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .build()
                val captureBuilder = ImageCapture.Builder()
                    .setResolutionSelector(resSelector)
                    .setJpegQuality(99)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)

                val pExt = Camera2Interop.Extender(previewBuilder)
                val cExt = Camera2Interop.Extender(captureBuilder)

                // Massimizza qualità ottica e ISP Hardware sul sensore
                val nrMode = if (state.captureFormat == CaptureFormat.RAW) {
                    CaptureRequest.NOISE_REDUCTION_MODE_FAST
                } else {
                    CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY
                }
                val edgeMode = if (state.captureFormat == CaptureFormat.RAW || state.aiMode == AiMode.LOW) {
                    CaptureRequest.EDGE_MODE_FAST
                } else {
                    CaptureRequest.EDGE_MODE_HIGH_QUALITY
                }

                pExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, nrMode)
                pExt.setCaptureRequestOption(CaptureRequest.EDGE_MODE, edgeMode)
                cExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, nrMode)
                cExt.setCaptureRequestOption(CaptureRequest.EDGE_MODE, edgeMode)
                cExt.setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_HIGH_QUALITY)
                cExt.setCaptureRequestOption(CaptureRequest.SHADING_MODE, CaptureRequest.SHADING_MODE_HIGH_QUALITY)

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

                if (state.captureFormat != CaptureFormat.PRO) {
                    pExt.setCaptureRequestOption(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, state.evCompensation)
                    cExt.setCaptureRequestOption(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, state.evCompensation)
                } else {
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

                when (state.flashMode) {
                    FlashMode.OFF -> {
                        captureBuilder.setFlashMode(ImageCapture.FLASH_MODE_OFF)
                        camera?.cameraControl?.enableTorch(false)
                    }
                    FlashMode.AUTO -> {
                        captureBuilder.setFlashMode(ImageCapture.FLASH_MODE_AUTO)
                        camera?.cameraControl?.enableTorch(false)
                    }
                    FlashMode.ON -> {
                        captureBuilder.setFlashMode(ImageCapture.FLASH_MODE_ON)
                        camera?.cameraControl?.enableTorch(false)
                    }
                    FlashMode.TORCH -> {
                        captureBuilder.setFlashMode(ImageCapture.FLASH_MODE_OFF)
                        camera?.cameraControl?.enableTorch(true)
                    }
                }

                val preview = previewBuilder.build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                imageCapture = captureBuilder.build()

                val recorder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.HIGHEST)).build()
                val vBuilder = VideoCapture.Builder(recorder)
                val vExt = Camera2Interop.Extender(vBuilder)
                vExt.setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, oisMode)
                vExt.setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, videoStabMode)
                vExt.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY)
                videoCapture = vBuilder.build()

                provider.unbindAll()
                val bound = if (state.shootingMode == ShootingMode.VIDEO) {
                    provider.bindToLifecycle(lifecycleOwner, selectorForLens(state.activeLens), preview, videoCapture)
                } else {
                    provider.bindToLifecycle(lifecycleOwner, selectorForLens(state.activeLens), preview, imageCapture)
                }
                camera = bound

                val chars = Camera2CameraInfo.extractCameraCharacteristics(bound.cameraInfo)
                minFocusDistanceDiopters = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f

                bound.cameraInfo.zoomState.value?.let { zs ->
                    onZoomBounds(zs.minZoomRatio, zs.maxZoomRatio)
                    bound.cameraControl.setZoomRatio(state.lensZoomRatio.coerceIn(zs.minZoomRatio, zs.maxZoomRatio))
                }
            } catch (e: Exception) {
                Log.e("Reno14FreshCam", "Errore avvio fotocamera", e)
                onBanner("Attivazione ${state.activeLens.label}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun setZoom(ratio: Float) {
        val zs = camera?.cameraInfo?.zoomState?.value ?: return
        camera?.cameraControl?.setZoomRatio(ratio.coerceIn(zs.minZoomRatio, zs.maxZoomRatio))
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
        onResult: (uriString: String?, message: String) -> Unit
    ) {
        val cap = imageCapture ?: run {
            onResult(null, "Fotocamera non pronta")
            return
        }

        val capturedLocation = if (snapshot.locationTagEnabled) getLastKnownLocation() else null

        cap.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bmp: Bitmap
                    val deg: Int
                    try {
                        deg = imageProxy.imageInfo.rotationDegrees
                        bmp = proxyToBitmap(imageProxy)
                    } catch (e: Exception) {
                        imageProxy.close()
                        onResult(null, "Errore acquisizione")
                        return
                    } finally {
                        imageProxy.close()
                    }

                    saveWorker.execute {
                        try {
                            val rotated = rotate(bmp, deg)
                            val capped = capSize(rotated, 4800)

                            val outcome = if (snapshot.captureFormat == CaptureFormat.RAW) {
                                NeuralProcessResult(
                                    bitmap = snapshot.profile.applyToBitmap(
                                        source = capped,
                                        strength = 0.0f,
                                        brightnessBoost = snapshot.brightnessBoost
                                    ),
                                    modeApplied = "RAW"
                                )
                            } else {
                                neuralEngine.process(
                                    source = capped,
                                    profile = snapshot.profile,
                                    profileIntensity = snapshot.profileIntensity,
                                    brightnessBoost = snapshot.brightnessBoost,
                                    aiMode = snapshot.aiMode,
                                    isTeleLens = snapshot.activeLens == LensType.TELE_50MP,
                                    lensZoomRatio = snapshot.lensZoomRatio,
                                    superResEnabled = snapshot.neuralSuperResEnabled,
                                    postSavePolish = snapshot.postSavePolishEnabled
                                )
                            }

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
                                    outcome.bitmap.compress(Bitmap.CompressFormat.JPEG, 98, out)
                                }
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
                                    onResult(uri.toString(), "Salvata (${outcome.modeApplied}$gpsNote)")
                                }
                            } else {
                                ContextCompat.getMainExecutor(context).execute {
                                    onResult(null, "Errore salvataggio in Galleria")
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("Reno14FreshCam", "Errore elaborazione foto", e)
                            ContextCompat.getMainExecutor(context).execute {
                                onResult(null, "Errore foto: ${e.localizedMessage}")
                            }
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    onResult(null, "Errore scatto: ${exception.message}")
                }
            }
        )
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
        val name = "FreshCam_VID_${state.activeLens.name.lowercase()}_$stamp.mp4"
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
                        onBanner(if (state.stabilizationEnabled) "Video avviato (OIS+EIS)" else "Video avviato")
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
