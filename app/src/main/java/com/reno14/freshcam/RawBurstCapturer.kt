package com.reno14.freshcam

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

data class RawFrameData(
    val width: Int,
    val height: Int,
    val data: ShortArray
)

data class RawBurstMeta(
    val cfaArrangement: Int,
    val blackLevels: IntArray,
    val whiteLevel: Int,
    val sensorOrientation: Int,
    val wbGains: FloatArray?,
    val ccm: FloatArray?,
    val iso: Int?,
    val exposureTimeNs: Long?,
    val aperture: Float?,
    val focalLength: Float?
)

data class RawBurstPayload(
    val frames: List<RawFrameData>,
    val meta: RawBurstMeta
)

class RawBurstCapturer(private val context: Context) {

    @SuppressLint("MissingPermission")
    fun captureRawBurstBlocking(
        cameraId: String,
        frameCount: Int,
        evCompensation: Int,
        stabilizationEnabled: Boolean
    ): Result<RawBurstPayload> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars = try {
            manager.getCameraCharacteristics(cameraId)
        } catch (e: Throwable) {
            return Result.failure(e)
        }

        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return Result.failure(IllegalStateException("Stream map non disponibile"))
        val rawSize = map.getOutputSizes(ImageFormat.RAW_SENSOR)
            ?.maxByOrNull { it.width.toLong() * it.height.toLong() }
            ?: return Result.failure(IllegalStateException("RAW_SENSOR non esposto su ID$cameraId"))

        val previewSize = map.getOutputSizes(ImageFormat.YUV_420_888)
            ?.filter { it.width in 480..960 && it.height in 360..720 }
            ?.minByOrNull { it.width * it.height }
            ?: Size(640, 480)

        val cfa = chars.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)
            ?: CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB
        val whiteLevel = chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023
        val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val blackPattern = chars.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
        val blackLevels = IntArray(4) { idx ->
            if (blackPattern != null) {
                val col = idx % 2
                val row = idx / 2
                blackPattern.getOffsetForIndex(col, row)
            } else {
                64
            }
        }
        val aeRange = chars.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

        val thread = HandlerThread("FreshCamRawBurst").apply { start() }
        val handler = Handler(thread.looper)

        var cameraDevice: CameraDevice? = null
        var captureSession: CameraCaptureSession? = null
        var previewReader: ImageReader? = null
        var rawReader: ImageReader? = null

        return try {
            previewReader = ImageReader.newInstance(
                previewSize.width,
                previewSize.height,
                ImageFormat.YUV_420_888,
                2
            ).apply {
                setOnImageAvailableListener({ reader ->
                    try {
                        reader.acquireLatestImage()?.close()
                    } catch (_: Throwable) {
                    }
                }, handler)
            }

            val safeCount = frameCount.coerceIn(2, 4)
            val collectedFrames = mutableListOf<RawFrameData>()
            val framesLatch = CountDownLatch(safeCount)
            val collecting = AtomicBoolean(false)

            rawReader = ImageReader.newInstance(
                rawSize.width,
                rawSize.height,
                ImageFormat.RAW_SENSOR,
                safeCount + 1
            ).apply {
                setOnImageAvailableListener({ reader ->
                    var img: Image? = null
                    try {
                        img = reader.acquireNextImage()
                        if (img != null && collecting.get()) {
                            synchronized(collectedFrames) {
                                if (collectedFrames.size < safeCount) {
                                    collectedFrames.add(extractRawFrame(img))
                                    framesLatch.countDown()
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        Log.e("RawBurstCapturer", "Errore lettura frame RAW", t)
                    } finally {
                        try { img?.close() } catch (_: Throwable) {}
                    }
                }, handler)
            }

            val openLatch = CountDownLatch(1)
            var openError: Throwable? = null
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    openLatch.countDown()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    openError = IllegalStateException("Camera disconnessa")
                    try { camera.close() } catch (_: Throwable) {}
                    openLatch.countDown()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    openError = IllegalStateException("Errore apertura Camera2 ($error)")
                    try { camera.close() } catch (_: Throwable) {}
                    openLatch.countDown()
                }
            }, handler)

            if (!openLatch.await(2500, TimeUnit.MILLISECONDS) || cameraDevice == null) {
                return Result.failure(openError ?: IllegalStateException("Timeout apertura Camera2"))
            }
            val cam = cameraDevice!!

            val sessionLatch = CountDownLatch(1)
            var sessionError: Throwable? = null
            val outputs = listOf(
                OutputConfiguration(previewReader.surface),
                OutputConfiguration(rawReader.surface)
            )
            val sessionConfig = SessionConfiguration(
                SessionConfiguration.SESSION_REGULAR,
                outputs,
                { r -> handler.post(r) },
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        sessionLatch.countDown()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        sessionError = IllegalStateException("Configurazione sessione RAW fallita")
                        sessionLatch.countDown()
                    }
                }
            )
            cam.createCaptureSession(sessionConfig)

            if (!sessionLatch.await(2500, TimeUnit.MILLISECONDS) || captureSession == null) {
                return Result.failure(sessionError ?: IllegalStateException("Timeout sessione RAW"))
            }
            val session = captureSession!!

            val aeConvergedLatch = CountDownLatch(1)
            var warmupFrames = 0
            var latestResult: TotalCaptureResult? = null

            val previewReqBuilder = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(previewReader.surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                if (aeRange != null && aeRange.lower < aeRange.upper) {
                    set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, evCompensation.coerceIn(aeRange.lower, aeRange.upper))
                }
                set(
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    if (stabilizationEnabled) CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                    else CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                )
            }

            session.setRepeatingRequest(
                previewReqBuilder.build(),
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        s: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        latestResult = result
                        warmupFrames++
                        val aeState = result.get(CaptureResult.CONTROL_AE_STATE)
                        val converged = aeState == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                            aeState == CaptureResult.CONTROL_AE_STATE_LOCKED ||
                            aeState == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED
                        if ((converged && warmupFrames >= 4) || warmupFrames >= 12) {
                            aeConvergedLatch.countDown()
                        }
                    }
                },
                handler
            )
            aeConvergedLatch.await(600, TimeUnit.MILLISECONDS)

            collecting.set(true)
            val burstResultLatch = CountDownLatch(1)
            var burstResult: TotalCaptureResult? = null

            val burstRequests = List(safeCount) {
                cam.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                    addTarget(rawReader.surface)
                    addTarget(previewReader.surface)
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.CONTROL_AE_LOCK, true)
                    set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AWB_LOCK, true)
                    if (aeRange != null && aeRange.lower < aeRange.upper) {
                        set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, evCompensation.coerceIn(aeRange.lower, aeRange.upper))
                    }
                    set(
                        CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                        if (stabilizationEnabled) CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                        else CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                    )
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE_ON)
                }.build()
            }

            session.captureBurst(
                burstRequests,
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        s: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        if (burstResult == null) {
                            burstResult = result
                            burstResultLatch.countDown()
                        }
                    }
                },
                handler
            )

            framesLatch.await(4000, TimeUnit.MILLISECONDS)
            burstResultLatch.await(500, TimeUnit.MILLISECONDS)

            val framesCopy = synchronized(collectedFrames) { collectedFrames.toList() }
            if (framesCopy.isEmpty()) {
                return Result.failure(IllegalStateException("Nessun frame RAW ricevuto dal sensore"))
            }

            val refRes = burstResult ?: latestResult
            val wbGains = extractWbGains(refRes)
            val ccm = extractCcm(refRes)

            val meta = RawBurstMeta(
                cfaArrangement = cfa,
                blackLevels = blackLevels,
                whiteLevel = whiteLevel,
                sensorOrientation = sensorOrientation,
                wbGains = wbGains,
                ccm = ccm,
                iso = refRes?.get(CaptureResult.SENSOR_SENSITIVITY),
                exposureTimeNs = refRes?.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                aperture = refRes?.get(CaptureResult.LENS_APERTURE),
                focalLength = refRes?.get(CaptureResult.LENS_FOCAL_LENGTH)
            )

            Result.success(RawBurstPayload(framesCopy, meta))
        } catch (t: Throwable) {
            Log.e("RawBurstCapturer", "Errore cattura RAW burst", t)
            Result.failure(t)
        } finally {
            try { captureSession?.stopRepeating() } catch (_: Throwable) {}
            try { captureSession?.close() } catch (_: Throwable) {}
            try { cameraDevice?.close() } catch (_: Throwable) {}
            try { rawReader?.close() } catch (_: Throwable) {}
            try { previewReader?.close() } catch (_: Throwable) {}
            try { thread.quitSafely() } catch (_: Throwable) {}
        }
    }

    private fun extractRawFrame(image: Image): RawFrameData {
        val w = image.width
        val h = image.height
        val plane = image.planes[0]
        val buf = plane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val rowStride = plane.rowStride
        val out = ShortArray(w * h)

        if (rowStride == w * 2) {
            buf.asShortBuffer().get(out)
        } else {
            val rowBytes = ByteArray(rowStride)
            for (y in 0 until h) {
                buf.position(y * rowStride)
                buf.get(rowBytes, 0, rowStride)
                val rowOff = y * w
                for (x in 0 until w) {
                    val b0 = rowBytes[x * 2].toInt() and 0xFF
                    val b1 = rowBytes[x * 2 + 1].toInt() and 0xFF
                    out[rowOff + x] = ((b1 shl 8) or b0).toShort()
                }
            }
        }
        return RawFrameData(w, h, out)
    }

    private fun extractWbGains(result: TotalCaptureResult?): FloatArray? {
        if (result == null) return null
        val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS)
        if (gains != null) {
            val r = gains.red
            val g = (gains.greenEven + gains.greenOdd) * 0.5f
            val b = gains.blue
            if (r.isFinite() && g.isFinite() && b.isFinite() && g > 0.01f && r > 0.01f && b > 0.01f) {
                val nr = r / g
                val nb = b / g
                if (kotlin.math.abs(nr - 1.0f) > 0.04f || kotlin.math.abs(nb - 1.0f) > 0.04f) {
                    return floatArrayOf(nr.coerceIn(0.6f, 4.0f), 1.0f, nb.coerceIn(0.6f, 4.0f))
                }
            }
        }
        val neutral = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
        if (neutral != null && neutral.size == 3) {
            val rn = neutral[0].toFloat()
            val gn = neutral[1].toFloat()
            val bn = neutral[2].toFloat()
            if (rn > 0.01f && gn > 0.01f && bn > 0.01f) {
                return floatArrayOf(
                    (gn / rn).coerceIn(0.6f, 4.0f),
                    1.0f,
                    (gn / bn).coerceIn(0.6f, 4.0f)
                )
            }
        }
        return null
    }

    private fun extractCcm(result: TotalCaptureResult?): FloatArray? {
        if (result == null) return null
        val ccmObj = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM) ?: return null
        val m = FloatArray(9)
        var nonZero = false
        for (row in 0..2) {
            var rowSum = 0f
            for (col in 0..2) {
                val v = ccmObj.getElement(col, row).toFloat()
                if (!v.isFinite()) return null
                m[row * 3 + col] = v
                rowSum += v
                if (kotlin.math.abs(v) > 0.01f) nonZero = true
            }
            if (rowSum < 0.5f || rowSum > 1.5f) return null
        }
        if (!nonZero) return null
        val isIdentity = kotlin.math.abs(m[0] - 1f) < 0.01f &&
            kotlin.math.abs(m[4] - 1f) < 0.01f &&
            kotlin.math.abs(m[8] - 1f) < 0.01f &&
            kotlin.math.abs(m[1]) < 0.01f &&
            kotlin.math.abs(m[2]) < 0.01f
        return if (isIdentity) null else m
    }
}
