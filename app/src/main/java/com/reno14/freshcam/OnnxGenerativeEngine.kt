package com.reno14.freshcam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import java.util.EnumSet
import kotlin.math.max
import kotlin.math.min

data class LocalAiOutcome(
    val bitmap: Bitmap,
    val label: String,
    val backendUsed: String,
    val elapsedMs: Long,
    val error: String? = null
)

/**
 * Motore Neurale ONNX v19 (Lazy-Loaded SOLO per Zoom Spinti >= 7x):
 * - Non viene MAI caricato all'avvio dell'app (zero impatto su memoria e velocità).
 * - Lavora solo sul ritaglio ottico reale a bassa risoluzione (zoom 7x-120x), dove una rete
 *   di Super-Resolution 4x (SPAN ClearReality) ricostruisce davvero dettagli e bordi in 1-3s!
 * - Non viene più usato su 1x/3.5x (dove il downscale distruggeva i 12MP reali).
 */
class OnnxGenerativeEngine(private val context: Context) {

    private var activeSession: OrtSession? = null
    private var loadedModel: LocalNeuralModel? = null
    private var loadedPreferNnapi: Boolean? = null
    private var activeBackendName: String = "Standby"
    private var inputName: String = "input"
    private var isNhwcInput: Boolean = false

    @Synchronized
    private fun ensureSession(model: LocalNeuralModel, preferNnapi: Boolean): OrtSession {
        val current = activeSession
        if (current != null && loadedModel == model && loadedPreferNnapi == preferNnapi) {
            return current
        }

        try { current?.close() } catch (_: Throwable) {}
        activeSession = null

        val modelFile = materializeAssetToFilesDir(model.assetPath)
        val env = OrtEnvironment.getEnvironment()

        if (preferNnapi) {
            try {
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(4)
                    addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                }
                val sess = env.createSession(modelFile.absolutePath, opts)
                inspectSessionIO(sess)
                activeSession = sess
                loadedModel = model
                loadedPreferNnapi = true
                activeBackendName = "NNAPI/NPU"
                return sess
            } catch (t: Throwable) {
                Log.w("OnnxGenAI", "NNAPI fallback a CPU: ${t.message}")
            }
        }

        val cpuOpts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
        }
        val sess = env.createSession(modelFile.absolutePath, cpuOpts)
        inspectSessionIO(sess)
        activeSession = sess
        loadedModel = model
        loadedPreferNnapi = preferNnapi
        activeBackendName = "CPU 4-Thread"
        return sess
    }

    private fun inspectSessionIO(session: OrtSession) {
        inputName = session.inputNames.firstOrNull() ?: "input"
        val info = session.inputInfo[inputName]?.info as? TensorInfo
        val shape = info?.shape
        isNhwcInput = shape != null && shape.size == 4 && shape[3] == 3L && shape[1] != 3L
    }

    private fun materializeAssetToFilesDir(assetPath: String): File {
        val dir = File(context.filesDir, "onnx_models")
        if (!dir.exists()) dir.mkdirs()
        val fileName = assetPath.substringAfterLast('/')
        val outFile = File(dir, fileName)
        if (outFile.exists() && outFile.length() > 500_000L) {
            return outFile
        }
        context.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output, bufferSize = 65536)
            }
        }
        if (outFile.length() < 100_000L) {
            throw IllegalStateException("File modello $fileName mancante negli assets")
        }
        return outFile
    }

    /**
     * Ricostruzione Neurale 4x ottimizzata per Zoom Spinti (>= 7x).
     * Usa solo 4-9 tasselli sul ritaglio ottico reale per massima velocità (1-3s) e alta resa.
     */
    fun enhanceZoomedCrop(
        source: Bitmap,
        model: LocalNeuralModel,
        level: Int,
        preferNnapi: Boolean
    ): LocalAiOutcome {
        val t0 = SystemClock.elapsedRealtime()

        val session = try {
            ensureSession(model, preferNnapi)
        } catch (t: Throwable) {
            return LocalAiOutcome(
                bitmap = source,
                label = "SPAN ERR",
                backendUsed = "Errore",
                elapsedMs = SystemClock.elapsedRealtime() - t0,
                error = t.localizedMessage ?: "Impossibile caricare ONNX"
            )
        }

        // Ingresso compatto e rapido (es. 320px * 4 = 1280px reali ricostruiti -> solo 4-6 tile!)
        val maxInSide = when (level) {
            1 -> 288
            2 -> 352
            else -> 416
        }

        val inputBmp = scaleToMaxSide(source, maxInSide)
        val srResult = try {
            runTiled4xInference(session, inputBmp)
        } catch (t: Throwable) {
            if (activeBackendName.contains("NNAPI")) {
                try {
                    val cpuSession = ensureSession(model, preferNnapi = false)
                    runTiled4xInference(cpuSession, inputBmp)
                } catch (cpuErr: Throwable) {
                    if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()
                    return LocalAiOutcome(
                        bitmap = source,
                        label = "SPAN ERR",
                        backendUsed = activeBackendName,
                        elapsedMs = SystemClock.elapsedRealtime() - t0,
                        error = cpuErr.localizedMessage ?: "Errore inferenza ONNX"
                    )
                }
            } else {
                if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()
                return LocalAiOutcome(
                    bitmap = source,
                    label = "SPAN ERR",
                    backendUsed = activeBackendName,
                    elapsedMs = SystemClock.elapsedRealtime() - t0,
                    error = t.localizedMessage ?: "Errore inferenza ONNX"
                )
            }
        }

        if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()

        val outW = max(source.width, srResult.width).coerceAtMost(3200)
        val outH = max(source.height, srResult.height).coerceAtMost(2400)
        val scaledSr = if (srResult.width != outW || srResult.height != outH) {
            val s = Bitmap.createScaledBitmap(srResult, outW, outH, true)
            srResult.recycle()
            s
        } else {
            srResult
        }

        val finalBmp = if (level >= 3) {
            scaledSr
        } else {
            val baseScaled = if (source.width != outW || source.height != outH) {
                Bitmap.createScaledBitmap(source, outW, outH, true)
            } else {
                source
            }
            val alpha = if (level == 1) 0.65f else 0.85f
            val blended = blendBitmaps(baseScaled, scaledSr, alpha)
            if (baseScaled !== source && !baseScaled.isRecycled) baseScaled.recycle()
            if (!scaledSr.isRecycled) scaledSr.recycle()
            blended
        }

        val elapsed = SystemClock.elapsedRealtime() - t0
        val secStr = String.format(java.util.Locale.US, "%.1fs", elapsed / 1000.0)
        return LocalAiOutcome(
            bitmap = finalBmp,
            label = "Zoom SPAN ($activeBackendName · $secStr)",
            backendUsed = activeBackendName,
            elapsedMs = elapsed
        )
    }

    private fun runTiled4xInference(session: OrtSession, inputBmp: Bitmap): Bitmap {
        val env = OrtEnvironment.getEnvironment()
        val inW = inputBmp.width
        val inH = inputBmp.height
        val srcPixels = IntArray(inW * inH)
        inputBmp.getPixels(srcPixels, 0, inW, 0, 0, inW, inH)

        val scale = 4
        val outW = inW * scale
        val outH = inH * scale
        val outPixels = IntArray(outW * outH)

        val coreStep = 144
        val pad = 10

        var y0 = 0
        while (y0 < inH) {
            val y1 = min(y0 + coreStep, inH)
            val cropY0 = max(0, y0 - pad)
            val cropY1 = min(inH, y1 + pad)
            val rawTileH = cropY1 - cropY0
            val tileH = max(32, ((rawTileH + 7) / 8) * 8)

            var x0 = 0
            while (x0 < inW) {
                val x1 = min(x0 + coreStep, inW)
                val cropX0 = max(0, x0 - pad)
                val cropX1 = min(inW, x1 + pad)
                val rawTileW = cropX1 - cropX0
                val tileW = max(32, ((rawTileW + 7) / 8) * 8)

                val planeSize = tileW * tileH
                val inFloats = FloatArray(3 * planeSize)

                if (!isNhwcInput) {
                    val gOff = planeSize
                    val bOff = planeSize * 2
                    for (ty in 0 until tileH) {
                        val sy = cropY0 + min(ty, rawTileH - 1)
                        val srcRow = sy * inW
                        val dstRow = ty * tileW
                        for (tx in 0 until tileW) {
                            val sx = cropX0 + min(tx, rawTileW - 1)
                            val c = srcPixels[srcRow + sx]
                            val p = dstRow + tx
                            inFloats[p] = ((c ushr 16) and 0xFF) * (1f / 255f)
                            inFloats[gOff + p] = ((c ushr 8) and 0xFF) * (1f / 255f)
                            inFloats[bOff + p] = (c and 0xFF) * (1f / 255f)
                        }
                    }
                } else {
                    for (ty in 0 until tileH) {
                        val sy = cropY0 + min(ty, rawTileH - 1)
                        val srcRow = sy * inW
                        val dstRow = ty * tileW * 3
                        for (tx in 0 until tileW) {
                            val sx = cropX0 + min(tx, rawTileW - 1)
                            val c = srcPixels[srcRow + sx]
                            val p = dstRow + tx * 3
                            inFloats[p] = ((c ushr 16) and 0xFF) * (1f / 255f)
                            inFloats[p + 1] = ((c ushr 8) and 0xFF) * (1f / 255f)
                            inFloats[p + 2] = (c and 0xFF) * (1f / 255f)
                        }
                    }
                }

                val shape = if (!isNhwcInput) {
                    longArrayOf(1L, 3L, tileH.toLong(), tileW.toLong())
                } else {
                    longArrayOf(1L, tileH.toLong(), tileW.toLong(), 3L)
                }

                OnnxTensor.createTensor(env, FloatBuffer.wrap(inFloats), shape).use { inputTensor ->
                    session.run(Collections.singletonMap(inputName, inputTensor)).use { result ->
                        val outTensor = result[0] as OnnxTensor
                        val outShape = outTensor.info.shape
                        val fb = outTensor.floatBuffer
                        fb.rewind()
                        val outFloats = FloatArray(fb.remaining())
                        fb.get(outFloats)

                        val outIsNhwc = outShape.size == 4 && outShape[3] == 3L && outShape[1] != 3L
                        val outTileH = if (outIsNhwc) outShape[1].toInt() else outShape[2].toInt()
                        val outTileW = if (outIsNhwc) outShape[2].toInt() else outShape[3].toInt()
                        val outPlane = outTileW * outTileH

                        var peak = 0f
                        var sIdx = 0
                        while (sIdx < outFloats.size) {
                            val v = outFloats[sIdx]
                            if (v > peak) peak = v
                            sIdx += 31
                        }
                        val mult = if (peak > 2.5f) 1f else 255f

                        val validStartX = (x0 - cropX0) * scale
                        val validStartY = (y0 - cropY0) * scale
                        val validW = (x1 - x0) * scale
                        val validH = (y1 - y0) * scale
                        val dstStartX = x0 * scale
                        val dstStartY = y0 * scale

                        for (vy in 0 until validH) {
                            val oy = validStartY + vy
                            if (oy >= outTileH) break
                            val dstRow = (dstStartY + vy) * outW + dstStartX
                            if (!outIsNhwc) {
                                val tileRow = oy * outTileW + validStartX
                                for (vx in 0 until validW) {
                                    val tp = tileRow + vx
                                    val r = (outFloats[tp] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    val g = (outFloats[outPlane + tp] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    val b = (outFloats[outPlane * 2 + tp] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    outPixels[dstRow + vx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                                }
                            } else {
                                val tileRow = (oy * outTileW + validStartX) * 3
                                for (vx in 0 until validW) {
                                    val tp = tileRow + vx * 3
                                    val r = (outFloats[tp] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    val g = (outFloats[tp + 1] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    val b = (outFloats[tp + 2] * mult + 0.5f).toInt().coerceIn(0, 255)
                                    outPixels[dstRow + vx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                                }
                            }
                        }
                    }
                }

                x0 += coreStep
            }
            y0 += coreStep
        }

        val outBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(outPixels, 0, outW, 0, 0, outW, outH)
        return outBmp
    }

    private fun blendBitmaps(original: Bitmap, neural: Bitmap, alpha: Float): Bitmap {
        val w = original.width
        val h = original.height
        val total = w * h
        val origPx = IntArray(total)
        val neurPx = IntArray(total)
        original.getPixels(origPx, 0, w, 0, 0, w, h)
        neural.getPixels(neurPx, 0, w, 0, 0, w, h)
        val inv = 1f - alpha

        for (i in 0 until total) {
            val o = origPx[i]
            val n = neurPx[i]
            val r = (((o ushr 16) and 0xFF) * inv + ((n ushr 16) and 0xFF) * alpha + 0.5f).toInt().coerceIn(0, 255)
            val g = (((o ushr 8) and 0xFF) * inv + ((n ushr 8) and 0xFF) * alpha + 0.5f).toInt().coerceIn(0, 255)
            val b = ((o and 0xFF) * inv + (n and 0xFF) * alpha + 0.5f).toInt().coerceIn(0, 255)
            origPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(origPx, 0, w, 0, 0, w, h)
        return out
    }

    private fun scaleToMaxSide(bmp: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bmp.width, bmp.height)
        if (longest <= maxSide) return bmp
        val scale = maxSide.toFloat() / longest.toFloat()
        return Bitmap.createScaledBitmap(
            bmp,
            (bmp.width * scale).toInt().coerceAtLeast(32),
            (bmp.height * scale).toInt().coerceAtLeast(32),
            true
        )
    }
}
