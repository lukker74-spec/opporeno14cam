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
 * Motore di AI Generativa / Super-Resolution REALE 100% Locale (Offline) via ONNX Runtime:
 * - Carica i veri pesi neurali .onnx da assets/models/ (ClearReality SPAN 4x o Real-ESRGAN v3 4x).
 * - Prova ad accelerare le operazioni su Dimensity NPU/DSP tramite Android NNAPI (FP16),
 *   con fallback automatico e trasparente a CPU ARM multi-thread se un operatore non è supportato.
 * - Elabora l'immagine a tasselli sovrapposti senza giunture visibili e applica miscela intelligente
 *   (più forte sulle ombre rumorose e sugli zoom spinti) restituendo tempo reale e backend usato.
 */
class OnnxGenerativeEngine(private val context: Context) {

    private var activeSession: OrtSession? = null
    private var loadedModel: LocalNeuralModel? = null
    private var loadedPreferNnapi: Boolean? = null
    private var activeBackendName: String = "Non inizializzato"
    private var inputName: String = "input"
    private var isNhwcInput: Boolean = false

    @Synchronized
    fun probeRuntimeStatus(model: LocalNeuralModel, preferNnapi: Boolean): String {
        return try {
            ensureSession(model, preferNnapi)
            "Pronto · ${model.shortName} ($activeBackendName)"
        } catch (t: Throwable) {
            "Errore modello ${model.shortName}: ${t.localizedMessage}"
        }
    }

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

        // 1. Prova prima con NNAPI (delega su Dimensity NPU/DSP quando supportato dal driver)
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
                Log.i("OnnxGenAI", "Sessione creata con NNAPI/NPU per ${model.shortName}")
                return sess
            } catch (t: Throwable) {
                Log.w("OnnxGenAI", "NNAPI non compatibile per ${model.shortName}, uso CPU ARM: ${t.message}")
            }
        }

        // 2. Fallback solido: CPU ARM 64-bit multi-thread (funziona sempre al 100%)
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
        Log.i("OnnxGenAI", "Sessione CPU creata per ${model.shortName}")
        return sess
    }

    private fun inspectSessionIO(session: OrtSession) {
        inputName = session.inputNames.firstOrNull() ?: "input"
        val info = session.inputInfo[inputName]?.info as? TensorInfo
        val shape = info?.shape
        // Standard è [1, 3, H, W] (NCHW); se shape[3] == 3 allora è NHWC
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
            throw IllegalStateException("File modello $fileName corrotto o mancante negli assets")
        }
        return outFile
    }

    /**
     * Esegue la vera rete neurale 4x in locale sul fotogramma.
     * - Su zoom spinti (>= 7x): prende la porzione ottica reale e la ricostruisce 4x con la GAN.
     * - Su 1x / 3.5x: esegue il restauro neurale 4x ad alta risoluzione e lo fonde in modo
     *   consapevole delle ombre (massima pulizia sulle zone scure, nitidezza naturale sulle luci).
     */
    fun enhancePhoto(
        source: Bitmap,
        model: LocalNeuralModel,
        level: Int,
        displayZoom: Float,
        preferNnapi: Boolean
    ): LocalAiOutcome {
        val t0 = SystemClock.elapsedRealtime()
        val isExtremeZoom = displayZoom >= 6.5f

        val session = try {
            ensureSession(model, preferNnapi)
        } catch (t: Throwable) {
            return LocalAiOutcome(
                bitmap = source,
                label = "AI Locale ERRORE",
                backendUsed = "Errore",
                elapsedMs = SystemClock.elapsedRealtime() - t0,
                error = t.localizedMessage ?: "Impossibile caricare il modello ONNX"
            )
        }

        // Dimensione di ingresso calibrata per qualità visibile + tempo sostenibile in background
        val targetMaxInSide = if (isExtremeZoom) {
            when (level) {
                1 -> 384
                2 -> 480
                else -> 576
            }
        } else {
            when (level) {
                1 -> 512
                2 -> 640
                else -> 768
            }
        }

        val inputBmp = scaleToMaxSide(source, targetMaxInSide)
        val srResult = try {
            runTiled4xInference(session, inputBmp)
        } catch (t: Throwable) {
            // Se NNAPI fallisce a runtime su un tile, riprova una volta su CPU pura
            if (activeBackendName.contains("NNAPI")) {
                Log.w("OnnxGenAI", "Errore runtime NNAPI (${t.message}), riprovo su CPU...")
                try {
                    val cpuSession = ensureSession(model, preferNnapi = false)
                    runTiled4xInference(cpuSession, inputBmp)
                } catch (cpuErr: Throwable) {
                    if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()
                    return LocalAiOutcome(
                        bitmap = source,
                        label = "AI Locale ERRORE",
                        backendUsed = activeBackendName,
                        elapsedMs = SystemClock.elapsedRealtime() - t0,
                        error = cpuErr.localizedMessage ?: "Errore inferenza ONNX"
                    )
                }
            } else {
                if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()
                return LocalAiOutcome(
                    bitmap = source,
                    label = "AI Locale ERRORE",
                    backendUsed = activeBackendName,
                    elapsedMs = SystemClock.elapsedRealtime() - t0,
                    error = t.localizedMessage ?: "Errore inferenza ONNX"
                )
            }
        }

        if (inputBmp !== source && !inputBmp.isRecycled) inputBmp.recycle()

        // Adatta l'output neurale alla risoluzione target e applica fusione guidata da livello e ombre
        val outW = if (isExtremeZoom) max(source.width, srResult.width).coerceAtMost(4096) else source.width
        val outH = if (isExtremeZoom) max(source.height, srResult.height).coerceAtMost(3072) else source.height

        val scaledSr = if (srResult.width != outW || srResult.height != outH) {
            val s = Bitmap.createScaledBitmap(srResult, outW, outH, true)
            srResult.recycle()
            s
        } else {
            srResult
        }

        val finalBmp = if (isExtremeZoom && level >= 3) {
            // Su zoom spinto + AI Alta: 100% ricostruzione neurale pura
            scaledSr
        } else {
            val baseScaled = if (source.width != outW || source.height != outH) {
                Bitmap.createScaledBitmap(source, outW, outH, true)
            } else {
                source
            }
            val baseStrength = if (isExtremeZoom) {
                when (level) {
                    1 -> 0.60f
                    2 -> 0.85f
                    else -> 1.0f
                }
            } else {
                when (level) {
                    1 -> 0.42f
                    2 -> 0.68f
                    else -> 0.90f
                }
            }
            val blended = shadowWeightedBlend(baseScaled, scaledSr, baseStrength, boostShadows = !isExtremeZoom)
            if (baseScaled !== source && !baseScaled.isRecycled) baseScaled.recycle()
            if (!scaledSr.isRecycled) scaledSr.recycle()
            blended
        }

        val elapsed = SystemClock.elapsedRealtime() - t0
        val secStr = String.format(java.util.Locale.US, "%.1fs", elapsed / 1000.0)
        return LocalAiOutcome(
            bitmap = finalBmp,
            label = "${model.shortName} AI ($activeBackendName · $secStr)",
            backendUsed = activeBackendName,
            elapsedMs = elapsed
        )
    }

    /**
     * Inferenza 4x a tasselli sovrapposti senza giunture:
     * Ogni tassello ha un bordo di contesto di 12px che viene scartato dopo la rete neurale,
     * copiando solo il nucleo interno privo di artefatti di bordo.
     */
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

        val coreStep = 128
        val pad = 12

        var y0 = 0
        while (y0 < inH) {
            val y1 = min(y0 + coreStep, inH)
            val cropY0 = max(0, y0 - pad)
            val cropY1 = min(inH, y1 + pad)
            val rawTileH = cropY1 - cropY0
            // Allinea dimensioni del tile a multiplo di 8 (minimo 32) per piena compatibilità ONNX/NNAPI
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

                        // Auto-rilevamento range [0..1] vs [0..255]
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

    /**
     * Miscela intelligente tra foto ottica/RAW e ricostruzione neurale:
     * Sugli oggetti scuri (dove nasce la grana) aumenta il peso della rete neurale per
     * pulire le ombre preservando naturalezza sulle zone molto illuminate.
     */
    private fun shadowWeightedBlend(
        original: Bitmap,
        neural: Bitmap,
        baseAlpha: Float,
        boostShadows: Boolean
    ): Bitmap {
        val w = original.width
        val h = original.height
        val total = w * h
        val origPx = IntArray(total)
        val neurPx = IntArray(total)
        original.getPixels(origPx, 0, w, 0, 0, w, h)
        neural.getPixels(neurPx, 0, w, 0, 0, w, h)

        val shadowAlpha = (baseAlpha + 0.22f).coerceIn(baseAlpha, 0.96f)
        for (i in 0 until total) {
            val o = origPx[i]
            val n = neurPx[i]
            val or = (o ushr 16) and 0xFF
            val og = (o ushr 8) and 0xFF
            val ob = o and 0xFF

            val nr = (n ushr 16) and 0xFF
            val ng = (n ushr 8) and 0xFF
            val nb = n and 0xFF

            val alpha = if (boostShadows) {
                val y = (77 * or + 150 * og + 29 * ob) shr 8
                if (y < 95) {
                    val t = (95 - y) / 95f
                    baseAlpha + (shadowAlpha - baseAlpha) * t
                } else {
                    baseAlpha
                }
            } else {
                baseAlpha
            }

            val inv = 1f - alpha
            val r = (or * inv + nr * alpha + 0.5f).toInt().coerceIn(0, 255)
            val g = (og * inv + ng * alpha + 0.5f).toInt().coerceIn(0, 255)
            val b = (ob * inv + nb * alpha + 0.5f).toInt().coerceIn(0, 255)
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
