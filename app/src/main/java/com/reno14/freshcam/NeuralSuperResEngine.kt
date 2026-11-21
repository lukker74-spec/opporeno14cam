package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Base64
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class NeuralProcessResult(
    val bitmap: Bitmap,
    val modeApplied: String
)

/**
 * Motore AI v8.0 Ottimizzato per MediaTek Dimensity (NNAPI/APU) + Cloud AI Reale:
 *
 * 1) ACCELERAZIONE MEDIATEK (NnApiDelegate):
 *    Attiva l'accelerazione hardware Android NNAPI per instradare il modello TFLite (QuickSRNetSmall / XLSR)
 *    sull'unità neurale MediaTek APU/GPU invece che sulla sola CPU, con fallback automatico su CPU 4 thread.
 *
 * 2) DOPPIA CALIBRAZIONE:
 *    - Foto normali (1x / 3.5x ottico): intervento leggero e naturale (pulizia rumore + recupero micro-contrasto
 *      e nitidezza bilanciata senza artefatti), proprio come la modalità standard potenziata.
 *    - Zoom spinto (8x – 60x): pre-pulizia forte della grana + iniezione neurale concentrata sulla zona
 *      centrale (dove cade il soggetto ingrandito) + dettaglio strutturale anti-impastamento.
 *
 * 3) CLOUD AI CON DIAGNOSTICA REALE (Hugging Face Router / Endpoint compatibile):
 *    Usa il nuovo endpoint `router.huggingface.co/hf-inference/models/...`, gestisce sia risposte immagine
 *    binarie sia risposte JSON base64, e restituisce un messaggio chiaro (es. "Cloud OK", "Cloud: Token mancante",
 *    "Cloud: HTTP 401/503") così sai sempre se e come sta lavorando.
 */
class NeuralSuperResEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var initAttempted: Boolean = false
    var loadedAssetName: String = "Nessuno"
        private set
    var acceleratorName: String = "CPU"
        private set

    @Synchronized
    fun isModelAvailable(): Boolean {
        return getInterpreter() != null
    }

    @Synchronized
    private fun getInterpreter(): Interpreter? {
        if (initAttempted) return interpreter
        initAttempted = true
        val candidates = listOf("superres.tflite", "esrgan.tflite")
        for (assetName in candidates) {
            try {
                val afd = context.assets.openFd(assetName)
                if (afd.declaredLength < 1024L) continue
                val inputStream = FileInputStream(afd.fileDescriptor)
                val fileChannel = inputStream.channel
                val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)

                // Prova prima con MediaTek NNAPI (APU/GPU), poi fallback su CPU
                var tflite: Interpreter? = null
                try {
                    val delegate = NnApiDelegate()
                    val nnOpts = Interpreter.Options().apply {
                        addDelegate(delegate)
                        setNumThreads(4)
                    }
                    tflite = Interpreter(mappedBuffer, nnOpts)
                    nnApiDelegate = delegate
                    acceleratorName = "MediaTek NNAPI"
                } catch (_: Throwable) {
                    nnApiDelegate?.close()
                    nnApiDelegate = null
                    val cpuOpts = Interpreter.Options().apply {
                        setNumThreads(4)
                    }
                    tflite = Interpreter(mappedBuffer, cpuOpts)
                    acceleratorName = "CPU 4T"
                }

                val inShape = tflite!!.getInputTensor(0).shape()
                val outShape = tflite.getOutputTensor(0).shape()
                if (inShape.size != 4 || outShape.size != 4) {
                    tflite.close()
                    continue
                }
                interpreter = tflite
                loadedAssetName = assetName
                Log.i("NeuralSR", "Modello $assetName caricato su $acceleratorName")
                return tflite
            } catch (e: Throwable) {
                Log.w("NeuralSR", "Asset $assetName non caricabile: ${e.message}")
            }
        }
        return null
    }

    fun processBurst(
        frames: List<Bitmap>,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiMode: AiMode,
        isTeleLens: Boolean,
        displayZoom: Float,
        lensZoomRatio: Float,
        superResEnabled: Boolean,
        cloudAiEnabled: Boolean,
        cloudEndpointUrl: String,
        cloudApiKey: String,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        val primaryFrame = frames.first()
        var working = profile.applyToBitmap(
            source = primaryFrame,
            strength = profileIntensity,
            brightnessBoost = brightnessBoost
        )

        if (!superResEnabled) {
            if (postSavePolish) {
                val toned = applyToneBalance(working, strength = 0.45f)
                if (toned !== working) working.recycle()
                working = toned
            }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = "AI OFF"
            )
        }

        val isZoomSpinto = displayZoom >= 8.0f || lensZoomRatio >= 2.2f
        val isZoomEstremo = displayZoom >= 25.0f || lensZoomRatio >= 7.0f

        // 1. Fusione Burst allineata (su zoom spinto o AI Medio/Alto) per pulire rumore reale
        var burstUsed = false
        if (frames.size > 1) {
            try {
                val gradedExtras = frames.drop(1).map {
                    profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                }
                val merged = alignAndMergeBurst(working, gradedExtras)
                gradedExtras.forEach { if (!it.isRecycled) it.recycle() }
                if (merged !== working) {
                    working.recycle()
                    working = merged
                }
                burstUsed = true
            } catch (t: Throwable) {
                Log.e("NeuralSR", "Burst fallback", t)
            }
        }

        // 2. Denoise bilaterale mirato (leggero a 1x, più deciso su zoom spinto)
        val denoised = try {
            filterChromaAndNoise(
                bitmap = working,
                chromaPasses = if (isZoomSpinto) 2 else 1,
                lumaPasses = if (isZoomEstremo) 1 else 0
            )
        } catch (_: Throwable) {
            working
        }
        if (denoised !== working && !working.isRecycled) working.recycle()
        working = denoised

        // 3. Rete Neurale TFLite su MediaTek NNAPI:
        //    - A 1x: intervento leggero ma ben percepibile su tutta la scena (come l'ottimizzazione AI stock)
        //    - Su zoom spinto (8x-60x): inferenza densa sulla regione del soggetto + ricostruzione bordi
        val tflite = getInterpreter()
        var neuralUsed = false
        if (tflite != null) {
            try {
                val neuralOut = runTargetedNeuralEnhancement(
                    tflite = tflite,
                    source = working,
                    aiMode = aiMode,
                    isTeleLens = isTeleLens,
                    isZoomSpinto = isZoomSpinto,
                    isZoomEstremo = isZoomEstremo
                )
                if (neuralOut !== working) {
                    working.recycle()
                    working = neuralOut
                }
                neuralUsed = true
            } catch (t: Throwable) {
                Log.e("NeuralSR", "Errore TFLite: ${t.message}", t)
            }
        }

        // 4. Cloud AI con feedback esplicito
        var cloudStatus = ""
        if (cloudAiEnabled) {
            if (cloudApiKey.isBlank()) {
                cloudStatus = "Cloud: inserisci Token HF"
            } else {
                val (cloudBmp, statusMsg) = tryCloudSuperResolution(
                    source = working,
                    endpointUrl = cloudEndpointUrl,
                    apiKey = cloudApiKey,
                    isZoomEstremo = isZoomEstremo
                )
                cloudStatus = statusMsg
                if (cloudBmp != null) {
                    if (cloudBmp !== working) working.recycle()
                    working = cloudBmp
                }
            }
        }

        // 5. Micro-contrasto locale e nitidezza adattiva (leggera a 1x, decisa sullo zoom spinto)
        val clarityAmount = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.18f
                AiMode.MEDIUM -> 0.28f
                AiMode.AGGRESSIVE -> 0.36f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.14f
                AiMode.MEDIUM -> 0.22f
                AiMode.AGGRESSIVE -> 0.30f
            }
            else -> when (aiMode) {
                // Foto normali 1x: migliora pulizia e tridimensionalità in modo discreto
                AiMode.LOW -> 0.08f
                AiMode.MEDIUM -> 0.14f
                AiMode.AGGRESSIVE -> 0.20f
            }
        }
        val clarified = try {
            localClarityBoost(working, clarityAmount)
        } catch (_: Throwable) {
            working
        }
        if (clarified !== working && !working.isRecycled) working.recycle()
        working = clarified

        if (postSavePolish) {
            val toned = try {
                applyToneBalance(working, strength = if (isZoomSpinto) 0.65f else 0.50f)
            } catch (_: Throwable) {
                working
            }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        val parts = mutableListOf<String>()
        val scope = if (isZoomSpinto) "Zoom" else "1x"
        if (burstUsed) parts.add("Burst ${frames.size}x")
        if (neuralUsed) parts.add("NN ${aiMode.shortLabel} ($scope)")
        if (cloudStatus.isNotEmpty()) parts.add(cloudStatus)
        if (parts.isEmpty()) parts.add("AI ${aiMode.shortLabel} ($scope)")

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" · ")
        )
    }

    private fun alignAndMergeBurst(base: Bitmap, extras: List<Bitmap>): Bitmap {
        val w = base.width
        val h = base.height
        val total = w * h
        val basePx = IntArray(total)
        base.getPixels(basePx, 0, w, 0, 0, w, h)

        val accR = IntArray(total) { (basePx[it] ushr 16) and 0xFF }
        val accG = IntArray(total) { (basePx[it] ushr 8) and 0xFF }
        val accB = IntArray(total) { basePx[it] and 0xFF }
        val accW = ByteArray(total) { 2 }

        val patchSize = min(140, min(w, h) / 3)
        val cx = w / 2 - patchSize / 2
        val cy = h / 2 - patchSize / 2
        val extPx = IntArray(total)

        for (extra in extras) {
            if (extra.width != w || extra.height != h) continue
            extra.getPixels(extPx, 0, w, 0, 0, w, h)

            var bestDx = 0
            var bestDy = 0
            var bestSad = Long.MAX_VALUE

            for (dy in -8..8 step 2) {
                for (dx in -8..8 step 2) {
                    var sad = 0L
                    for (py in 0 until patchSize step 4) {
                        val rowBase = (cy + py) * w
                        val rowExt = (cy + py + dy) * w
                        for (px in 0 until patchSize step 4) {
                            val bVal = basePx[rowBase + cx + px]
                            val eVal = extPx[rowExt + cx + px + dx]
                            sad += abs(((bVal ushr 8) and 0xFF) - ((eVal ushr 8) and 0xFF))
                        }
                    }
                    if (sad < bestSad) {
                        bestSad = sad
                        bestDx = dx
                        bestDy = dy
                    }
                }
            }

            for (y in 8 until h - 8) {
                val rowB = y * w
                val rowE = (y + bestDy) * w
                for (x in 8 until w - 8) {
                    val idxB = rowB + x
                    val idxE = rowE + (x + bestDx)
                    val bp = basePx[idxB]
                    val ep = extPx[idxE]

                    val br = (bp ushr 16) and 0xFF
                    val bg = (bp ushr 8) and 0xFF
                    val bb = bp and 0xFF

                    val er = (ep ushr 16) and 0xFF
                    val eg = (ep ushr 8) and 0xFF
                    val eb = ep and 0xFF

                    if ((abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3 < 26) {
                        accR[idxB] += er
                        accG[idxB] += eg
                        accB[idxB] += eb
                        accW[idxB] = (accW[idxB] + 1).toByte()
                    }
                }
            }
        }

        for (i in basePx.indices) {
            val weight = accW[i].toInt() and 0xFF
            val r = (accR[i] / weight).coerceIn(0, 255)
            val g = (accG[i] / weight).coerceIn(0, 255)
            val b = (accB[i] / weight).coerceIn(0, 255)
            basePx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(basePx, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * Esegue il modello TFLite leggero su NNAPI/MediaTek:
     * - A 1x usa una griglia 3x3 sull'intera foto con peso residuale leggero (0.28 - 0.55)
     * - Su zoom spinto (8x-60x) concentra una griglia 4x4 sull'area centrale (75% del frame)
     *   dove i tasselli hanno risoluzione quasi 1:1 con l'input della rete neurale, evitando
     *   l'effetto "impastato" e ricostruendo i contorni reali con peso 0.75 - 1.35.
     */
    private fun runTargetedNeuralEnhancement(
        tflite: Interpreter,
        source: Bitmap,
        aiMode: AiMode,
        isTeleLens: Boolean,
        isZoomSpinto: Boolean,
        isZoomEstremo: Boolean
    ): Bitmap {
        val inTensor = tflite.getInputTensor(0)
        val outTensor = tflite.getOutputTensor(0)
        val inShape = inTensor.shape()
        val outShape = outTensor.shape()
        val inType = inTensor.dataType()
        val outType = outTensor.dataType()

        val isNchw = inShape.size == 4 && inShape[1] == 3 && inShape[3] != 3
        val rawInH = if (isNchw) inShape[2] else inShape[1]
        val rawInW = if (isNchw) inShape[3] else inShape[2]
        val inH = (if (rawInH > 0) rawInH else 128).coerceIn(32, 256)
        val inW = (if (rawInW > 0) rawInW else 128).coerceIn(32, 256)

        val outIsNchw = outShape.size == 4 && outShape[1] == 3 && outShape[3] != 3
        val rawOutH = if (outIsNchw) outShape[2] else outShape[1]
        val rawOutW = if (outIsNchw) outShape[3] else outShape[2]
        val outH = (if (rawOutH > 0) rawOutH else inH * 4).coerceAtLeast(inH)
        val outW = (if (rawOutW > 0) rawOutW else inW * 4).coerceAtLeast(inW)

        // Su zoom spinto lavoriamo sul 75% centrale (dove sta il soggetto ingrandito),
        // a 1x lavoriamo sul 92% dell'intera scena
        val coverage = if (isZoomSpinto) 0.75f else 0.92f
        val cols = if (isZoomSpinto) 4 else 3
        val rows = if (isZoomSpinto) 4 else 3

        val workBase = scaleToMaxSide(source, 1600)
        val regW = (workBase.width * coverage).toInt().coerceAtLeast(cols * 32)
        val regH = (workBase.height * coverage).toInt().coerceAtLeast(rows * 32)
        val startX = (workBase.width - regW) / 2
        val startY = (workBase.height - regH) / 2

        val neuralMap = workBase.copy(Bitmap.Config.ARGB_8888, true)
        val baselineMap = workBase.copy(Bitmap.Config.ARGB_8888, true)
        val neuralCanvas = Canvas(neuralMap)
        val baseCanvas = Canvas(baselineMap)

        val cellW = regW / cols
        val cellH = regH / rows

        val bytesPerChannelIn = if (inType == DataType.UINT8) 1 else 4
        val bytesPerChannelOut = if (outType == DataType.UINT8) 1 else 4

        val inByteCount = inTensor.numBytes().coerceAtLeast(1 * inH * inW * 3 * bytesPerChannelIn)
        val outByteCount = outTensor.numBytes().coerceAtLeast(1 * outH * outW * 3 * bytesPerChannelOut)

        val inputBuffer = ByteBuffer.allocateDirect(inByteCount).order(ByteOrder.nativeOrder())
        val outputBuffer = ByteBuffer.allocateDirect(outByteCount).order(ByteOrder.nativeOrder())
        val tilePixels = IntArray(inW * inH)
        val outPixels = IntArray(outW * outH)
        val srPatch = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val left = startX + c * cellW
                val top = startY + r * cellH
                val right = if (c == cols - 1) startX + regW else left + cellW
                val bottom = if (r == rows - 1) startY + regH else top + cellH

                val cw = (right - left).coerceAtLeast(1)
                val ch = (bottom - top).coerceAtLeast(1)

                val crop = Bitmap.createBitmap(workBase, left, top, cw, ch)
                val scaledIn = Bitmap.createScaledBitmap(crop, inW, inH, true)
                crop.recycle()
                scaledIn.getPixels(tilePixels, 0, inW, 0, 0, inW, inH)

                inputBuffer.rewind()
                if (inType == DataType.UINT8) {
                    if (isNchw) {
                        for (p in tilePixels) inputBuffer.put(((p ushr 16) and 0xFF).toByte())
                        for (p in tilePixels) inputBuffer.put(((p ushr 8) and 0xFF).toByte())
                        for (p in tilePixels) inputBuffer.put((p and 0xFF).toByte())
                    } else {
                        for (p in tilePixels) {
                            inputBuffer.put(((p ushr 16) and 0xFF).toByte())
                            inputBuffer.put(((p ushr 8) and 0xFF).toByte())
                            inputBuffer.put((p and 0xFF).toByte())
                        }
                    }
                } else {
                    val scaleIn = if (inH >= 64) 1f / 255f else 1f
                    if (isNchw) {
                        for (p in tilePixels) inputBuffer.putFloat(((p ushr 16) and 0xFF) * scaleIn)
                        for (p in tilePixels) inputBuffer.putFloat(((p ushr 8) and 0xFF) * scaleIn)
                        for (p in tilePixels) inputBuffer.putFloat((p and 0xFF) * scaleIn)
                    } else {
                        for (p in tilePixels) {
                            inputBuffer.putFloat(((p ushr 16) and 0xFF) * scaleIn)
                            inputBuffer.putFloat(((p ushr 8) and 0xFF) * scaleIn)
                            inputBuffer.putFloat((p and 0xFF) * scaleIn)
                        }
                    }
                }
                inputBuffer.rewind()
                outputBuffer.rewind()

                tflite.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                if (outType == DataType.UINT8) {
                    if (outIsNchw) {
                        val plane = outW * outH
                        for (i in 0 until plane) {
                            val red = outputBuffer.get(i).toInt() and 0xFF
                            val green = outputBuffer.get(plane + i).toInt() and 0xFF
                            val blue = outputBuffer.get(2 * plane + i).toInt() and 0xFF
                            outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    } else {
                        for (i in outPixels.indices) {
                            val red = outputBuffer.get().toInt() and 0xFF
                            val green = outputBuffer.get().toInt() and 0xFF
                            val blue = outputBuffer.get().toInt() and 0xFF
                            outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    }
                } else {
                    val scaleOut = if (inH >= 64) 255f else 1f
                    if (outIsNchw) {
                        val plane = outW * outH
                        for (i in 0 until plane) {
                            val red = (outputBuffer.getFloat(i * 4) * scaleOut).toInt().coerceIn(0, 255)
                            val green = (outputBuffer.getFloat((plane + i) * 4) * scaleOut).toInt().coerceIn(0, 255)
                            val blue = (outputBuffer.getFloat((2 * plane + i) * 4) * scaleOut).toInt().coerceIn(0, 255)
                            outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    } else {
                        for (i in outPixels.indices) {
                            val red = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                            val green = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                            val blue = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                            outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    }
                }

                srPatch.setPixels(outPixels, 0, outW, 0, 0, outW, outH)
                val dstRect = Rect(left, top, right, bottom)
                neuralCanvas.drawBitmap(srPatch, null, dstRect, paint)
                baseCanvas.drawBitmap(scaledIn, null, dstRect, paint)
                scaledIn.recycle()
            }
        }
        srPatch.recycle()
        if (workBase !== source) workBase.recycle()

        val fullNeural = Bitmap.createScaledBitmap(neuralMap, source.width, source.height, true)
        val fullBaseline = Bitmap.createScaledBitmap(baselineMap, source.width, source.height, true)
        neuralMap.recycle()
        baselineMap.recycle()

        val residualStrength = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.70f
                AiMode.MEDIUM -> 1.05f
                AiMode.AGGRESSIVE -> 1.35f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.55f
                AiMode.MEDIUM -> 0.85f
                AiMode.AGGRESSIVE -> 1.15f
            }
            isTeleLens -> when (aiMode) {
                AiMode.LOW -> 0.35f
                AiMode.MEDIUM -> 0.55f
                AiMode.AGGRESSIVE -> 0.75f
            }
            else -> when (aiMode) {
                // Foto normali 1x: miglioramento leggero e naturale
                AiMode.LOW -> 0.28f
                AiMode.MEDIUM -> 0.42f
                AiMode.AGGRESSIVE -> 0.58f
            }
        }

        val w = source.width
        val h = source.height
        val origPx = IntArray(w * h)
        val neurPx = IntArray(w * h)
        val basePx = IntArray(w * h)

        source.getPixels(origPx, 0, w, 0, 0, w, h)
        fullNeural.getPixels(neurPx, 0, w, 0, 0, w, h)
        fullBaseline.getPixels(basePx, 0, w, 0, 0, w, h)
        fullNeural.recycle()
        fullBaseline.recycle()

        for (i in origPx.indices) {
            val o = origPx[i]
            val n = neurPx[i]
            val b = basePx[i]

            val or = (o ushr 16) and 0xFF
            val og = (o ushr 8) and 0xFF
            val ob = o and 0xFF

            val yOrig = 0.299f * or + 0.587f * og + 0.114f * ob
            val yNeur = 0.299f * ((n ushr 16) and 0xFF) + 0.587f * ((n ushr 8) and 0xFF) + 0.114f * (n and 0xFF)
            val yBase = 0.299f * ((b ushr 16) and 0xFF) + 0.587f * ((b ushr 8) and 0xFF) + 0.114f * (b and 0xFF)

            val neuralDetail = (yNeur - yBase).coerceIn(-30f, 30f)
            val yEnhanced = (yOrig + neuralDetail * residualStrength).coerceIn(0f, 255f)
            val delta = yEnhanced - yOrig

            val outR = (or + delta).toInt().coerceIn(0, 255)
            val outG = (og + delta).toInt().coerceIn(0, 255)
            val outB = (ob + delta).toInt().coerceIn(0, 255)

            origPx[i] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(origPx, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * Invia l'immagine (o il ritaglio centrale su zoom spinto) al modello Cloud Swin2SR / Diffusion
     * tramite Hugging Face Router API e gestisce sia risposte immagine binarie sia JSON.
     */
    private fun tryCloudSuperResolution(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        isZoomEstremo: Boolean
    ): Pair<Bitmap?, String> {
        return try {
            val cropRatio = if (isZoomEstremo) 0.55f else 0.70f
            val cw = (source.width * cropRatio).toInt().coerceIn(256, source.width)
            val ch = (source.height * cropRatio).toInt().coerceIn(256, source.height)
            val sx = (source.width - cw) / 2
            val sy = (source.height - ch) / 2

            val centerCrop = Bitmap.createBitmap(source, sx, sy, cw, ch)
            val uploadBmp = scaleToMaxSide(centerCrop, 640)
            if (uploadBmp !== centerCrop) centerCrop.recycle()

            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
            uploadBmp.recycle()
            val payload = baos.toByteArray()

            val conn = (URL(endpointUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 7000
                readTimeout = 20000
                doOutput = true
                setRequestProperty("Content-Type", "image/jpeg")
                setRequestProperty("Accept", "image/png, image/jpeg, application/json")
                val token = if (apiKey.trim().startsWith("Bearer ")) apiKey.trim() else "Bearer ${apiKey.trim()}"
                setRequestProperty("Authorization", token)
            }

            conn.outputStream.use { it.write(payload) }
            val code = conn.responseCode

            if (code in 200..299) {
                val cType = conn.contentType ?: ""
                val respBytes = conn.inputStream.use { it.readBytes() }
                val cloudPatch: Bitmap? = if (cType.contains("json", ignoreCase = true)) {
                    val jsonStr = String(respBytes, Charsets.UTF_8)
                    val obj = JSONObject(jsonStr)
                    val b64 = obj.optString("image").ifBlank { obj.optString("output") }
                    if (b64.isNotBlank()) {
                        val cleanB64 = b64.substringAfter("base64,", b64)
                        val decoded = Base64.decode(cleanB64, Base64.DEFAULT)
                        BitmapFactory.decodeByteArray(decoded, 0, decoded.size)
                    } else null
                } else {
                    BitmapFactory.decodeByteArray(respBytes, 0, respBytes.size)
                }

                if (cloudPatch != null) {
                    val result = source.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = Canvas(result)
                    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                        alpha = 210
                    }
                    canvas.drawBitmap(cloudPatch, null, Rect(sx, sy, sx + cw, sy + ch), paint)
                    cloudPatch.recycle()
                    return result to "Cloud OK"
                }
                return null to "Cloud: risposta vuota"
            } else {
                val errHint = when (code) {
                    401, 403 -> "Token HF non valido"
                    404 -> "Modello Cloud non trovato"
                    503 -> "Modello in avvio, riprova"
                    else -> "HTTP $code"
                }
                return null to "Cloud: $errHint"
            }
        } catch (e: Throwable) {
            Log.w("NeuralSR", "Errore Cloud AI: ${e.message}")
            null to "Cloud: offline/timeout"
        }
    }

    private fun scaleToMaxSide(bmp: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bmp.width, bmp.height)
        if (longest <= maxSide) return bmp
        val scale = maxSide.toFloat() / longest.toFloat()
        return Bitmap.createScaledBitmap(
            bmp,
            (bmp.width * scale).toInt().coerceAtLeast(1),
            (bmp.height * scale).toInt().coerceAtLeast(1),
            true
        )
    }

    private fun filterChromaAndNoise(bitmap: Bitmap, chromaPasses: Int, lumaPasses: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height

        var src = IntArray(w * h)
        var dst = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)

        repeat(chromaPasses.coerceIn(0, 2)) {
            for (y in 1 until h - 1) {
                val row = y * w
                for (x in 1 until w - 1) {
                    val idx = row + x
                    val c = src[idx]
                    val r = (c ushr 16) and 0xFF
                    val g = (c ushr 8) and 0xFF
                    val b = c and 0xFF
                    val yCenter = (77 * r + 150 * g + 29 * b) shr 8

                    var sumU = (b - yCenter) * 2
                    var sumV = (r - yCenter) * 2
                    var cnt = 2

                    val offsets = intArrayOf(-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
                    for (off in offsets) {
                        val n = src[idx + off]
                        val nr = (n ushr 16) and 0xFF
                        val ng = (n ushr 8) and 0xFF
                        val nb = n and 0xFF
                        val yN = (77 * nr + 150 * ng + 29 * nb) shr 8
                        if (abs(yCenter - yN) < 28) {
                            sumU += (nb - yN)
                            sumV += (nr - yN)
                            cnt++
                        }
                    }

                    val outR = (yCenter + sumV / cnt).coerceIn(0, 255)
                    val outB = (yCenter + sumU / cnt).coerceIn(0, 255)
                    val outG = ((yCenter * 256 - 77 * outR - 29 * outB) / 150).coerceIn(0, 255)
                    dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
            copyBorders(src, dst, w, h)
            val tmp = src; src = dst; dst = tmp
        }

        repeat(lumaPasses.coerceIn(0, 1)) {
            for (y in 1 until h - 1) {
                val row = y * w
                for (x in 1 until w - 1) {
                    val idx = row + x
                    val c = src[idx]
                    val cr = (c ushr 16) and 0xFF
                    val cg = (c ushr 8) and 0xFF
                    val cb = c and 0xFF

                    var sumR = cr * 3
                    var sumG = cg * 3
                    var sumB = cb * 3
                    var cnt = 3

                    val offsets = intArrayOf(-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
                    for (off in offsets) {
                        val n = src[idx + off]
                        val nr = (n ushr 16) and 0xFF
                        val ng = (n ushr 8) and 0xFF
                        val nb = n and 0xFF
                        if ((abs(cr - nr) + abs(cg - ng) + abs(cb - nb)) / 3 < 18) {
                            sumR += nr
                            sumG += ng
                            sumB += nb
                            cnt++
                        }
                    }
                    dst[idx] = (0xFF shl 24) or
                        ((sumR / cnt).coerceIn(0, 255) shl 16) or
                        ((sumG / cnt).coerceIn(0, 255) shl 8) or
                        (sumB / cnt).coerceIn(0, 255)
                }
            }
            copyBorders(src, dst, w, h)
            val tmp = src; src = dst; dst = tmp
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    private fun localClarityBoost(bitmap: Bitmap, amount: Float): Bitmap {
        if (amount <= 0.02f) return bitmap
        val w = bitmap.width
        val h = bitmap.height
        val small = Bitmap.createScaledBitmap(bitmap, max(1, w / 4), max(1, h / 4), true)
        val blur = Bitmap.createScaledBitmap(small, w, h, true)
        small.recycle()

        val src = IntArray(w * h)
        val blr = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)
        blur.getPixels(blr, 0, w, 0, 0, w, h)
        blur.recycle()

        for (i in src.indices) {
            val s = src[i]
            val b = blr[i]
            val sr = (s ushr 16) and 0xFF
            val sg = (s ushr 8) and 0xFF
            val sb = s and 0xFF
            val dR = sr - ((b ushr 16) and 0xFF)
            val dG = sg - ((b ushr 8) and 0xFF)
            val dB = sb - (b and 0xFF)
            val nr = (sr + dR * amount).toInt().coerceIn(0, 255)
            val ng = (sg + dG * amount).toInt().coerceIn(0, 255)
            val nb = (sb + dB * amount).toInt().coerceIn(0, 255)
            src[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    private fun applyToneBalance(bitmap: Bitmap, strength: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val curve = 0.045 * strength
        val lift = 8.0 * strength

        val lut = IntArray(256)
        for (v in 0..255) {
            val norm = v / 255.0
            val sCurve = norm - curve * sin(2.0 * Math.PI * norm)
            val shadowLift = (lift / 255.0) * (1.0 - norm) * (1.0 - norm)
            lut[v] = ((sCurve + shadowLift) * 255.0).toInt().coerceIn(0, 255)
        }
        for (i in pixels.indices) {
            val p = pixels[i]
            pixels[i] = (0xFF shl 24) or
                (lut[(p ushr 16) and 0xFF] shl 16) or
                (lut[(p ushr 8) and 0xFF] shl 8) or
                lut[p and 0xFF]
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }

    private fun copyBorders(src: IntArray, dst: IntArray, w: Int, h: Int) {
        for (x in 0 until w) {
            dst[x] = src[x]
            dst[(h - 1) * w + x] = src[(h - 1) * w + x]
        }
        for (y in 0 until h) {
            dst[y * w] = src[y * w]
            dst[y * w + (w - 1)] = src[y * w + (w - 1)]
        }
    }
}
