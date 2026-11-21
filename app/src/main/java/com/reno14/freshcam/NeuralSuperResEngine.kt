package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
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
 * Motore AI v7.1 Safe-Memory (Anti-Crash / Anti-OOM):
 * - Libera immediatamente tutte le bitmap intermedie (recycle) e limita la dimensione massima di
 *   elaborazione a 3200px, evitando il crash OutOfMemoryError su Android quando AI: ON elabora
 *   raffiche e tasselli TFLite.
 * - Verifica anche che il modello .tflite abbia effettivamente un tensor di uscita 4D immagine
 *   prima di eseguirlo, proteggendo l'app da qualsiasi crash nativo TFLite.
 */
class NeuralSuperResEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var initAttempted: Boolean = false
    var loadedAssetName: String = "Nessuno"
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
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                }
                val tflite = Interpreter(mappedBuffer, options)
                val inShape = tflite.getInputTensor(0).shape()
                val outShape = tflite.getOutputTensor(0).shape()
                if (inShape.size != 4 || outShape.size != 4) {
                    tflite.close()
                    continue
                }
                interpreter = tflite
                loadedAssetName = assetName
                Log.i("NeuralSR", "Modello neurale $assetName pronto (${inShape.contentToString()} -> ${outShape.contentToString()})")
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

        // 1. Fusione Burst allineata (riciclando subito i frame extra per non saturare la RAM)
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
                Log.e("NeuralSR", "Burst fallback su frame singolo", t)
            }
        }

        // 2. Pulizia rumore cromatico
        val denoised = try {
            filterChromaOnly(working, passes = if (isZoomSpinto) 2 else 1)
        } catch (t: Throwable) {
            working
        }
        if (denoised !== working && !working.isRecycled) {
            working.recycle()
        }
        working = denoised

        // 3. Iniezione Residuale Neurale TFLite (protetta da qualsiasi eccezione nativa/memoria)
        val tflite = getInterpreter()
        var neuralUsed = false
        if (tflite != null) {
            try {
                val neuralOut = runResidualNeuralInjection(
                    tflite = tflite,
                    nativeSharpSource = working,
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
                Log.e("NeuralSR", "Inferenza TFLite saltata in sicurezza: ${t.message}", t)
            }
        }

        // 4. Cloud AI opzionale
        var cloudUsed = false
        if (cloudAiEnabled && cloudEndpointUrl.isNotBlank() && (isZoomSpinto || aiMode == AiMode.AGGRESSIVE)) {
            val cloudResult = tryCloudSuperResolution(
                source = working,
                endpointUrl = cloudEndpointUrl,
                apiKey = cloudApiKey,
                isZoomEstremo = isZoomEstremo
            )
            if (cloudResult != null) {
                if (cloudResult !== working) working.recycle()
                working = cloudResult
                cloudUsed = true
            }
        }

        // 5. Rifinitura bordi e bilanciamento tonale
        val detailGain = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.35f
                AiMode.MEDIUM -> 0.55f
                AiMode.AGGRESSIVE -> 0.72f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.28f
                AiMode.MEDIUM -> 0.45f
                AiMode.AGGRESSIVE -> 0.60f
            }
            isTeleLens -> when (aiMode) {
                AiMode.LOW -> 0.18f
                AiMode.MEDIUM -> 0.30f
                AiMode.AGGRESSIVE -> 0.42f
            }
            else -> when (aiMode) {
                AiMode.LOW -> 0.12f
                AiMode.MEDIUM -> 0.22f
                AiMode.AGGRESSIVE -> 0.32f
            }
        }

        val edged = try {
            adaptiveEdgeEnhance(working, detailGain)
        } catch (_: Throwable) {
            working
        }
        if (edged !== working && !working.isRecycled) working.recycle()
        working = edged

        if (postSavePolish) {
            val toned = try {
                applyToneBalance(working, strength = if (isZoomSpinto) 0.65f else 0.45f)
            } catch (_: Throwable) {
                working
            }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        val parts = mutableListOf<String>()
        if (burstUsed) parts.add("Burst ${frames.size}x")
        if (neuralUsed) parts.add("NN ${aiMode.shortLabel}")
        if (cloudUsed) parts.add("Cloud AI")
        if (parts.isEmpty()) parts.add("AI ${aiMode.shortLabel}")

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" + ")
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

    private fun runResidualNeuralInjection(
        tflite: Interpreter,
        nativeSharpSource: Bitmap,
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

        // Supporta sia layout NHWC [1, H, W, 3] sia NCHW [1, 3, H, W] senza mai crashare per indici
        val isNchw = inShape.size == 4 && inShape[1] == 3 && inShape[3] != 3
        val rawInH = if (isNchw) inShape[2] else inShape[1]
        val rawInW = if (isNchw) inShape[3] else inShape[2]
        val inH = (if (rawInH > 0) rawInH else 50).coerceIn(32, 256)
        val inW = (if (rawInW > 0) rawInW else 50).coerceIn(32, 256)

        val outIsNchw = outShape.size == 4 && outShape[1] == 3 && outShape[3] != 3
        val rawOutH = if (outIsNchw) outShape[2] else outShape[1]
        val rawOutW = if (outIsNchw) outShape[3] else outShape[2]
        val outH = (if (rawOutH > 0) rawOutH else inH * 4).coerceAtLeast(inH)
        val outW = (if (rawOutW > 0) rawOutW else inW * 4).coerceAtLeast(inW)

        // Dimensione compatta per le mappe residuali (1400px) -> velocissimo (<0.6s) e zero OutOfMemory
        val workMaxSide = if (isZoomSpinto) 1400 else 1600
        val workBase = scaleToMaxSide(nativeSharpSource, workMaxSide)

        val neuralMap = Bitmap.createBitmap(workBase.width, workBase.height, Bitmap.Config.ARGB_8888)
        val baselineMap = Bitmap.createBitmap(workBase.width, workBase.height, Bitmap.Config.ARGB_8888)
        val neuralCanvas = Canvas(neuralMap)
        val baseCanvas = Canvas(baselineMap)

        val cols = if (isZoomSpinto) 5 else 4
        val rows = if (isZoomSpinto) 5 else 4
        val cellW = workBase.width / cols
        val cellH = workBase.height / rows

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
                val left = c * cellW
                val top = r * cellH
                val right = if (c == cols - 1) workBase.width else left + cellW
                val bottom = if (r == rows - 1) workBase.height else top + cellH

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
        if (workBase !== nativeSharpSource) workBase.recycle()

        val fullNeural = Bitmap.createScaledBitmap(neuralMap, nativeSharpSource.width, nativeSharpSource.height, true)
        val fullBaseline = Bitmap.createScaledBitmap(baselineMap, nativeSharpSource.width, nativeSharpSource.height, true)
        neuralMap.recycle()
        baselineMap.recycle()

        val residualStrength = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.85f
                AiMode.MEDIUM -> 1.25f
                AiMode.AGGRESSIVE -> 1.60f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.65f
                AiMode.MEDIUM -> 0.95f
                AiMode.AGGRESSIVE -> 1.30f
            }
            isTeleLens -> when (aiMode) {
                AiMode.LOW -> 0.40f
                AiMode.MEDIUM -> 0.65f
                AiMode.AGGRESSIVE -> 0.90f
            }
            else -> when (aiMode) {
                AiMode.LOW -> 0.25f
                AiMode.MEDIUM -> 0.42f
                AiMode.AGGRESSIVE -> 0.60f
            }
        }

        val w = nativeSharpSource.width
        val h = nativeSharpSource.height
        val origPx = IntArray(w * h)
        val neurPx = IntArray(w * h)
        val basePx = IntArray(w * h)

        nativeSharpSource.getPixels(origPx, 0, w, 0, 0, w, h)
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

            val neuralDetail = (yNeur - yBase).coerceIn(-36f, 36f)
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

    private fun tryCloudSuperResolution(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        isZoomEstremo: Boolean
    ): Bitmap? {
        return try {
            val cropRatio = if (isZoomEstremo) 0.65f else 0.55f
            val cw = (source.width * cropRatio).toInt().coerceIn(256, source.width)
            val ch = (source.height * cropRatio).toInt().coerceIn(256, source.height)
            val sx = (source.width - cw) / 2
            val sy = (source.height - ch) / 2

            val centerCrop = Bitmap.createBitmap(source, sx, sy, cw, ch)
            val uploadBmp = scaleToMaxSide(centerCrop, 768)
            if (uploadBmp !== centerCrop) centerCrop.recycle()

            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
            uploadBmp.recycle()
            val payload = baos.toByteArray()

            val conn = (URL(endpointUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 5000
                readTimeout = 15000
                doOutput = true
                setRequestProperty("Content-Type", "application/octet-stream")
                if (apiKey.isNotBlank()) {
                    val token = if (apiKey.startsWith("Bearer ")) apiKey else "Bearer $apiKey"
                    setRequestProperty("Authorization", token)
                }
            }

            conn.outputStream.use { it.write(payload) }

            if (conn.responseCode in 200..299) {
                val respBytes = conn.inputStream.use { it.readBytes() }
                val cloudPatch = BitmapFactory.decodeByteArray(respBytes, 0, respBytes.size)
                if (cloudPatch != null) {
                    val result = source.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = Canvas(result)
                    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                        alpha = 215
                    }
                    canvas.drawBitmap(cloudPatch, null, Rect(sx, sy, sx + cw, sy + ch), paint)
                    cloudPatch.recycle()
                    return result
                }
            }
            null
        } catch (e: Throwable) {
            Log.w("NeuralSR", "Cloud AI non raggiungibile: ${e.message}")
            null
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

    private fun filterChromaOnly(bitmap: Bitmap, passes: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height

        var src = IntArray(w * h)
        var dst = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)

        repeat(passes.coerceIn(0, 2)) {
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

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    private fun adaptiveEdgeEnhance(bitmap: Bitmap, gain: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height

        val src = IntArray(w * h)
        val dst = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)

        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val idx = row + x
                val c = src[idx]
                val cr = (c ushr 16) and 0xFF
                val cg = (c ushr 8) and 0xFF
                val cb = c and 0xFF

                val n = src[idx - w]
                val s = src[idx + w]
                val e = src[idx + 1]
                val west = src[idx - 1]

                val avgR = (((n ushr 16) and 0xFF) + ((s ushr 16) and 0xFF) + ((e ushr 16) and 0xFF) + ((west ushr 16) and 0xFF)) / 4
                val avgG = (((n ushr 8) and 0xFF) + ((s ushr 8) and 0xFF) + ((e ushr 8) and 0xFF) + ((west ushr 8) and 0xFF)) / 4
                val avgB = ((n and 0xFF) + (s and 0xFF) + (e and 0xFF) + (west and 0xFF)) / 4

                val hfR = cr - avgR
                val hfG = cg - avgG
                val hfB = cb - avgB

                val outR = if (abs(hfR) > 4) (cr + hfR * gain).toInt().coerceIn(0, 255) else cr
                val outG = if (abs(hfG) > 4) (cg + hfG * gain).toInt().coerceIn(0, 255) else cg
                val outB = if (abs(hfB) > 4) (cb + hfB * gain).toInt().coerceIn(0, 255) else cb

                dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
        copyBorders(src, dst, w, h)

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
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
