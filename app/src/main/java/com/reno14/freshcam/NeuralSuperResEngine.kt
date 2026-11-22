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
 * Motore Fotografico & AI v12.0 (Generale, Pulito, Potente):
 * - Nessuna modalità rigida che rovina altri scatti: il motore si adatta automaticamente
 *   alla lente (IMX882 1x vs Tele JN5 3.5x-7x), al fattore di zoom (fino a 120x) e al movimento.
 * - Anti-Ghosting automatico nel Burst: se rileva movimento nella scena scarta i frame mossi
 *   invece di creare doppie sagome.
 * - Teleobiettivo JN5 (3.5x-7x): deconvoluzione ottica dedicata anche con AI OFF.
 * - Zoom spinto (20x-120x): ritaglio digitale ad alta precisione + Burst anti-ghosting + NNAPI + Cloud AI.
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
    fun isModelAvailable(): Boolean = getInterpreter() != null

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

                var tflite: Interpreter? = null
                try {
                    val delegate = NnApiDelegate()
                    val nnOpts = Interpreter.Options().apply {
                        addDelegate(delegate)
                        setNumThreads(4)
                    }
                    tflite = Interpreter(mappedBuffer, nnOpts)
                    nnApiDelegate = delegate
                    acceleratorName = "MediaTek APU 780 (NNAPI)"
                } catch (_: Throwable) {
                    nnApiDelegate?.close()
                    nnApiDelegate = null
                    val cpuOpts = Interpreter.Options().apply {
                        setNumThreads(4)
                    }
                    tflite = Interpreter(mappedBuffer, cpuOpts)
                    acceleratorName = "CPU Dimensity 4T"
                }

                val inShape = tflite!!.getInputTensor(0).shape()
                val outShape = tflite.getOutputTensor(0).shape()
                if (inShape.size != 4 || outShape.size != 4) {
                    tflite.close()
                    continue
                }
                interpreter = tflite
                loadedAssetName = assetName
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
        aiPreset: UnifiedAiPreset,
        activeLens: LensType,
        displayZoom: Float,
        digitalExtraZoom: Float,
        cloudEndpointUrl: String,
        cloudApiKey: String,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        // Se lo zoom richiesto supera il massimo zoom hardware della lente (es. 60x-120x),
        // esegue un ritaglio digitale pulito prima dell'elaborazione AI
        val croppedFrames = if (digitalExtraZoom > 1.03f) {
            frames.map { applyDigitalCrop(it, digitalExtraZoom) }
        } else {
            frames
        }

        val primaryFrame = croppedFrames.first()
        var working = profile.applyToBitmap(
            source = primaryFrame,
            strength = profileIntensity,
            brightnessBoost = brightnessBoost
        )

        val isTeleLens = activeLens == LensType.TELE_50MP

        if (!aiPreset.isEnabled) {
            if (isTeleLens) {
                val teleOpt = applyTeleOpticalDeconvolution(working, strength = 0.34f)
                if (teleOpt !== working) working.recycle()
                working = teleOpt
            }
            if (postSavePolish) {
                val toned = applyTrueColorToneCurve(working, strength = 0.30f)
                if (toned !== working) working.recycle()
                working = toned
            }
            if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = if (isTeleLens) "Tele Ottico (AI Off)" else "Naturale (AI Off)"
            )
        }

        val isOpticalRange = displayZoom <= 7.05f
        val isTeleSpinto = displayZoom >= 8.0f
        val isTeleEstremo = displayZoom >= 20.0f
        val level = aiPreset.level

        // 1. Burst Multi-Frame con protezione automatica Anti-Ghosting (solo sopra i 7x ottici)
        var burstUsed = false
        if (isTeleSpinto && croppedFrames.size > 1) {
            try {
                val gradedExtras = croppedFrames.drop(1).map {
                    profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                }
                val merged = alignAndMergeBurstAntiGhost(working, gradedExtras)
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
        if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }

        // 2. Pulizia rumore calibrata (solo chroma sotto 7x, chroma + luma leggera sopra 20x)
        val denoised = try {
            filterChromaAndNoise(
                bitmap = working,
                chromaPasses = if (isTeleSpinto || isTeleLens) 2 else 1,
                lumaPasses = if (isTeleEstremo) 1 else 0
            )
        } catch (_: Throwable) {
            working
        }
        if (denoised !== working && !working.isRecycled) working.recycle()
        working = denoised

        // 3. Motore Neurale TFLite (MediaTek APU 780 NNAPI)
        val tflite = getInterpreter()
        var neuralUsed = false
        if (tflite != null) {
            try {
                val neuralOut = runNeuralDetailEngine(
                    tflite = tflite,
                    source = working,
                    level = level,
                    isOpticalRange = isOpticalRange,
                    isTeleEstremo = isTeleEstremo
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

        // 4. Cloud AI (quando selezionato Cloud Bassa / Media / Alta dal tasto unico AI)
        var cloudStatus = ""
        if (aiPreset.isCloud) {
            val cloudAlpha = when (level) {
                1 -> 150
                2 -> 200
                else -> 240
            }
            val (cloudBmp, statusMsg) = tryCloudSuperResolution(
                source = working,
                endpointUrl = cloudEndpointUrl,
                apiKey = cloudApiKey,
                isTeleEstremo = isTeleEstremo,
                blendAlpha = cloudAlpha
            )
            cloudStatus = statusMsg
            if (cloudBmp != null) {
                if (cloudBmp !== working) working.recycle()
                working = cloudBmp
            }
        }

        // 5. Deconvoluzione ottica dedicata per Tele JN5 o Micro-dettaglio per Principale IMX882
        val teleBonus = if (isTeleLens) 0.14f else 0.0f
        val clarityGain = when {
            isTeleEstremo -> when (level) {
                1 -> 0.24f
                2 -> 0.34f
                else -> 0.44f
            }
            isTeleSpinto -> when (level) {
                1 -> 0.20f
                2 -> 0.28f
                else -> 0.36f
            }
            else -> when (level) {
                1 -> 0.14f + teleBonus
                2 -> 0.22f + teleBonus
                else -> 0.30f + teleBonus
            }
        }
        val detailed = try {
            if (isTeleLens) applyTeleOpticalDeconvolution(working, strength = clarityGain)
            else applyMultiScaleDetail(working, gain = clarityGain)
        } catch (_: Throwable) {
            working
        }
        if (detailed !== working && !working.isRecycled) working.recycle()
        working = detailed

        if (postSavePolish) {
            val toned = try {
                applyTrueColorToneCurve(working, strength = if (isTeleSpinto) 0.45f else 0.30f)
            } catch (_: Throwable) {
                working
            }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        val parts = mutableListOf<String>()
        if (burstUsed) parts.add("Burst")
        if (neuralUsed) parts.add(aiPreset.shortBadge)
        if (cloudStatus.isNotEmpty()) parts.add(cloudStatus)
        if (parts.isEmpty()) parts.add(aiPreset.shortBadge)

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" · ")
        )
    }

    private fun applyDigitalCrop(bmp: Bitmap, factor: Float): Bitmap {
        val f = factor.coerceIn(1.0f, 4.0f)
        val newW = (bmp.width / f).toInt().coerceAtLeast(256)
        val newH = (bmp.height / f).toInt().coerceAtLeast(256)
        val sx = (bmp.width - newW) / 2
        val sy = (bmp.height - newH) / 2
        val crop = Bitmap.createBitmap(bmp, sx, sy, newW, newH)
        val upscaled = Bitmap.createScaledBitmap(crop, bmp.width, bmp.height, true)
        crop.recycle()
        return upscaled
    }

    private fun applyTeleOpticalDeconvolution(bitmap: Bitmap, strength: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val src = IntArray(w * h)
        val dst = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)

        val fineGain = (strength * 1.15f).coerceIn(0.15f, 0.70f)
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

                val avgR = (((n ushr 16) and 0xFF) + ((s ushr 16) and 0xFF) + ((e ushr 16) and 0xFF) + ((west ushr 16) and 0xFF)) shr 2
                val avgG = (((n ushr 8) and 0xFF) + ((s ushr 8) and 0xFF) + ((e ushr 8) and 0xFF) + ((west ushr 8) and 0xFF)) shr 2
                val avgB = ((n and 0xFF) + (s and 0xFF) + (e and 0xFF) + (west and 0xFF)) shr 2

                val dR = (cr - avgR).coerceIn(-22, 22)
                val dG = (cg - avgG).coerceIn(-22, 22)
                val dB = (cb - avgB).coerceIn(-22, 22)

                val nr = if (abs(dR) > 3) (cr + dR * fineGain).toInt().coerceIn(0, 255) else cr
                val ng = if (abs(dG) > 3) (cg + dG * fineGain).toInt().coerceIn(0, 255) else cg
                val nb = if (abs(dB) > 3) (cb + dB * fineGain).toInt().coerceIn(0, 255) else cb

                dst[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }
        copyBorders(src, dst, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
        return out
    }

    private fun alignAndMergeBurstAntiGhost(base: Bitmap, extras: List<Bitmap>): Bitmap {
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

            // Soglia severa anti-ghosting (16): se un pixel si è spostato tra i frame viene ignorato
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

                    if ((abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3 < 16) {
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

    private fun runNeuralDetailEngine(
        tflite: Interpreter,
        source: Bitmap,
        level: Int,
        isOpticalRange: Boolean,
        isTeleEstremo: Boolean
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

        val coverage = if (!isOpticalRange) 0.72f else 0.94f
        val cols = if (isTeleEstremo) 6 else if (!isOpticalRange) 5 else 4
        val rows = if (isTeleEstremo) 6 else if (!isOpticalRange) 5 else 4

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
            isTeleEstremo -> when (level) {
                1 -> 0.85f
                2 -> 1.15f
                else -> 1.40f
            }
            !isOpticalRange -> when (level) {
                1 -> 0.60f
                2 -> 0.85f
                else -> 1.10f
            }
            else -> when (level) {
                1 -> 0.30f
                2 -> 0.45f
                else -> 0.60f
            }
        }

        val detailCap = if (isOpticalRange) 18f else 26f
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

            val rawDetail = (yNeur - yBase).coerceIn(-detailCap, detailCap)
            val cleanDetail = if (abs(rawDetail) < 2.2f) 0f else rawDetail
            val yEnhanced = (yOrig + cleanDetail * residualStrength).coerceIn(0f, 255f)
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
        isTeleEstremo: Boolean,
        blendAlpha: Int
    ): Pair<Bitmap?, String> {
        return try {
            val cropRatio = if (isTeleEstremo) 0.55f else 0.72f
            val cw = (source.width * cropRatio).toInt().coerceIn(256, source.width)
            val ch = (source.height * cropRatio).toInt().coerceIn(256, source.height)
            val sx = (source.width - cw) / 2
            val sy = (source.height - ch) / 2

            val centerCrop = Bitmap.createBitmap(source, sx, sy, cw, ch)
            val uploadBmp = scaleToMaxSide(centerCrop, 768)
            if (uploadBmp !== centerCrop) centerCrop.recycle()

            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 94, baos)
            uploadBmp.recycle()
            val payload = baos.toByteArray()

            val conn = (URL(endpointUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 8000
                readTimeout = 25000
                doOutput = true
                setRequestProperty("Content-Type", "image/jpeg")
                setRequestProperty("Accept", "image/png, image/jpeg, application/json")
                if (apiKey.isNotBlank()) {
                    val token = if (apiKey.trim().startsWith("Bearer ")) apiKey.trim() else "Bearer ${apiKey.trim()}"
                    setRequestProperty("Authorization", token)
                }
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
                        alpha = blendAlpha.coerceIn(100, 255)
                    }
                    canvas.drawBitmap(cloudPatch, null, Rect(sx, sy, sx + cw, sy + ch), paint)
                    cloudPatch.recycle()
                    return result to "Cloud OK"
                }
                return null to "Cloud: risposta vuota"
            } else {
                val errHint = when (code) {
                    401, 403 -> "Token HF richiesto"
                    404 -> "Endpoint non trovato"
                    503 -> "Modello in avvio"
                    else -> "HTTP $code"
                }
                return null to "Cloud: $errHint"
            }
        } catch (e: Throwable) {
            Log.w("NeuralSR", "Errore Cloud AI: ${e.message}")
            null to "Cloud: timeout/rete"
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

    private fun applyMultiScaleDetail(bitmap: Bitmap, gain: Float): Bitmap {
        if (gain <= 0.02f) return bitmap
        val w = bitmap.width
        val h = bitmap.height
        val small = Bitmap.createScaledBitmap(bitmap, max(1, w / 3), max(1, h / 3), true)
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

            val lum = (77 * sr + 150 * sg + 29 * sb) shr 8
            val lumProtection = when {
                lum > 224 -> ((255 - lum) / 31f).coerceIn(0.20f, 1.0f)
                lum < 18 -> (lum / 18f).coerceIn(0.30f, 1.0f)
                else -> 1.0f
            }
            val effGain = gain * lumProtection

            val dR = (sr - ((b ushr 16) and 0xFF)).coerceIn(-24, 24)
            val dG = (sg - ((b ushr 8) and 0xFF)).coerceIn(-24, 24)
            val dB = (sb - (b and 0xFF)).coerceIn(-24, 24)

            val nr = (sr + dR * effGain).toInt().coerceIn(0, 255)
            val ng = (sg + dG * effGain).toInt().coerceIn(0, 255)
            val nb = (sb + dB * effGain).toInt().coerceIn(0, 255)
            src[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    private fun applyTrueColorToneCurve(bitmap: Bitmap, strength: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val curve = 0.025 * strength
        val lift = 2.0 * strength

        val lut = IntArray(256)
        for (v in 0..255) {
            val norm = v / 255.0
            val sCurve = norm - curve * sin(2.0 * Math.PI * norm)
            val shadowLift = (lift / 255.0) * (1.0 - norm) * (1.0 - norm) * (1.0 - norm)
            val mapped = sCurve + shadowLift
            val safe = if (mapped > 0.88) 0.88 + (mapped - 0.88) * 0.88 else mapped
            lut[v] = (safe * 255.0).toInt().coerceIn(0, 255)
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
