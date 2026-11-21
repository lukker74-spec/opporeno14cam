package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
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
                interpreter = tflite
                loadedAssetName = assetName
                Log.i("NeuralSR", "Modello neurale $assetName caricato con successo")
                return tflite
            } catch (_: Exception) {
            }
        }
        Log.w("NeuralSR", "Nessun modello .tflite trovato in assets")
        return null
    }

    fun process(
        source: Bitmap,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiMode: AiMode,
        isTeleLens: Boolean,
        displayZoom: Float,
        lensZoomRatio: Float,
        superResEnabled: Boolean,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        var working = profile.applyToBitmap(
            source = source,
            strength = profileIntensity,
            brightnessBoost = brightnessBoost
        )

        if (!superResEnabled) {
            if (postSavePolish) {
                working = applyToneBalance(working, strength = 0.45f)
            }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = "AI OFF"
            )
        }

        val isZoomSpinto = displayZoom >= 8.0f || lensZoomRatio >= 2.2f
        val isZoomEstremo = displayZoom >= 25.0f || lensZoomRatio >= 7.0f

        val denoised = filterChromaAndLuma(
            bitmap = working,
            chromaPasses = when {
                isZoomEstremo -> 2
                isZoomSpinto || isTeleLens -> 2
                else -> 1
            },
            lumaPasses = when {
                isZoomEstremo -> 2
                isZoomSpinto && aiMode != AiMode.LOW -> 1
                else -> 0
            }
        )

        val tflite = getInterpreter()
        var neuralUsed = false
        var afterNeural = denoised

        if (tflite != null) {
            try {
                afterNeural = runNeuralReconstruction(
                    tflite = tflite,
                    cleanSource = denoised,
                    aiMode = aiMode,
                    isTeleLens = isTeleLens,
                    isZoomSpinto = isZoomSpinto,
                    isZoomEstremo = isZoomEstremo
                )
                neuralUsed = true
            } catch (e: Exception) {
                Log.e("NeuralSR", "Errore inferenza TFLite", e)
                afterNeural = denoised
            }
        }

        val detailGain = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.55f
                AiMode.MEDIUM -> 0.78f
                AiMode.AGGRESSIVE -> 0.95f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.42f
                AiMode.MEDIUM -> 0.62f
                AiMode.AGGRESSIVE -> 0.80f
            }
            isTeleLens -> when (aiMode) {
                AiMode.LOW -> 0.25f
                AiMode.MEDIUM -> 0.40f
                AiMode.AGGRESSIVE -> 0.55f
            }
            else -> when (aiMode) {
                AiMode.LOW -> 0.16f
                AiMode.MEDIUM -> 0.26f
                AiMode.AGGRESSIVE -> 0.36f
            }
        }

        var finalBmp = guidedEdgeSharpen(afterNeural, detailGain, isZoomSpinto)
        if (postSavePolish) {
            val toneStrength = if (isZoomSpinto) 0.75f else 0.50f
            finalBmp = applyToneBalance(finalBmp, strength = toneStrength)
        }

        val zoomTag = when {
            isZoomEstremo -> "Zoom Estremo"
            isZoomSpinto -> "Zoom Spinto"
            isTeleLens -> "Tele"
            else -> "1x"
        }
        val label = if (neuralUsed) {
            "Rete Neurale ${aiMode.shortLabel} ($zoomTag)"
        } else {
            "AI ${aiMode.shortLabel} ($zoomTag · senza .tflite)"
        }

        return NeuralProcessResult(
            bitmap = finalBmp,
            modeApplied = label
        )
    }

    private fun runNeuralReconstruction(
        tflite: Interpreter,
        cleanSource: Bitmap,
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

        val inH = if (inShape.size == 4 && inShape[1] > 0) inShape[1] else 50
        val inW = if (inShape.size == 4 && inShape[2] > 0) inShape[2] else 50
        val outH = if (outShape.size == 4 && outShape[1] > 0) outShape[1] else inH * 4
        val outW = if (outShape.size == 4 && outShape[2] > 0) outShape[2] else inW * 4

        val workMaxSide = if (isZoomSpinto) 2400 else 3200
        val workBase = scaleToMaxSide(cleanSource, workMaxSide)
        val neuralCanvasBmp = workBase.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(neuralCanvasBmp)

        val cols = if (isZoomSpinto) 6 else 4
        val rows = if (isZoomSpinto) 6 else 4

        val cellW = workBase.width / cols
        val cellH = workBase.height / rows

        val bytesPerChannelIn = if (inType == DataType.UINT8) 1 else 4
        val bytesPerChannelOut = if (outType == DataType.UINT8) 1 else 4

        val inputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 3 * bytesPerChannelIn).order(ByteOrder.nativeOrder())
        val outputBuffer = ByteBuffer.allocateDirect(1 * outH * outW * 3 * bytesPerChannelOut).order(ByteOrder.nativeOrder())
        val tilePixels = IntArray(inW * inH)
        val outPixels = IntArray(outW * outH)

        val tilePaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        val overlapX = (cellW * 0.12f).toInt()
        val overlapY = (cellH * 0.12f).toInt()

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val baseLeft = c * cellW
                val baseTop = r * cellH
                val baseRight = if (c == cols - 1) workBase.width else baseLeft + cellW
                val baseBottom = if (r == rows - 1) workBase.height else baseTop + cellH

                val srcLeft = (baseLeft - overlapX).coerceAtLeast(0)
                val srcTop = (baseTop - overlapY).coerceAtLeast(0)
                val srcRight = (baseRight + overlapX).coerceAtMost(workBase.width)
                val srcBottom = (baseBottom + overlapY).coerceAtMost(workBase.height)

                val cw = (srcRight - srcLeft).coerceAtLeast(1)
                val ch = (srcBottom - srcTop).coerceAtLeast(1)

                val crop = Bitmap.createBitmap(workBase, srcLeft, srcTop, cw, ch)
                val scaledIn = Bitmap.createScaledBitmap(crop, inW, inH, true)
                scaledIn.getPixels(tilePixels, 0, inW, 0, 0, inW, inH)

                inputBuffer.rewind()
                if (inType == DataType.UINT8) {
                    for (p in tilePixels) {
                        inputBuffer.put(((p ushr 16) and 0xFF).toByte())
                        inputBuffer.put(((p ushr 8) and 0xFF).toByte())
                        inputBuffer.put((p and 0xFF).toByte())
                    }
                } else {
                    val scaleIn = if (loadedAssetName == "superres.tflite" && inH >= 64) 1f / 255f else 1f
                    for (p in tilePixels) {
                        inputBuffer.putFloat(((p ushr 16) and 0xFF) * scaleIn)
                        inputBuffer.putFloat(((p ushr 8) and 0xFF) * scaleIn)
                        inputBuffer.putFloat((p and 0xFF) * scaleIn)
                    }
                }
                inputBuffer.rewind()
                outputBuffer.rewind()

                tflite.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                if (outType == DataType.UINT8) {
                    for (i in outPixels.indices) {
                        val red = outputBuffer.get().toInt() and 0xFF
                        val green = outputBuffer.get().toInt() and 0xFF
                        val blue = outputBuffer.get().toInt() and 0xFF
                        outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                    }
                } else {
                    val scaleOut = if (loadedAssetName == "superres.tflite" && inH >= 64) 255f else 1f
                    for (i in outPixels.indices) {
                        val red = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        val green = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        val blue = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                    }
                }

                val srPatch = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                srPatch.setPixels(outPixels, 0, outW, 0, 0, outW, outH)

                canvas.drawBitmap(srPatch, null, Rect(srcLeft, srcTop, srcRight, srcBottom), tilePaint)
            }
        }

        val fullNeural = if (neuralCanvasBmp.width != cleanSource.width || neuralCanvasBmp.height != cleanSource.height) {
            Bitmap.createScaledBitmap(neuralCanvasBmp, cleanSource.width, cleanSource.height, true)
        } else {
            neuralCanvasBmp
        }

        val lumaWeight = when {
            isZoomEstremo -> when (aiMode) {
                AiMode.LOW -> 0.65f
                AiMode.MEDIUM -> 0.82f
                AiMode.AGGRESSIVE -> 0.92f
            }
            isZoomSpinto -> when (aiMode) {
                AiMode.LOW -> 0.55f
                AiMode.MEDIUM -> 0.72f
                AiMode.AGGRESSIVE -> 0.85f
            }
            isTeleLens -> when (aiMode) {
                AiMode.LOW -> 0.35f
                AiMode.MEDIUM -> 0.50f
                AiMode.AGGRESSIVE -> 0.65f
            }
            else -> when (aiMode) {
                AiMode.LOW -> 0.22f
                AiMode.MEDIUM -> 0.32f
                AiMode.AGGRESSIVE -> 0.42f
            }
        }

        return fuseNeuralStructureWithRealColor(cleanSource, fullNeural, lumaWeight)
    }

    private fun fuseNeuralStructureWithRealColor(
        original: Bitmap,
        neural: Bitmap,
        structureWeight: Float
    ): Bitmap {
        val w = original.width
        val h = original.height
        val origPx = IntArray(w * h)
        val neurPx = IntArray(w * h)
        original.getPixels(origPx, 0, w, 0, 0, w, h)
        neural.getPixels(neurPx, 0, w, 0, 0, w, h)

        val invWeight = 1f - structureWeight
        val chromaWeight = (structureWeight * 0.35f).coerceIn(0.08f, 0.32f)
        val invChroma = 1f - chromaWeight

        for (i in origPx.indices) {
            val o = origPx[i]
            val n = neurPx[i]

            val or = (o ushr 16) and 0xFF
            val og = (o ushr 8) and 0xFF
            val ob = o and 0xFF

            val nr = (n ushr 16) and 0xFF
            val ng = (n ushr 8) and 0xFF
            val nb = n and 0xFF

            val yOrig = 0.299f * or + 0.587f * og + 0.114f * ob
            val yNeur = 0.299f * nr + 0.587f * ng + 0.114f * nb

            val clampedYNeur = yNeur.coerceIn(yOrig - 42f, yOrig + 42f)
            val yFused = yOrig * invWeight + clampedYNeur * structureWeight

            val uOrig = ob - yOrig
            val vOrig = or - yOrig
            val uNeur = nb - yNeur
            val vNeur = nr - yNeur

            val uFused = uOrig * invChroma + uNeur * chromaWeight
            val vFused = vOrig * invChroma + vNeur * chromaWeight

            val outR = (yFused + vFused).toInt().coerceIn(0, 255)
            val outB = (yFused + uFused).toInt().coerceIn(0, 255)
            val outG = ((yFused - 0.299f * outR - 0.114f * outB) / 0.587f).toInt().coerceIn(0, 255)

            origPx[i] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
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
            (bmp.width * scale).toInt().coerceAtLeast(1),
            (bmp.height * scale).toInt().coerceAtLeast(1),
            true
        )
    }

    private fun filterChromaAndLuma(bitmap: Bitmap, chromaPasses: Int, lumaPasses: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 24_000_000) return bitmap

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

        repeat(lumaPasses.coerceIn(0, 2)) {
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
                        if ((abs(cr - nr) + abs(cg - ng) + abs(cb - nb)) / 3 < 20) {
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

    private fun guidedEdgeSharpen(bitmap: Bitmap, gain: Float, isZoomSpinto: Boolean): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 24_000_000) return bitmap

        val src = IntArray(w * h)
        val dst = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)

        val edgeThreshold = if (isZoomSpinto) 4 else 5

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
                val nw = src[idx - w - 1]
                val ne = src[idx - w + 1]
                val sw = src[idx + w - 1]
                val se = src[idx + w + 1]

                val gradH = abs(((e ushr 8) and 0xFF) - ((west ushr 8) and 0xFF))
                val gradV = abs(((n ushr 8) and 0xFF) - ((s ushr 8) and 0xFF))
                val gradD1 = abs(((nw ushr 8) and 0xFF) - ((se ushr 8) and 0xFF))
                val gradD2 = abs(((ne ushr 8) and 0xFF) - ((sw ushr 8) and 0xFF))

                val (p1, p2) = when (minOf(gradH, gradV, gradD1, gradD2)) {
                    gradH -> e to west
                    gradV -> n to s
                    gradD1 -> nw to se
                    else -> ne to sw
                }

                val dirR = (((p1 ushr 16) and 0xFF) + ((p2 ushr 16) and 0xFF)) / 2
                val dirG = (((p1 ushr 8) and 0xFF) + ((p2 ushr 8) and 0xFF)) / 2
                val dirB = ((p1 and 0xFF) + (p2 and 0xFF)) / 2

                val avgR = (((n ushr 16) and 0xFF) + ((s ushr 16) and 0xFF) + ((e ushr 16) and 0xFF) + ((west ushr 16) and 0xFF)) / 4
                val avgG = (((n ushr 8) and 0xFF) + ((s ushr 8) and 0xFF) + ((e ushr 8) and 0xFF) + ((west ushr 8) and 0xFF)) / 4
                val avgB = ((n and 0xFF) + (s and 0xFF) + (e and 0xFF) + (west and 0xFF)) / 4

                val hfR = cr - avgR
                val hfG = cg - avgG
                val hfB = cb - avgB

                val outR = if (abs(hfR) > edgeThreshold) ((cr * 0.76f + dirR * 0.24f) + hfR * gain).toInt().coerceIn(0, 255) else cr
                val outG = if (abs(hfG) > edgeThreshold) ((cg * 0.76f + dirG * 0.24f) + hfG * gain).toInt().coerceIn(0, 255) else cg
                val outB = if (abs(hfB) > edgeThreshold) ((cb * 0.76f + dirB * 0.24f) + hfB * gain).toInt().coerceIn(0, 255) else cb

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

        val curve = 0.05 * strength
        val lift = 9.0 * strength

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
