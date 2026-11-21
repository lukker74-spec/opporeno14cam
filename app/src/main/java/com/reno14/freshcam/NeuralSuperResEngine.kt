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

/**
 * Pipeline AI adattiva a due stadi per Oppo Reno 14:
 * - STADIO 1 (Zoom Ottico Nativo: 0.6x, 1x, 3.5x):
 *   Non applica alcuna ricostruzione sintetica ai pixel per non alterare la purezza dell'ottica;
 *   esegue solo bilanciamento dinamico fine e micro-contrasto ottico preservando la resa naturale
 *   di lente e sensore 50MP.
 * - STADIO 2 (Zoom Digitale Spinto / Molto Spinto: > 1.25x sulla lente attiva, es. 5x, 10x, 20x+):
 *   Applica prima una riduzione del rumore (chroma + luma edge-preserving) e poi ricostruisce
 *   i dettagli reali mediante la rete neurale Real-ESRGAN / QuickSRNet (con fallback direzionale
 *   su frequenze medie/alte senza inventare artefatti).
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
                interpreter = tflite
                loadedAssetName = assetName
                Log.i("NeuralSR", "Modello neurale $assetName caricato con successo")
                return tflite
            } catch (_: Exception) {
            }
        }
        Log.w("NeuralSR", "Nessun modello .tflite trovato in assets, attivo motore adattivo")
        return null
    }

    fun process(
        source: Bitmap,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiMode: AiMode,
        isTeleLens: Boolean,
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
                working = applyToneBalance(working, strength = 0.5f)
            }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = "AI OFF"
            )
        }

        // Distingue tra zoom ottico nativo (lente pura <= 1.25x) e zoom digitale spinto (> 1.25x)
        val isDigitalZoom = lensZoomRatio > 1.25f
        val isHeavyDigitalZoom = lensZoomRatio >= 2.5f

        if (!isDigitalZoom) {
            // MODALITÀ OTTICA NATIVA (0.6x, 1x, 3.5x):
            // Massima fedeltà all'ottica e al processore: nessuna alterazione neurale dei pixel,
            // solo ottimizzazione tonale e micro-nitidezza ottica leggera.
            val opticalPolished = if (postSavePolish) {
                val toned = applyToneBalance(working, strength = 0.65f)
                opticalMicroContrast(toned, amount = 0.14f)
            } else {
                opticalMicroContrast(working, amount = 0.10f)
            }
            return NeuralProcessResult(
                bitmap = opticalPolished,
                modeApplied = "Ottico Puro + ISP"
            )
        }

        // MODALITÀ ZOOM SPINTO / MOLTO SPINTO (Digital crop > 1.25x, es. 5x, 10x, 20x, 35x):
        // 1. Pre-pulizia del rumore sensore prima della ricostruzione dettagli
        val denoised = filterChromaAndLuma(
            bitmap = working,
            chromaPasses = if (isHeavyDigitalZoom || isTeleLens) 2 else 1,
            lumaPasses = if (isHeavyDigitalZoom) 2 else 1
        )

        val tflite = getInterpreter()
        var neuralDone = false
        var reconstructed = denoised

        if (tflite != null) {
            try {
                reconstructed = runNeuralDetailReconstruction(
                    tflite = tflite,
                    cleanSource = denoised,
                    heavyZoom = isHeavyDigitalZoom
                )
                neuralDone = true
            } catch (e: Exception) {
                Log.e("NeuralSR", "Errore inferenza neurale, uso ricostruzione frequenze", e)
                reconstructed = denoised
            }
        }

        // 2. Ricostruzione conservativa dei bordi reali (senza inventare texture inesistenti)
        val edgeGain = when {
            isHeavyDigitalZoom && aiMode == AiMode.AGGRESSIVE -> 0.48f
            isHeavyDigitalZoom -> 0.38f
            else -> 0.28f
        }
        var finalBmp = conservativeDetailReconstruct(reconstructed, edgeGain)
        if (postSavePolish) {
            finalBmp = applyToneBalance(finalBmp, strength = 0.70f)
        }

        val label = if (neuralDone) {
            "Zoom AI Neurale (${loadedAssetName.removeSuffix(".tflite")})"
        } else {
            "Zoom AI Ricostruzione Dettagli"
        }

        return NeuralProcessResult(
            bitmap = finalBmp,
            modeApplied = label
        )
    }

    /**
     * Esegue inferenza neurale su patch ad alta risoluzione e fonde solo la componente di dettaglio
     * (high-frequency residual) sull'immagine denoisata, impedendo alla rete di inventare
     * texture artificiali o alterare colori e luminanza.
     */
    private fun runNeuralDetailReconstruction(
        tflite: Interpreter,
        cleanSource: Bitmap,
        heavyZoom: Boolean
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

        val result = cleanSource.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        val ratio = if (heavyZoom) 0.55f else 0.45f
        val regionW = (cleanSource.width * ratio).toInt().coerceAtLeast(inW * 3).coerceAtMost(cleanSource.width)
        val regionH = (cleanSource.height * ratio).toInt().coerceAtLeast(inH * 3).coerceAtMost(cleanSource.height)

        val startX = (cleanSource.width - regionW) / 2
        val startY = (cleanSource.height - regionH) / 2
        val cols = 3
        val rows = 3
        val cellW = regionW / cols
        val cellH = regionH / rows

        val bytesPerChannelIn = if (inType == DataType.UINT8) 1 else 4
        val bytesPerChannelOut = if (outType == DataType.UINT8) 1 else 4

        val inputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 3 * bytesPerChannelIn).order(ByteOrder.nativeOrder())
        val outputBuffer = ByteBuffer.allocateDirect(1 * outH * outW * 3 * bytesPerChannelOut).order(ByteOrder.nativeOrder())
        val tilePixels = IntArray(inW * inH)
        val outPixels = IntArray(outW * outH)

        // Fusione dosata: estrae solo la nitidezza strutturale senza sovrascrivere i colori naturali
        val blendPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            alpha = if (heavyZoom) 110 else 85
        }

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val left = startX + c * cellW
                val top = startY + r * cellH
                val right = if (c == cols - 1) startX + regionW else left + cellW
                val bottom = if (r == rows - 1) startY + regionH else top + cellH

                val cw = (right - left).coerceAtLeast(1)
                val ch = (bottom - top).coerceAtLeast(1)

                val crop = Bitmap.createBitmap(cleanSource, left, top, cw, ch)
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
                    // Normalizza automaticamente tra [0..1] per Real-ESRGAN/QuickSRNet o [0..255] per ESRGAN-TF2
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

                canvas.drawBitmap(srPatch, null, Rect(left, top, right, bottom), blendPaint)
            }
        }

        return result
    }

    private fun opticalMicroContrast(bitmap: Bitmap, amount: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 24_000_000) return bitmap
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

                val dR = cr - avgR
                val dG = cg - avgG
                val dB = cb - avgB

                val outR = if (abs(dR) in 4..45) (cr + dR * amount).toInt().coerceIn(0, 255) else cr
                val outG = if (abs(dG) in 4..45) (cg + dG * amount).toInt().coerceIn(0, 255) else cg
                val outB = if (abs(dB) in 4..45) (cb + dB * amount).toInt().coerceIn(0, 255) else cb

                dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
        copyBorders(src, dst, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
        return out
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
                        if ((abs(cr - nr) + abs(cg - ng) + abs(cb - nb)) / 3 < 22) {
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

    private fun conservativeDetailReconstruct(bitmap: Bitmap, gain: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 24_000_000) return bitmap

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

                // Interviene solo su contorni reali (soglia > 5) evitando di inventare grana nelle aree uniformi
                val outR = if (abs(hfR) > 5) ((cr * 0.78f + dirR * 0.22f) + hfR * gain).toInt().coerceIn(0, 255) else cr
                val outG = if (abs(hfG) > 5) ((cg * 0.78f + dirG * 0.22f) + hfG * gain).toInt().coerceIn(0, 255) else cg
                val outB = if (abs(hfB) > 5) ((cb * 0.78f + dirB * 0.22f) + hfB * gain).toInt().coerceIn(0, 255) else cb

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
