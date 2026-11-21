package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
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
    val neuralApplied: Boolean,
    val detailNote: String
)

/**
 * Motore ibrido ottimizzato (Scelta migliore tra Opzione A e Opzione B):
 * 1. Pulisce il rumore cromatico e di luminanza PRIMA dell'elaborazione, evitando che il rumore
 *    del sensore teleobiettivo venga amplificato.
 * 2. Applica ESRGAN (se presente) su piccoli ritagli nativi senza schiacciare intere porzioni
 *    a bassa risoluzione, fondendo il dettaglio neurale con l'immagine pulita per evitare artefatti.
 * 3. Applica un recupero direzionale dei bordi calibrato per preservare naturalezza.
 */
class NeuralSuperResEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var initAttempted: Boolean = false

    @Synchronized
    fun isModelAvailable(): Boolean {
        return getInterpreter() != null
    }

    @Synchronized
    private fun getInterpreter(): Interpreter? {
        if (initAttempted) return interpreter
        initAttempted = true
        return try {
            val afd = context.assets.openFd("esrgan.tflite")
            if (afd.declaredLength < 1024L) {
                Log.w("NeuralSR", "File esrgan.tflite vuoto o non valido")
                return null
            }
            val inputStream = FileInputStream(afd.fileDescriptor)
            val fileChannel = inputStream.channel
            val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            Interpreter(mappedBuffer, options).also {
                interpreter = it
                Log.i("NeuralSR", "Modello neurale esrgan.tflite caricato con successo")
            }
        } catch (e: Exception) {
            Log.e("NeuralSR", "Impossibile caricare esrgan.tflite: ${e.message}")
            null
        }
    }

    fun process(
        source: Bitmap,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiMode: AiMode,
        isTeleLens: Boolean,
        zoomRatio: Float,
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
                working = applyToneBalance(working, strong = false)
            }
            return NeuralProcessResult(
                bitmap = working,
                neuralApplied = false,
                detailNote = "SR: OFF"
            )
        }

        val extremeZoom = isTeleLens && zoomRatio >= 3.0f
        val highZoom = isTeleLens && zoomRatio >= 1.5f

        // 1. Pre-denoise mirato per eliminare la grana del sensore prima di qualsiasi dettaglio
        val denoised = filterChromaAndLuma(
            bitmap = working,
            chromaPasses = if (extremeZoom || isTeleLens) 2 else 1,
            lumaPasses = if (extremeZoom || aiMode == AiMode.AGGRESSIVE) 2 else 1
        )

        val tflite = getInterpreter()
        var neuralUsed = false
        var baseAfterNeural = denoised

        if (tflite != null) {
            try {
                baseAfterNeural = runSelectiveEsrganBlend(
                    tflite = tflite,
                    cleanSource = denoised,
                    extremeZoom = extremeZoom,
                    highZoom = highZoom
                )
                neuralUsed = true
            } catch (e: Exception) {
                Log.e("NeuralSR", "Errore inferenza ESRGAN selettiva, uso fusione locale", e)
                baseAfterNeural = denoised
            }
        }

        // 2. Recupero direzionale controllato e fusione naturale per non impastare
        val detailGain = when {
            extremeZoom && aiMode == AiMode.AGGRESSIVE -> 0.55f
            extremeZoom -> 0.42f
            highZoom || isTeleLens -> 0.35f
            aiMode == AiMode.AGGRESSIVE -> 0.40f
            else -> 0.28f
        }
        var blended = edgeDirectedBlend(baseAfterNeural, detailGain)
        if (postSavePolish) {
            blended = applyToneBalance(blended, strong = false)
        }

        val note = if (neuralUsed) {
            "SR Ibrido (ESRGAN + Denoise)"
        } else {
            "SR Ibrido (Denoise + Dettagli)"
        }

        return NeuralProcessResult(
            bitmap = blended,
            neuralApplied = true,
            detailNote = note
        )
    }

    /**
     * Elabora con ESRGAN solo un riquadro centrale contenuto (50% del lato) diviso in una griglia
     * 3x3 di patch native senza schiacciamenti estremi, fondendo il risultato con alpha moderato
     * per evitare artefatti e stacchi visivi rispetto al resto della foto.
     */
    private fun runSelectiveEsrganBlend(
        tflite: Interpreter,
        cleanSource: Bitmap,
        extremeZoom: Boolean,
        highZoom: Boolean
    ): Bitmap {
        val inShape = tflite.getInputTensor(0).shape()
        val outShape = tflite.getOutputTensor(0).shape()

        val inH = if (inShape.size == 4 && inShape[1] > 0) inShape[1] else 50
        val inW = if (inShape.size == 4 && inShape[2] > 0) inShape[2] else 50
        val outH = if (outShape.size == 4 && outShape[1] > 0) outShape[1] else inH * 4
        val outW = if (outShape.size == 4 && outShape[2] > 0) outShape[2] else inW * 4

        val result = cleanSource.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        val ratio = if (extremeZoom) 0.45f else if (highZoom) 0.50f else 0.40f
        val regionW = (cleanSource.width * ratio).toInt().coerceAtLeast(inW * 3).coerceAtMost(cleanSource.width)
        val regionH = (cleanSource.height * ratio).toInt().coerceAtLeast(inH * 3).coerceAtMost(cleanSource.height)

        val startX = (cleanSource.width - regionW) / 2
        val startY = (cleanSource.height - regionH) / 2
        val cols = 3
        val rows = 3
        val cellW = regionW / cols
        val cellH = regionH / rows

        val inputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 3 * 4).order(ByteOrder.nativeOrder())
        val outputBuffer = ByteBuffer.allocateDirect(1 * outH * outW * 3 * 4).order(ByteOrder.nativeOrder())
        val tilePixels = IntArray(inW * inH)
        val outPixels = IntArray(outW * outH)

        // Fusione moderata per mantenere texture reale e zero aloni
        val blendAlpha = if (extremeZoom) 115 else 90
        val blendPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            alpha = blendAlpha
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
                for (p in tilePixels) {
                    inputBuffer.putFloat(((p ushr 16) and 0xFF).toFloat())
                    inputBuffer.putFloat(((p ushr 8) and 0xFF).toFloat())
                    inputBuffer.putFloat((p and 0xFF).toFloat())
                }
                inputBuffer.rewind()
                outputBuffer.rewind()

                tflite.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                for (i in outPixels.indices) {
                    val red = outputBuffer.float.toInt().coerceIn(0, 255)
                    val green = outputBuffer.float.toInt().coerceIn(0, 255)
                    val blue = outputBuffer.float.toInt().coerceIn(0, 255)
                    outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                }

                val srPatch = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                srPatch.setPixels(outPixels, 0, outW, 0, 0, outW, outH)

                canvas.drawBitmap(srPatch, null, Rect(left, top, right, bottom), blendPaint)
            }
        }

        return result
    }

    private fun filterChromaAndLuma(bitmap: Bitmap, chromaPasses: Int, lumaPasses: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 20_000_000) return bitmap

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
                        if ((abs(cr - nr) + abs(cg - ng) + abs(cb - nb)) / 3 < 24) {
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

    private fun edgeDirectedBlend(bitmap: Bitmap, gain: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 20_000_000) return bitmap

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

                // Soglia più alta (4) per non amplificare la grana fine
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

    private fun applyToneBalance(bitmap: Bitmap, strong: Boolean): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val curve = if (strong) 0.045 else 0.03
        val lift = if (strong) 9.0 else 6.0

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
