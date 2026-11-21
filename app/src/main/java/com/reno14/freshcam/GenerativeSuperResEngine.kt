package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin

/**
 * Motore di Super-Resolution Generativa on-device (con controllo esplicito dell'utente):
 * - Tenta di caricare un modello neurale opzionale `super_res_gan.tflite` dagli asset se presente.
 * - Se il modello non è presente negli asset, applica un algoritmo generativo di sintesi dei dettagli
 *   e ricostruzione dei bordi basato su gradiente locale + micro-trama coerente, solo quando l'utente
 *   ha attivato `generativeSuperResEnabled = true`.
 */
class GenerativeSuperResEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var modelChecked: Boolean = false

    @Synchronized
    private fun getOrLoadModel(): Interpreter? {
        if (modelChecked) return interpreter
        modelChecked = true
        return try {
            val assetFd = context.assets.openFd("super_res_gan.tflite")
            val inputStream = assetFd.createInputStream()
            val bytes = inputStream.readBytes()
            inputStream.close()
            assetFd.close()

            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes)
            buffer.rewind()
            Interpreter(buffer).also { interpreter = it }
        } catch (_: Exception) {
            Log.d("GenerativeSuperRes", "Modello super_res_gan.tflite non presente negli asset: uso sintetizzatore locale")
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
    ): Bitmap {
        var current = profile.applyToBitmap(
            source = source,
            strength = profileIntensity,
            brightnessBoost = brightnessBoost
        )

        val highZoom = isTeleLens && zoomRatio >= 1.8f
        val extremeZoom = isTeleLens && zoomRatio >= 3.5f

        // Pulizia rumore cromatico e luminanza solo se richiesto dall'AI o dalla Super-Resolution
        if (superResEnabled || aiMode != AiMode.LOW) {
            val chromaPasses = if (extremeZoom && superResEnabled) 2 else 1
            val lumaPasses = if (extremeZoom && superResEnabled && aiMode != AiMode.LOW) 2 else 1
            current = filterChromaAndLuma(current, chromaPasses, lumaPasses)
        }

        // Applica la Super-Resolution Generativa SOLO se l'utente l'ha attivata esplicitamente
        if (superResEnabled) {
            val model = getOrLoadModel()
            if (model != null) {
                current = runNeuralTileSuperRes(model, current)
            } else {
                val gain = when {
                    extremeZoom && aiMode == AiMode.AGGRESSIVE -> 1.15f
                    extremeZoom -> 0.95f
                    highZoom -> 0.80f
                    else -> 0.65f
                }
                current = generativeDetailSynthesis(current, gain)
            }
        } else {
            val sharp = when (aiMode) {
                AiMode.LOW -> 0.15f
                AiMode.MEDIUM -> 0.35f
                AiMode.AGGRESSIVE -> 0.55f
            }
            current = cleanUnsharp(current, sharp)
        }

        if (postSavePolish) {
            current = adaptiveTonePolish(current)
        }

        return current
    }

    private fun runNeuralTileSuperRes(interpreter: Interpreter, bitmap: Bitmap): Bitmap {
        return try {
            val inTensor = interpreter.getInputTensor(0)
            val shape = inTensor.shape() // atteso [1, H, W, 3]
            if (shape.size != 4) return generativeDetailSynthesis(bitmap, 0.90f)
            val tileH = shape[1]
            val tileW = shape[2]
            if (tileW <= 0 || tileH <= 0) return generativeDetailSynthesis(bitmap, 0.90f)

            val scaled = Bitmap.createScaledBitmap(bitmap, tileW, tileH, true)
            val pixels = IntArray(tileW * tileH)
            scaled.getPixels(pixels, 0, tileW, 0, 0, tileW, tileH)

            val inputBuf = ByteBuffer.allocateDirect(1 * tileH * tileW * 3 * 4).order(ByteOrder.nativeOrder())
            for (p in pixels) {
                inputBuf.putFloat(((p ushr 16) and 0xFF) / 255f)
                inputBuf.putFloat(((p ushr 8) and 0xFF) / 255f)
                inputBuf.putFloat((p and 0xFF) / 255f)
            }
            inputBuf.rewind()

            val outTensor = interpreter.getOutputTensor(0)
            val outShape = outTensor.shape()
            val outH = outShape[1]
            val outW = outShape[2]
            val outBuf = ByteBuffer.allocateDirect(1 * outH * outW * 3 * 4).order(ByteOrder.nativeOrder())

            interpreter.run(inputBuf, outBuf)
            outBuf.rewind()

            val outPixels = IntArray(outW * outH)
            for (i in outPixels.indices) {
                val r = (outBuf.float * 255f).toInt().coerceIn(0, 255)
                val g = (outBuf.float * 255f).toInt().coerceIn(0, 255)
                val b = (outBuf.float * 255f).toInt().coerceIn(0, 255)
                outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            val srBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            srBmp.setPixels(outPixels, 0, outW, 0, 0, outW, outH)
            Bitmap.createScaledBitmap(srBmp, bitmap.width, bitmap.height, true)
        } catch (e: Exception) {
            Log.w("GenerativeSuperRes", "Fallback su sintesi generativa locale: ${e.message}")
            generativeDetailSynthesis(bitmap, 0.90f)
        }
    }

    private fun generativeDetailSynthesis(bitmap: Bitmap, gain: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w * h > 14_000_000) return bitmap

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

                val detailGain = gain * 0.90f
                val outR = if (abs(hfR) > 3) ((cr * 0.72f + dirR * 0.28f) + hfR * detailGain).toInt().coerceIn(0, 255) else cr
                val outG = if (abs(hfG) > 3) ((cg * 0.72f + dirG * 0.28f) + hfG * detailGain).toInt().coerceIn(0, 255) else cg
                val outB = if (abs(hfB) > 3) ((cb * 0.72f + dirB * 0.28f) + hfB * detailGain).toInt().coerceIn(0, 255) else cb

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
        if (w * h > 14_000_000) return bitmap

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
                        if (abs(yCenter - yN) < 25) {
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

    private fun cleanUnsharp(bitmap: Bitmap, strength: Float): Bitmap {
        if (strength <= 0.02f) return bitmap
        val w = bitmap.width
        val h = bitmap.height
        val blur = Bitmap.createScaledBitmap(
            Bitmap.createScaledBitmap(bitmap, max(1, w / 3), max(1, h / 3), true),
            w, h, true
        )
        val src = IntArray(w * h)
        val blr = IntArray(w * h)
        bitmap.getPixels(src, 0, w, 0, 0, w, h)
        blur.getPixels(blr, 0, w, 0, 0, w, h)

        for (i in src.indices) {
            val s = src[i]
            val b = blr[i]
            val sr = (s ushr 16) and 0xFF
            val sg = (s ushr 8) and 0xFF
            val sb = s and 0xFF
            val dR = sr - ((b ushr 16) and 0xFF)
            val dG = sg - ((b ushr 8) and 0xFF)
            val dB = sb - (b and 0xFF)
            val nr = if (abs(dR) > 4) (sr + dR * strength).toInt().coerceIn(0, 255) else sr
            val ng = if (abs(dG) > 4) (sg + dG * strength).toInt().coerceIn(0, 255) else sg
            val nb = if (abs(dB) > 4) (sb + dB * strength).toInt().coerceIn(0, 255) else sb
            src[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    private fun adaptiveTonePolish(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val lut = IntArray(256)
        for (v in 0..255) {
            val norm = v / 255.0
            val sCurve = norm - 0.045 * sin(2.0 * Math.PI * norm)
            val shadowLift = (10.0 / 255.0) * (1.0 - norm) * (1.0 - norm)
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
