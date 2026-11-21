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

data class NeuralProcessResult(
    val bitmap: Bitmap,
    val neuralApplied: Boolean,
    val detailNote: String
)

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
                working = applyToneBalance(working)
            }
            return NeuralProcessResult(
                bitmap = working,
                neuralApplied = false,
                detailNote = "SR: OFF"
            )
        }

        val tflite = getInterpreter()
        if (tflite == null) {
            if (postSavePolish) {
                working = applyToneBalance(working)
            }
            return NeuralProcessResult(
                bitmap = working,
                neuralApplied = false,
                detailNote = "Modello .tflite non trovato nell'APK"
            )
        }

        return try {
            val enhanced = runEsrganTiledInference(tflite, working)
            val finalBmp = if (postSavePolish) applyToneBalance(enhanced) else enhanced
            NeuralProcessResult(
                bitmap = finalBmp,
                neuralApplied = true,
                detailNote = "Rete Neurale ESRGAN applicata"
            )
        } catch (e: Exception) {
            Log.e("NeuralSR", "Errore inferenza ESRGAN", e)
            NeuralProcessResult(
                bitmap = working,
                neuralApplied = false,
                detailNote = "Errore inferenza: ${e.localizedMessage}"
            )
        }
    }

    /**
     * Esegue inferenza neurale reale ESRGAN (50x50 -> 200x200, fattore 4x)
     * su una griglia centrale 4x4 di tasselli ad alta definizione e ricompone il risultato
     * sull'immagine salvata, rendendo visibile la ricostruzione generativa della rete neurale.
     */
    private fun runEsrganTiledInference(tflite: Interpreter, source: Bitmap): Bitmap {
        val inShape = tflite.getInputTensor(0).shape() // [1, 50, 50, 3]
        val outShape = tflite.getOutputTensor(0).shape() // [1, 200, 200, 3]

        val inH = if (inShape.size == 4 && inShape[1] > 0) inShape[1] else 50
        val inW = if (inShape.size == 4 && inShape[2] > 0) inShape[2] else 50
        val outH = if (outShape.size == 4 && outShape[1] > 0) outShape[1] else inH * 4
        val outW = if (outShape.size == 4 && outShape[2] > 0) outShape[2] else inW * 4

        val resultBitmap = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

        // Elabora una regione centrale pari al 70% dell'inquadratura suddivisa in tasselli (4x4)
        val gridCols = 4
        val gridRows = 4
        val regionW = (source.width * 0.70f).toInt().coerceAtLeast(inW * gridCols)
            .coerceAtMost(source.width)
        val regionH = (source.height * 0.70f).toInt().coerceAtLeast(inH * gridRows)
            .coerceAtMost(source.height)

        val startX = (source.width - regionW) / 2
        val startY = (source.height - regionH) / 2
        val cellW = regionW / gridCols
        val cellH = regionH / gridRows

        val inputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 3 * 4).order(ByteOrder.nativeOrder())
        val outputBuffer = ByteBuffer.allocateDirect(1 * outH * outW * 3 * 4).order(ByteOrder.nativeOrder())
        val tilePixels = IntArray(inW * inH)
        val outPixels = IntArray(outW * outH)

        for (r in 0 until gridRows) {
            for (c in 0 until gridCols) {
                val srcLeft = startX + c * cellW
                val srcTop = startY + r * cellH
                val srcRight = if (c == gridCols - 1) startX + regionW else srcLeft + cellW
                val srcBottom = if (r == gridRows - 1) startY + regionH else srcTop + cellH

                val cropW = (srcRight - srcLeft).coerceAtLeast(1)
                val cropH = (srcBottom - srcTop).coerceAtLeast(1)

                val crop = Bitmap.createBitmap(source, srcLeft, srcTop, cropW, cropH)
                val modelInputBmp = Bitmap.createScaledBitmap(crop, inW, inH, true)
                modelInputBmp.getPixels(tilePixels, 0, inW, 0, 0, inW, inH)

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

                val srTile = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                srTile.setPixels(outPixels, 0, outW, 0, 0, outW, outH)

                val dstRect = Rect(srcLeft, srcTop, srcRight, srcBottom)
                canvas.drawBitmap(srTile, null, dstRect, paint)
            }
        }

        return resultBitmap
    }

    private fun applyToneBalance(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val lut = IntArray(256)
        for (v in 0..255) {
            val norm = v / 255.0
            val sCurve = norm - 0.04 * kotlin.math.sin(2.0 * Math.PI * norm)
            val shadowLift = (8.0 / 255.0) * (1.0 - norm) * (1.0 - norm)
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
}
