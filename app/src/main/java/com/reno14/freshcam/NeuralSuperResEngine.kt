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
 * Motore AI v7.0 con architettura Multi-Frame Burst + Rete Neurale (SwinIR / ESRGAN) + Cloud AI opzionale:
 *
 * Perché nella v6 l'AI sembrava peggiorare lo scatto rispetto a AI OFF:
 * Nella v6 prendevamo un riquadro grande e lo rimpicciolivamo a 50x50 pixel prima di darlo alla rete,
 * distruggendo i dettagli reali dell'immagine originale.
 *
 * Come funziona invece ora nella v7.0:
 * 1. MULTI-FRAME BURST ALIGNMENT (Scatto multiplo allineato):
 *    Quando AI: ON è attivo, fonde 3 fotogrammi catturati in sequenza rapida allineandoli tra loro
 *    per abbattere il rumore reale del sensore e recuperare segnale vero.
 * 2. RETE NEURALE IN MODALITÀ RESIDUALE (High-Frequency Detail Injection):
 *    La rete neurale (.tflite) NON sostituisce mai i pixel originali con una versione sfocata/rimpicciolita:
 *    calcoliamo solo la differenza di dettaglio ad alta frequenza (dettaglio neurale - base neurale)
 *    e la sommiamo direttamente ai pixel originali ad altissima risoluzione!
 *    In questo modo la foto non perde MAI nitidezza rispetto ad AI: OFF, ma guadagna definizione
 *    reale e pulizia sia a 1x sia sullo zoom spinto fino a 60x.
 * 3. CLOUD AI GENERATIVA OPZIONALE (attivabile/disattivabile):
 *    Se attivata nelle impostazioni (con API Key gratuita Hugging Face / endpoint compatibile),
 *    invia il ritaglio centrale dello zoom spinto al modello Swin2SR/Diffusion in Cloud per una
 *    ricostruzione generativa ad altissima definizione.
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
        Log.w("NeuralSR", "Nessun modello .tflite trovato in assets")
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
                working = applyToneBalance(working, strength = 0.45f)
            }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = "AI OFF"
            )
        }

        val isZoomSpinto = displayZoom >= 8.0f || lensZoomRatio >= 2.2f
        val isZoomEstremo = displayZoom >= 25.0f || lensZoomRatio >= 7.0f

        // 1. MULTI-FRAME BURST FUSION: se abbiamo catturato più frame in raffica, li allineiamo e fondiamo
        //    per ridurre il rumore reale e aumentare la nitidezza nativa prima della rete neurale
        var burstMerged = working
        var burstUsed = false
        if (frames.size > 1) {
            val gradedExtras = frames.drop(1).map {
                profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
            }
            burstMerged = alignAndMergeBurst(working, gradedExtras)
            burstUsed = true
        }

        // 2. Pulizia rumore cromatico sensibile ai bordi
        val denoised = filterChromaOnly(
            bitmap = burstMerged,
            passes = if (isZoomSpinto) 2 else 1
        )

        // 3. RETE NEURALE RESIDUALE (SwinIR / ESRGAN): inietta dettagli ad alta frequenza
        //    senza mai degradare o sfocare i pixel nativi della foto originale
        val tflite = getInterpreter()
        var neuralUsed = false
        var afterNeural = denoised

        if (tflite != null) {
            try {
                afterNeural = runResidualNeuralInjection(
                    tflite = tflite,
                    nativeSharpSource = denoised,
                    aiMode = aiMode,
                    isTeleLens = isTeleLens,
                    isZoomSpinto = isZoomSpinto,
                    isZoomEstremo = isZoomEstremo
                )
                neuralUsed = true
            } catch (e: Exception) {
                Log.e("NeuralSR", "Errore inferenza TFLite residuale", e)
                afterNeural = denoised
            }
        }

        // 4. CLOUD AI GENERATIVA OPZIONALE (solo se attivata dall'utente e su zoom spinto o AI Alto)
        var cloudUsed = false
        if (cloudAiEnabled && cloudEndpointUrl.isNotBlank() && (isZoomSpinto || aiMode == AiMode.AGGRESSIVE)) {
            val cloudResult = tryCloudSuperResolution(
                source = afterNeural,
                endpointUrl = cloudEndpointUrl,
                apiKey = cloudApiKey,
                isZoomEstremo = isZoomEstremo
            )
            if (cloudResult != null) {
                afterNeural = cloudResult
                cloudUsed = true
            }
        }

        // 5. Rifinitura micro-contrasto adattivo calibrata
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

        var finalBmp = adaptiveEdgeEnhance(afterNeural, detailGain)
        if (postSavePolish) {
            finalBmp = applyToneBalance(finalBmp, strength = if (isZoomSpinto) 0.65f else 0.45f)
        }

        val parts = mutableListOf<String>()
        if (burstUsed) parts.add("Burst ${frames.size}x")
        if (neuralUsed) parts.add("NN ${aiMode.shortLabel}")
        if (cloudUsed) parts.add("Cloud AI")
        if (parts.isEmpty()) parts.add("AI ${aiMode.shortLabel}")

        return NeuralProcessResult(
            bitmap = finalBmp,
            modeApplied = parts.joinToString(" + ")
        )
    }

    /**
     * Allinea i fotogrammi della raffica cercando lo spostamento (dx, dy) con minima differenza
     * al centro dell'immagine e li media pesando la nitidezza, abbattendo il rumore del sensore.
     */
    private fun alignAndMergeBurst(base: Bitmap, extras: List<Bitmap>): Bitmap {
        val w = base.width
        val h = base.height
        val basePx = IntArray(w * h)
        base.getPixels(basePx, 0, w, 0, 0, w, h)

        val accR = IntArray(w * h) { (basePx[it] ushr 16) and 0xFF }
        val accG = IntArray(w * h) { (basePx[it] ushr 8) and 0xFF }
        val accB = IntArray(w * h) { basePx[it] and 0xFF }
        val accW = IntArray(w * h) { 2 } // peso doppia fiducia al frame principale

        val patchSize = min(160, min(w, h) / 3)
        val cx = w / 2 - patchSize / 2
        val cy = h / 2 - patchSize / 2

        for (extra in extras) {
            if (extra.width != w || extra.height != h) continue
            val extPx = IntArray(w * h)
            extra.getPixels(extPx, 0, w, 0, 0, w, h)

            // Cerca miglior allineamento entro +-10 pixel
            var bestDx = 0
            var bestDy = 0
            var bestSad = Long.MAX_VALUE

            for (dy in -10..10 step 2) {
                for (dx in -10..10 step 2) {
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

            // Fusione ghost-free (scarta pixel che si sono mossi troppo rispetto al frame base)
            for (y in 10 until h - 10) {
                val rowB = y * w
                val rowE = (y + bestDy) * w
                for (x in 10 until w - 10) {
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

                    val diff = (abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3
                    if (diff < 26) {
                        accR[idxB] += er
                        accG[idxB] += eg
                        accB[idxB] += eb
                        accW[idxB] += 1
                    }
                }
            }
        }

        for (i in basePx.indices) {
            val weight = accW[i]
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
     * Iniezione Residuale Neurale:
     * Calcola la mappa neurale tramite TFLite e ne estrae SOLO la componente di dettaglio ricostruita
     * (High-Frequency Residual = OutputNeurale - InputDownscaledUpscaled), sommandola ai pixel
     * nativi ad alta risoluzione. In questo modo l'immagine non viene MAI rimpicciolita o impastata,
     * ma riceve solo il recupero strutturale della rete neurale.
     */
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

        val inH = if (inShape.size == 4 && inShape[1] > 0) inShape[1] else 50
        val inW = if (inShape.size == 4 && inShape[2] > 0) inShape[2] else 50
        val outH = if (outShape.size == 4 && outShape[1] > 0) outShape[1] else inH * 4
        val outW = if (outShape.size == 4 && outShape[2] > 0) outShape[2] else inW * 4

        val workMaxSide = if (isZoomSpinto) 2200 else 2800
        val workBase = scaleToMaxSide(nativeSharpSource, workMaxSide)

        // Costruiamo due mappe:
        // 1) neuralMap: uscita della rete neurale
        // 2) baselineMap: stesso ritaglio passato attraverso il solo downscale/upscale lineare
        // La differenza (neuralMap - baselineMap) è il puro dettaglio aggiunto dalla rete neurale!
        val neuralMap = Bitmap.createBitmap(workBase.width, workBase.height, Bitmap.Config.ARGB_8888)
        val baselineMap = Bitmap.createBitmap(workBase.width, workBase.height, Bitmap.Config.ARGB_8888)
        val neuralCanvas = Canvas(neuralMap)
        val baseCanvas = Canvas(baselineMap)

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
                scaledIn.getPixels(tilePixels, 0, inW, 0, 0, inW, inH)

                inputBuffer.rewind()
                if (inType == DataType.UINT8) {
                    for (p in tilePixels) {
                        inputBuffer.put(((p ushr 16) and 0xFF).toByte())
                        inputBuffer.put(((p ushr 8) and 0xFF).toByte())
                        inputBuffer.put((p and 0xFF).toByte())
                    }
                } else {
                    val scaleIn = if (inH >= 64) 1f / 255f else 1f
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
                    val scaleOut = if (inH >= 64) 255f else 1f
                    for (i in outPixels.indices) {
                        val red = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        val green = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        val blue = (outputBuffer.float * scaleOut).toInt().coerceIn(0, 255)
                        outPixels[i] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                    }
                }

                val srPatch = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                srPatch.setPixels(outPixels, 0, outW, 0, 0, outW, outH)

                val dstRect = Rect(left, top, right, bottom)
                neuralCanvas.drawBitmap(srPatch, null, dstRect, paint)
                baseCanvas.drawBitmap(scaledIn, null, dstRect, paint)
            }
        }

        val fullNeural = if (neuralMap.width != nativeSharpSource.width || neuralMap.height != nativeSharpSource.height) {
            Bitmap.createScaledBitmap(neuralMap, nativeSharpSource.width, nativeSharpSource.height, true)
        } else {
            neuralMap
        }
        val fullBaseline = if (baselineMap.width != nativeSharpSource.width || baselineMap.height != nativeSharpSource.height) {
            Bitmap.createScaledBitmap(baselineMap, nativeSharpSource.width, nativeSharpSource.height, true)
        } else {
            baselineMap
        }

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
                // Foto normali 1x: discreto ma nitido
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

            // Estrae solo il dettaglio strutturale creato dalla rete neurale e lo applica ai pixel originali
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

    /**
     * Integrazione Cloud AI opzionale: invia la regione centrale dello zoom spinto
     * a un endpoint di Super-Resolution generativa (es. Swin2SR / Real-ESRGAN Cloud)
     * e fonde il risultato ricostruito nel fotogramma.
     */
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

            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
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
                    return result
                }
            }
            null
        } catch (e: Exception) {
            Log.w("NeuralSR", "Cloud AI non raggiungibile o non configurata: ${e.message}")
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
        if (w * h > 24_000_000) return bitmap

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
