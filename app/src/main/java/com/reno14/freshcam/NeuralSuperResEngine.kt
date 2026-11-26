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
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

data class NeuralProcessResult(
    val bitmap: Bitmap,
    val modeApplied: String
)

/**
 * Motore Fotografico v15.0:
 * - Teleobiettivo 3.5x-120x (S5KJN5): Burst multi-frame 3-4 scatti ATTIVO GIA' DA 3.5x
 *   per colmare il divario di luce e definizione rispetto alla principale IMX882.
 * - Principale 1x: sfrutta HW Notte (unica estensione multi-frame reale esposta da ColorOS)
 *   oppure Burst multi-frame pulito su AI Media/Alta.
 * - Micro-contrasto a doppia scala pulito (senza il bug del 1/24 che lavava le ombre).
 * - Nuova Cloud AI: supporta Replicate Real-ESRGAN (ricostruzione generativa vera) con
 *   fusione completa o radiale e diagnostica chiara nel banner.
 */
class NeuralSuperResEngine(@Suppress("UNUSED_PARAMETER") private val context: Context) {

    fun processBurst(
        frames: List<Bitmap>,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiPreset: UnifiedAiPreset,
        activeHardwareExtension: String?,
        activeLens: LensType,
        displayZoom: Float,
        digitalExtraZoom: Float,
        cloudModelPreset: CloudModelPreset,
        cloudEndpointUrl: String,
        cloudApiKey: String,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        val croppedFrames = if (digitalExtraZoom > 1.03f) {
            frames.map { applyDigitalCrop(it, digitalExtraZoom) }
        } else {
            frames
        }

        val primaryFrame = croppedFrames.first()
        var working = if (profile == CameraProfile.NATURAL && abs(brightnessBoost) < 0.005f) {
            primaryFrame.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            profile.applyToBitmap(
                source = primaryFrame,
                strength = profileIntensity,
                brightnessBoost = brightnessBoost
            )
        }

        val isTeleLens = activeLens == LensType.TELE_50MP || displayZoom >= 3.45f

        if (!aiPreset.isEnabled) {
            if (isTeleLens) {
                val teleOpt = applyDualScaleOpticalClarity(working, fineGain = 0.20f, medGain = 0.06f, noiseGate = 4.0f)
                if (teleOpt !== working) working.recycle()
                working = teleOpt
            }
            if (postSavePolish) {
                val toned = applySubtleFilmicCurve(working, strength = 0.22f)
                if (toned !== working) working.recycle()
                working = toned
            }
            if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = if (isTeleLens) "Tele Ottico (AI Off)" else "Ottica Pura (AI Off)"
            )
        }

        val level = aiPreset.level

        // 1. Burst Multi-Frame Anti-Ghosting: attivo sia su Tele (>= 3.5x) sia su 1x quando non c'è HW Extension
        var burstUsed = false
        if (activeHardwareExtension == null && croppedFrames.size > 1) {
            try {
                val gradedExtras = croppedFrames.drop(1).map {
                    if (profile == CameraProfile.NATURAL && abs(brightnessBoost) < 0.005f) {
                        it.copy(Bitmap.Config.ARGB_8888, false)
                    } else {
                        profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                    }
                }
                val merged = alignAndMergeBurstFixed(working, gradedExtras)
                gradedExtras.forEach { if (!it.isRecycled) it.recycle() }
                if (merged !== working) {
                    working.recycle()
                    working = merged
                    burstUsed = true
                }
            } catch (t: Throwable) {
                Log.e("PhotoEngine", "Burst fallback", t)
            }
        }
        if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }

        // 2. Chroma Denoise (pulisce rumore colore del sensore tele JN5 senza toccare la nitidezza)
        if (activeHardwareExtension == null && (isTeleLens || level >= 2)) {
            val cleanChroma = try { fastChromaDenoise(working) } catch (_: Throwable) { working }
            if (cleanChroma !== working && !working.isRecycled) working.recycle()
            working = cleanChroma
        }

        // 3. Micro-Contrasto a Doppia Scala (tarato più deciso sul Tele 3.5x+ per superare la morbidezza dell'ottica f/2.8)
        if (activeHardwareExtension == null) {
            val teleBonus = if (isTeleLens) 0.14f else 0.0f
            val fineGain = when (level) {
                1 -> 0.22f + teleBonus
                2 -> 0.34f + teleBonus
                else -> 0.46f + teleBonus
            }
            val medGain = when (level) {
                1 -> if (isTeleLens) 0.10f else 0.06f
                2 -> if (isTeleLens) 0.16f else 0.10f
                else -> if (isTeleLens) 0.22f else 0.14f
            }
            val noiseGate = if (displayZoom >= 10f) 4.5f else 3.2f
            val sharpened = try {
                applyDualScaleOpticalClarity(working, fineGain = fineGain, medGain = medGain, noiseGate = noiseGate)
            } catch (_: Throwable) {
                working
            }
            if (sharpened !== working && !working.isRecycled) working.recycle()
            working = sharpened
        } else if (level >= 3) {
            val refined = applyDualScaleOpticalClarity(working, fineGain = 0.16f, medGain = 0.06f, noiseGate = 4.0f)
            if (refined !== working) working.recycle()
            working = refined
        }

        // 4. Nuova Cloud AI Generativa (Replicate Real-ESRGAN o Hugging Face)
        var cloudStatus = ""
        if (aiPreset.isCloud) {
            val cloudAlpha = when (level) {
                1 -> 0.55f
                2 -> 0.80f
                else -> 1.0f
            }
            val (cloudBmp, statusMsg) = if (cloudModelPreset.isReplicate || cloudEndpointUrl.contains("replicate.com")) {
                tryReplicateRealEsrgan(
                    source = working,
                    endpointUrl = cloudEndpointUrl,
                    apiKey = cloudApiKey,
                    blendStrength = cloudAlpha
                )
            } else {
                tryHuggingFaceSuperRes(
                    source = working,
                    endpointUrl = cloudEndpointUrl,
                    apiKey = cloudApiKey,
                    blendStrength = cloudAlpha
                )
            }
            cloudStatus = statusMsg
            if (cloudBmp != null) {
                if (cloudBmp !== working) working.recycle()
                working = cloudBmp
            }
        }

        if (postSavePolish) {
            val toned = try { applySubtleFilmicCurve(working, strength = 0.24f) } catch (_: Throwable) { working }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        val parts = mutableListOf<String>()
        if (activeHardwareExtension != null) parts.add(activeHardwareExtension)
        if (burstUsed) parts.add("Burst ${frames.size}x")
        if (activeHardwareExtension == null) {
            parts.add(if (isTeleLens) "Tele ${aiPreset.shortBadge}" else aiPreset.shortBadge)
        }
        if (cloudStatus.isNotEmpty()) parts.add(cloudStatus)

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" · ")
        )
    }

    /**
     * Doppia scala pulita su sola luminanza:
     * - Scala fine (3x3 Laplacian): incide micro-dettagli con Noise-Gate.
     * - Scala media (1/4): aggiunge micro-contrasto locale (Clarity) senza aloni né ombre lavate.
     */
    private fun applyDualScaleOpticalClarity(
        source: Bitmap,
        fineGain: Float,
        medGain: Float,
        noiseGate: Float
    ): Bitmap {
        val w = source.width
        val h = source.height
        val src = IntArray(w * h)
        val dst = IntArray(w * h)
        source.getPixels(src, 0, w, 0, 0, w, h)

        val medSmall = Bitmap.createScaledBitmap(source, max(2, w / 4), max(2, h / 4), true)
        val medBlur = Bitmap.createScaledBitmap(medSmall, w, h, true)
        medSmall.recycle()
        val medPx = IntArray(w * h)
        medBlur.getPixels(medPx, 0, w, 0, 0, w, h)
        medBlur.recycle()

        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val idx = row + x
                val c = src[idx]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val yC = 0.299f * r + 0.587f * g + 0.114f * b

                val n = src[idx - w]
                val s = src[idx + w]
                val e = src[idx + 1]
                val west = src[idx - 1]

                val yN = 0.299f * ((n ushr 16) and 0xFF) + 0.587f * ((n ushr 8) and 0xFF) + 0.114f * (n and 0xFF)
                val yS = 0.299f * ((s ushr 16) and 0xFF) + 0.587f * ((s ushr 8) and 0xFF) + 0.114f * (s and 0xFF)
                val yE = 0.299f * ((e ushr 16) and 0xFF) + 0.587f * ((e ushr 8) and 0xFF) + 0.114f * (e and 0xFF)
                val yW = 0.299f * ((west ushr 16) and 0xFF) + 0.587f * ((west ushr 8) and 0xFF) + 0.114f * (west and 0xFF)

                val yAvg = (yN + yS + yE + yW) * 0.25f
                val fineD = yC - yAvg
                val absFine = abs(fineD)

                val gatedFine = if (absFine <= noiseGate) {
                    0f
                } else {
                    val sign = if (fineD >= 0f) 1f else -1f
                    sign * (absFine - noiseGate).coerceAtMost(22f)
                }

                val mc = medPx[idx]
                val yMed = 0.299f * ((mc ushr 16) and 0xFF) + 0.587f * ((mc ushr 8) and 0xFF) + 0.114f * (mc and 0xFF)
                val medD = (yC - yMed).coerceIn(-18f, 18f)

                val haloGuard = when {
                    yC > 224f -> ((255f - yC) / 31f).coerceIn(0.15f, 1.0f)
                    yC < 16f -> (yC / 16f).coerceIn(0.20f, 1.0f)
                    else -> 1.0f
                }

                val delta = (gatedFine * fineGain + medD * medGain) * haloGuard
                val nr = (r + delta).toInt().coerceIn(0, 255)
                val ng = (g + delta).toInt().coerceIn(0, 255)
                val nb = (b + delta).toInt().coerceIn(0, 255)
                dst[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }

        for (x in 0 until w) {
            dst[x] = src[x]
            dst[(h - 1) * w + x] = src[(h - 1) * w + x]
        }
        for (y in 0 until h) {
            dst[y * w] = src[y * w]
            dst[y * w + (w - 1)] = src[y * w + (w - 1)]
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * Replicate Real-ESRGAN API (sincrono con `Prefer: wait=30`):
     * Invia l'intera foto ridimensionata a 1024px (per restare veloce), ottiene la ricostruzione
     * generativa completa x2 da Real-ESRGAN e la fonde in modo pulito sull'originale.
     */
    private fun tryReplicateRealEsrgan(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        blendStrength: Float
    ): Pair<Bitmap?, String> {
        if (apiKey.isBlank()) return null to "Cloud: inserisci Token Replicate"
        return try {
            val uploadBmp = scaleToMaxSide(source, 1024)
            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
            if (uploadBmp !== source) uploadBmp.recycle()
            val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

            val inputObj = JSONObject().apply {
                put("image", "data:image/jpeg;base64,$b64")
                put("scale", 2)
                put("face_enhance", false)
            }
            val bodyJson = JSONObject().apply {
                put("input", inputObj)
            }.toString()

            val conn = (URL(endpointUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 10000
                readTimeout = 40000
                doOutput = true
                setRequestProperty("Authorization", "Bearer ${apiKey.trim()}")
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Prefer", "wait=30")
            }

            conn.outputStream.use { it.write(bodyJson.toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            if (code !in 200..299) {
                val hint = when (code) {
                    401, 403 -> "Token Replicate non valido"
                    402 -> "Credito Replicate esaurito"
                    else -> "Replicate HTTP $code"
                }
                return null to "Cloud: $hint"
            }

            val respStr = conn.inputStream.use { String(it.readBytes(), Charsets.UTF_8) }
            val obj = JSONObject(respStr)
            var outUrl = obj.optString("output", "")
            val getUrl = obj.optJSONObject("urls")?.optString("get", "") ?: ""

            // Se non è finito nei 30s di wait, fa polling veloce fino a 15s
            var attempts = 0
            while (outUrl.isBlank() && getUrl.isNotBlank() && attempts < 8) {
                Thread.sleep(2000)
                attempts++
                val pollConn = (URL(getUrl).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 8000
                    readTimeout = 10000
                    setRequestProperty("Authorization", "Bearer ${apiKey.trim()}")
                }
                if (pollConn.responseCode in 200..299) {
                    val pObj = JSONObject(pollConn.inputStream.use { String(it.readBytes(), Charsets.UTF_8) })
                    val status = pObj.optString("status")
                    if (status == "succeeded") {
                        outUrl = pObj.optString("output", "")
                        break
                    } else if (status == "failed" || status == "canceled") {
                        return null to "Cloud: elaborazione fallita"
                    }
                }
            }

            if (outUrl.isBlank()) return null to "Cloud: timeout Replicate"

            val imgConn = (URL(outUrl).openConnection() as HttpURLConnection).apply {
                connectTimeout = 10000
                readTimeout = 20000
            }
            val imgBytes = imgConn.inputStream.use { it.readBytes() }
            val srBmp = BitmapFactory.decodeByteArray(imgBytes, 0, imgBytes.size)
                ?: return null to "Cloud: decodifica fallita"

            val out = if (blendStrength >= 0.95f) {
                srBmp
            } else {
                val scaledSr = Bitmap.createScaledBitmap(srBmp, source.width, source.height, true)
                srBmp.recycle()
                val blended = source.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(blended)
                val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                    alpha = (blendStrength * 255).toInt().coerceIn(80, 255)
                }
                canvas.drawBitmap(scaledSr, 0f, 0f, paint)
                scaledSr.recycle()
                blended
            }
            out to "Cloud Real-ESRGAN OK"
        } catch (e: Throwable) {
            Log.w("PhotoEngine", "Errore Replicate Cloud: ${e.message}")
            null to "Cloud: errore rete"
        }
    }

    private fun tryHuggingFaceSuperRes(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        blendStrength: Float
    ): Pair<Bitmap?, String> {
        if (endpointUrl.isBlank()) return null to "Cloud: URL mancante"
        return try {
            val uploadBmp = scaleToMaxSide(source, 768)
            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
            if (uploadBmp !== source) uploadBmp.recycle()

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
            conn.outputStream.use { it.write(baos.toByteArray()) }
            val code = conn.responseCode
            if (code in 200..299) {
                val respBytes = conn.inputStream.use { it.readBytes() }
                val srBmp = BitmapFactory.decodeByteArray(respBytes, 0, respBytes.size)
                    ?: return null to "Cloud: risposta non valida"
                val scaledSr = Bitmap.createScaledBitmap(srBmp, source.width, source.height, true)
                srBmp.recycle()
                val blended = source.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(blended)
                val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                    alpha = (blendStrength * 255).toInt().coerceIn(80, 255)
                }
                canvas.drawBitmap(scaledSr, 0f, 0f, paint)
                scaledSr.recycle()
                blended to "Cloud HF OK"
            } else {
                val errHint = when (code) {
                    401, 403 -> "Token HF richiesto"
                    404 -> "Modello non trovato"
                    503 -> "Modello in avvio"
                    else -> "HTTP $code"
                }
                null to "Cloud: $errHint"
            }
        } catch (e: Throwable) {
            null to "Cloud: timeout/rete"
        }
    }

    private fun alignAndMergeBurstFixed(base: Bitmap, extras: List<Bitmap>): Bitmap {
        val w = base.width
        val h = base.height
        val total = w * h
        val basePx = IntArray(total)
        base.getPixels(basePx, 0, w, 0, 0, w, h)

        val accR = IntArray(total) { ((basePx[it] ushr 16) and 0xFF) * 2 }
        val accG = IntArray(total) { ((basePx[it] ushr 8) and 0xFF) * 2 }
        val accB = IntArray(total) { (basePx[it] and 0xFF) * 2 }
        val accW = ByteArray(total) { 2 }

        val patchSize = min(120, min(w, h) / 3)
        val cx = w / 2 - patchSize / 2
        val cy = h / 2 - patchSize / 2
        val extPx = IntArray(total)
        var mergedAny = false

        for (extra in extras) {
            if (extra.width != w || extra.height != h) continue
            extra.getPixels(extPx, 0, w, 0, 0, w, h)

            var bestDx = 0
            var bestDy = 0
            var bestSad = Long.MAX_VALUE
            var samples = 0

            for (dy in -8..8 step 2) {
                for (dx in -8..8 step 2) {
                    var sad = 0L
                    var cnt = 0
                    for (py in 0 until patchSize step 4) {
                        val rowBase = (cy + py) * w
                        val rowExt = (cy + py + dy) * w
                        for (px in 0 until patchSize step 4) {
                            val bVal = basePx[rowBase + cx + px]
                            val eVal = extPx[rowExt + cx + px + dx]
                            sad += abs(((bVal ushr 8) and 0xFF) - ((eVal ushr 8) and 0xFF))
                            cnt++
                        }
                    }
                    if (sad < bestSad) {
                        bestSad = sad
                        bestDx = dx
                        bestDy = dy
                        samples = cnt
                    }
                }
            }

            if (samples == 0 || (bestSad / samples) > 15) continue
            mergedAny = true

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

                    if ((abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3 < 15) {
                        accR[idxB] += er
                        accG[idxB] += eg
                        accB[idxB] += eb
                        accW[idxB] = (accW[idxB] + 1).toByte()
                    }
                }
            }
        }

        if (!mergedAny) return base

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

    private fun fastChromaDenoise(bitmap: Bitmap): Bitmap {
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
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val yC = (77 * r + 150 * g + 29 * b) shr 8

                var sumU = (b - yC) * 2
                var sumV = (r - yC) * 2
                var cnt = 2

                val neighbors = intArrayOf(src[idx - w], src[idx + w], src[idx - 1], src[idx + 1])
                for (n in neighbors) {
                    val nr = (n ushr 16) and 0xFF
                    val ng = (n ushr 8) and 0xFF
                    val nb = n and 0xFF
                    val yN = (77 * nr + 150 * ng + 29 * nb) shr 8
                    if (abs(yC - yN) < 22) {
                        sumU += (nb - yN)
                        sumV += (nr - yN)
                        cnt++
                    }
                }

                val outR = (yC + sumV / cnt).coerceIn(0, 255)
                val outB = (yC + sumU / cnt).coerceIn(0, 255)
                val outG = ((yC * 256 - 77 * outR - 29 * outB) / 150).coerceIn(0, 255)
                dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
        for (x in 0 until w) {
            dst[x] = src[x]
            dst[(h - 1) * w + x] = src[(h - 1) * w + x]
        }
        for (y in 0 until h) {
            dst[y * w] = src[y * w]
            dst[y * w + (w - 1)] = src[y * w + (w - 1)]
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
        return out
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

    private fun applySubtleFilmicCurve(bitmap: Bitmap, strength: Float): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val curve = 0.020 * strength
        val lut = IntArray(256)
        for (v in 0..255) {
            val norm = v / 255.0
            val sCurve = norm - curve * sin(2.0 * Math.PI * norm)
            val safe = if (sCurve > 0.89) 0.89 + (sCurve - 0.89) * 0.88 else sCurve
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
}
