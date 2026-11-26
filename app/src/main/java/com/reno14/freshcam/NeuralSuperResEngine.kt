package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class NeuralProcessResult(
    val bitmap: Bitmap,
    val modeApplied: String,
    val backendUsed: String? = null
)

/**
 * Motore Fotografico v18.0:
 * - Vera AI Generativa Locale tramite OnnxGenerativeEngine (ClearReality SPAN 4x / Real-ESRGAN v3 4x).
 * - Multi-Frame ombra-pulito come base quando RAW-in-RAM è spento.
 * - Solo Cloud Gratuito opzionale (Hugging Face gratis con token hf_... o URL personalizzato gratuito).
 *   Rimossa del tutto la parte Cloud a pagamento.
 */
class NeuralSuperResEngine(
    @Suppress("UNUSED_PARAMETER") private val context: Context,
    private val onnxEngine: OnnxGenerativeEngine
) {

    fun processBurst(
        frames: List<Bitmap>,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        aiPreset: UnifiedAiPreset,
        localNeuralModel: LocalNeuralModel,
        preferNnapiNpu: Boolean,
        activeHardwareExtension: String?,
        activeLens: LensType,
        displayZoom: Float,
        digitalExtraZoom: Float,
        cloudEndpointUrl: String,
        cloudApiKey: String,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        val croppedFrames = if (digitalExtraZoom > 1.03f) {
            frames.map { cropOpticalCenter(it, digitalExtraZoom, keepNativeSize = aiPreset.isEnabled && !aiPreset.isCloud) }
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
            if (postSavePolish) {
                val toned = applySubtleFilmicCurve(working, strength = 0.22f)
                if (toned !== working) working.recycle()
                working = toned
            }
            if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = if (isTeleLens) "Tele (AI Off)" else "Ottica Pura (AI Off)"
            )
        }

        val level = aiPreset.level
        var burstUsed = false

        // 1. Fusione Multi-Frame di base prima della rete neurale (abbatte il rumore casuale)
        if (activeHardwareExtension == null && croppedFrames.size > 1) {
            try {
                val gradedExtras = croppedFrames.drop(1).map {
                    if (profile == CameraProfile.NATURAL && abs(brightnessBoost) < 0.005f) {
                        it.copy(Bitmap.Config.ARGB_8888, false)
                    } else {
                        profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                    }
                }
                val merged = alignAndMergeBurstShadowClean(working, gradedExtras)
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

        val parts = mutableListOf<String>()
        if (activeHardwareExtension != null) parts.add(activeHardwareExtension)
        if (burstUsed) parts.add("Burst ${frames.size}x")
        var usedBackend: String? = null

        // 2. VERA AI GENERATIVA LOCALE OFFLINE (ONNX Runtime: ClearReality SPAN 4x o Real-ESRGAN v3 4x)
        if (!aiPreset.isCloud) {
            val aiOutcome = onnxEngine.enhancePhoto(
                source = working,
                model = localNeuralModel,
                level = level,
                displayZoom = displayZoom,
                preferNnapi = preferNnapiNpu
            )
            usedBackend = aiOutcome.backendUsed
            if (aiOutcome.error != null) {
                parts.add("AI ERR: ${aiOutcome.error}")
            } else {
                if (aiOutcome.bitmap !== working && !working.isRecycled) working.recycle()
                working = aiOutcome.bitmap
                parts.add(aiOutcome.label)
            }
        } else {
            // 3. CLOUD GRATUITO OPZIONALE (Hugging Face Free Token o Endpoint Gratuito personalizzato)
            val cloudAlpha = if (level >= 3) 1.0f else 0.78f
            val (cloudBmp, statusMsg) = tryFreeCloudSuperRes(
                source = working,
                endpointUrl = cloudEndpointUrl,
                apiKey = cloudApiKey,
                blendStrength = cloudAlpha
            )
            if (cloudBmp != null) {
                if (cloudBmp !== working && !working.isRecycled) working.recycle()
                working = cloudBmp
            }
            parts.add(statusMsg)
        }

        if (postSavePolish) {
            val toned = try { applySubtleFilmicCurve(working, strength = 0.22f) } catch (_: Throwable) { working }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" · "),
            backendUsed = usedBackend
        )
    }

    fun applyFreeCloudToBitmap(
        source: Bitmap,
        level: Int,
        cloudEndpointUrl: String,
        cloudApiKey: String
    ): Pair<Bitmap, String> {
        val cloudAlpha = if (level >= 3) 1.0f else 0.78f
        val (cloudBmp, statusMsg) = tryFreeCloudSuperRes(source, cloudEndpointUrl, cloudApiKey, cloudAlpha)
        return if (cloudBmp != null) {
            if (cloudBmp !== source && !source.isRecycled) source.recycle()
            cloudBmp to statusMsg
        } else {
            source to statusMsg
        }
    }

    private fun tryFreeCloudSuperRes(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        blendStrength: Float
    ): Pair<Bitmap?, String> {
        if (endpointUrl.isBlank()) return null to "Cloud Gratis: URL mancante nelle Impostazioni"
        if (endpointUrl.contains("huggingface.co") && apiKey.isBlank()) {
            return null to "Cloud Gratis: inserisci Token gratuito HF (hf_...)"
        }
        return try {
            val uploadBmp = scaleToMaxSide(source, 768)
            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 92, baos)
            if (uploadBmp !== source) uploadBmp.recycle()

            val conn = (URL(endpointUrl.trim()).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 10000
                readTimeout = 30000
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
                    ?: return null to "Cloud Gratis: risposta non immagine"
                val targetW = max(source.width, srBmp.width).coerceAtMost(4096)
                val targetH = max(source.height, srBmp.height).coerceAtMost(3072)
                val scaledSr = Bitmap.createScaledBitmap(srBmp, targetW, targetH, true)
                srBmp.recycle()
                if (blendStrength >= 0.95f) {
                    scaledSr to "Cloud Gratis OK"
                } else {
                    val base = Bitmap.createScaledBitmap(source, targetW, targetH, true)
                    val canvas = Canvas(base)
                    val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                        alpha = (blendStrength * 255).toInt().coerceIn(80, 255)
                    }
                    canvas.drawBitmap(scaledSr, 0f, 0f, paint)
                    scaledSr.recycle()
                    base to "Cloud Gratis OK"
                }
            } else {
                val errHint = when (code) {
                    401, 403 -> "Token HF non valido"
                    402 -> "Quota gratuita HF esaurita"
                    404 -> "Endpoint non trovato"
                    503 -> "Server HF in avvio (riprova)"
                    else -> "HTTP $code"
                }
                null to "Cloud Gratis: $errHint"
            }
        } catch (e: Throwable) {
            null to "Cloud Gratis: errore rete/timeout"
        }
    }

    private fun alignAndMergeBurstShadowClean(base: Bitmap, extras: List<Bitmap>): Bitmap {
        val w = base.width
        val h = base.height
        if (w < 64 || h < 64) return base
        val total = w * h
        val basePx = IntArray(total)
        base.getPixels(basePx, 0, w, 0, 0, w, h)

        val accR = IntArray(total)
        val accG = IntArray(total)
        val accB = IntArray(total)
        val accW = ByteArray(total)

        for (i in 0 until total) {
            val c = basePx[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            val yC = (77 * r + 150 * g + 29 * b) shr 8
            val initW = if (yC < 95) 1 else 2
            accR[i] = r * initW
            accG[i] = g * initW
            accB[i] = b * initW
            accW[i] = initW.toByte()
        }

        val margin = 12
        val patchSize = min(140, min(w, h) / 3).coerceAtMost(min(w, h) - margin * 2 - 4)
        if (patchSize < 24) return base
        val cx = (w - patchSize) / 2
        val cy = (h - patchSize) / 2
        val extPx = IntArray(total)
        var mergedAny = false

        fun computeSad(dx: Int, dy: Int, step: Int): Pair<Long, Int> {
            var sad = 0L
            var cnt = 0
            var py = 0
            while (py < patchSize) {
                val rowBase = (cy + py) * w
                val rowExt = (cy + py + dy) * w
                var px = 0
                while (px < patchSize) {
                    val bVal = basePx[rowBase + cx + px]
                    val eVal = extPx[rowExt + cx + px + dx]
                    sad += abs(((bVal ushr 8) and 0xFF) - ((eVal ushr 8) and 0xFF))
                    cnt++
                    px += step
                }
                py += step
            }
            return sad to cnt
        }

        for (extra in extras) {
            if (extra.width != w || extra.height != h) continue
            extra.getPixels(extPx, 0, w, 0, 0, w, h)

            var bestDx = 0
            var bestDy = 0
            var bestSad = Long.MAX_VALUE
            var samples = 0

            for (dy in -8..8 step 2) {
                for (dx in -8..8 step 2) {
                    val (sad, cnt) = computeSad(dx, dy, 4)
                    if (sad < bestSad) {
                        bestSad = sad
                        bestDx = dx
                        bestDy = dy
                        samples = cnt
                    }
                }
            }

            val coarseDx = bestDx
            val coarseDy = bestDy
            bestSad = Long.MAX_VALUE
            for (dy in (coarseDy - 1)..(coarseDy + 1)) {
                for (dx in (coarseDx - 1)..(coarseDx + 1)) {
                    if (abs(dx) > 10 || abs(dy) > 10) continue
                    val (sad, cnt) = computeSad(dx, dy, 3)
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

            for (y in margin until h - margin) {
                val rowB = y * w
                val rowE = (y + bestDy) * w
                for (x in margin until w - margin) {
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

                    val yB = (77 * br + 150 * bg + 29 * bb) shr 8
                    val diffLimit = if (yB < 95) 18 else 13
                    if ((abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3 < diffLimit) {
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

    private fun cropOpticalCenter(bmp: Bitmap, factor: Float, keepNativeSize: Boolean): Bitmap {
        val f = factor.coerceIn(1.0f, 4.0f)
        val newW = (bmp.width / f).toInt().coerceAtLeast(160)
        val newH = (bmp.height / f).toInt().coerceAtLeast(160)
        val sx = (bmp.width - newW) / 2
        val sy = (bmp.height - newH) / 2
        val crop = Bitmap.createBitmap(bmp, sx, sy, newW, newH)
        if (keepNativeSize) return crop
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
