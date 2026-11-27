package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
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
 * Motore AI Locale Ibrido v19 — Veloce, Reale e Senza Perdita di Risoluzione:
 * 1. Su 1x, 0.6x e 3.5x (fino a 6.5x): NON ridimensiona MAI la foto a bassa risoluzione!
 *    Applica in ~0.4-0.9s a piena risoluzione 12MP un algoritmo fotografico completo:
 *    - Mappa di illuminazione multi-scala (Local HDR / Retinex) per aprire le ombre scure
 *      e recuperare le luci forti senza aloni.
 *    - Pulizia croma/ombre adattiva + Micro-contrasto & Nitidezza multi-scala (solo luminanza,
 *      protetta sulle ombre per zero grana aggiunta).
 *    - Vibrance intelligente (esalta i colori spenti proteggendo incarnati e toni già saturi).
 * 2. Su Zoom Spinto (>= 7x): consegna i pixel ottici puri alla rete neurale SPAN 4x (ONNX)
 *    per ricostruire dettagli e bordi dove il sensore ha pochi pixel, più veloce 3x rispetto a v18.
 * 3. Cloud Gratuito opzionale (Hugging Face con token hf_... o URL personalizzato).
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
        val t0 = SystemClock.elapsedRealtime()
        val isExtremeZoom = displayZoom >= 6.5f

        val croppedFrames = if (digitalExtraZoom > 1.03f) {
            frames.map { cropOpticalCenter(it, digitalExtraZoom, keepNativeSize = aiPreset.isEnabled && !aiPreset.isCloud && isExtremeZoom) }
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

        // Fusione Multi-Frame opzionale quando sono stati catturati più fotogrammi (Media/Alta)
        if (activeHardwareExtension == null && croppedFrames.size > 1) {
            try {
                val gradedExtras = croppedFrames.drop(1).map {
                    if (profile == CameraProfile.NATURAL && abs(brightnessBoost) < 0.005f) {
                        it.copy(Bitmap.Config.ARGB_8888, false)
                    } else {
                        profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                    }
                }
                val merged = alignAndMergeBurstFast(working, gradedExtras)
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
        if (burstUsed) parts.add("MF ${frames.size}x")
        var usedBackend: String? = null

        if (!aiPreset.isCloud) {
            if (isExtremeZoom) {
                // ZOOM SPINTO (>= 7x): Rete Neurale SPAN 4x ONNX sul ritaglio ottico reale + rifinitura HDR
                val aiOutcome = onnxEngine.enhanceZoomedCrop(
                    source = working,
                    model = localNeuralModel,
                    level = level,
                    preferNnapi = preferNnapiNpu
                )
                usedBackend = aiOutcome.backendUsed
                if (aiOutcome.error != null) {
                    parts.add("SPAN ERR: ${aiOutcome.error}")
                } else {
                    if (aiOutcome.bitmap !== working && !working.isRecycled) working.recycle()
                    working = aiOutcome.bitmap
                    parts.add(aiOutcome.label)
                }
                val polishedZoom = enhanceFullResPhoto(working, level = 1, isTele = true)
                if (polishedZoom !== working && !working.isRecycled) working.recycle()
                working = polishedZoom
            } else {
                // FOTO NORMALI (0.6x / 1x / 3.5x): Algoritmo Full-Res HDR + Pulizia Ombre + Nitidezza Multi-Scala
                // Mantiene il 100% della nitidezza ottica 12MP con differenza netta e salvataggio rapidissimo!
                val enhanced = enhanceFullResPhoto(working, level = level, isTele = isTeleLens)
                if (enhanced !== working && !working.isRecycled) working.recycle()
                working = enhanced
                val secStr = String.format(Locale.US, "%.1fs", (SystemClock.elapsedRealtime() - t0) / 1000.0)
                parts.add("AI Full-Res ${aiPreset.shortBadge.removePrefix("AI ")} ($secStr)")
            }
        } else {
            // CLOUD GRATUITO OPZIONALE
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

    /**
     * Motore Fotografico Full-Res v19 (single-pass ottimizzato ~0.4-0.9s su 12MP):
     * 1. Mappa di illuminazione a larga scala (Local HDR / Retinex): apre le ombre chiuse e
     *    recupera le alte luci senza schiacciare il contrasto locale.
     * 2. Micro-contrasto a media scala (Clarity) + Dettaglio fine (Unsharp adattivo su luminanza)
     *    con blocco automatico sulle ombre scure per non amplificare mai la grana.
     * 3. Denoise selettivo su ombre e crominanza + Vibrance intelligente (protegge gli incarnati).
     */
    private fun enhanceFullResPhoto(source: Bitmap, level: Int, isTele: Boolean): Bitmap {
        val w = source.width
        val h = source.height
        val total = w * h
        val src = IntArray(total)
        val dst = IntArray(total)
        source.getPixels(src, 0, w, 0, 0, w, h)

        // Mappa di illuminazione a larga scala (raggio ampio per HDR locale senza aloni)
        val illumSmallW = max(8, w / 28)
        val illumSmallH = max(8, h / 28)
        val illumTiny = Bitmap.createScaledBitmap(source, illumSmallW, illumSmallH, true)
        val illumFull = Bitmap.createScaledBitmap(illumTiny, w, h, true)
        illumTiny.recycle()
        val illumPx = IntArray(total)
        illumFull.getPixels(illumPx, 0, w, 0, 0, w, h)
        illumFull.recycle()

        // Mappa a media scala per Clarity / struttura fotografica
        val medSmallW = max(16, w / 6)
        val medSmallH = max(16, h / 6)
        val medTiny = Bitmap.createScaledBitmap(source, medSmallW, medSmallH, true)
        val medFull = Bitmap.createScaledBitmap(medTiny, w, h, true)
        medTiny.recycle()
        val medPx = IntArray(total)
        medFull.getPixels(medPx, 0, w, 0, 0, w, h)
        medFull.recycle()

        // Parametri calibrati per rendere il miglioramento chiaramente visibile ma naturale
        val shadowLiftStrength = when (level) {
            1 -> 0.24f
            2 -> 0.40f
            else -> 0.56f
        }
        val highlightRecover = when (level) {
            1 -> 0.14f
            2 -> 0.24f
            else -> 0.34f
        }
        val clarityGain = when (level) {
            1 -> if (isTele) 0.24f else 0.18f
            2 -> if (isTele) 0.36f else 0.28f
            else -> if (isTele) 0.48f else 0.38f
        }
        val fineSharpGain = when (level) {
            1 -> if (isTele) 0.40f else 0.32f
            2 -> if (isTele) 0.58f else 0.46f
            else -> if (isTele) 0.75f else 0.60f
        }
        val vibranceBoost = when (level) {
            1 -> 0.12f
            2 -> 0.20f
            else -> 0.28f
        }
        val contrastCurve = when (level) {
            1 -> 0.045f
            2 -> 0.075f
            else -> 0.105f
        }
        val noiseGate = if (isTele) 3.5f else 3.0f

        // Pre-calcola LUT S-curve morbida per profondità dei neri e micro-contrasto tonale
        val sCurveDelta = FloatArray(256) { v ->
            val n = v / 255f
            val s = -contrastCurve * sin(2.0 * Math.PI * n).toFloat()
            s * 255f
        }

        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val idx = row + x
                val c = src[idx]
                var r = (c ushr 16) and 0xFF
                var g = (c ushr 8) and 0xFF
                var b = c and 0xFF

                val yC = 0.299f * r + 0.587f * g + 0.114f * b

                // 1. Denoise bilaterale veloce sulle ombre scure (Y < 72) prima di alzarle!
                if (yC < 72f) {
                    val n1 = src[idx - w]
                    val n2 = src[idx + w]
                    val n3 = src[idx - 1]
                    val n4 = src[idx + 1]
                    val n5 = src[idx - w - 1]
                    val n6 = src[idx + w + 1]
                    var sr = r * 2
                    var sg = g * 2
                    var sb = b * 2
                    var sw = 2
                    val thresh = 15
                    val neighbors = intArrayOf(n1, n2, n3, n4, n5, n6)
                    for (nb in neighbors) {
                        val nr = (nb ushr 16) and 0xFF
                        val ng = (nb ushr 8) and 0xFF
                        val nbv = nb and 0xFF
                        val yn = (77 * nr + 150 * ng + 29 * nbv) shr 8
                        if (abs(yC.toInt() - yn) <= thresh) {
                            sr += nr
                            sg += ng
                            sb += nbv
                            sw++
                        }
                    }
                    val denoiseBlend = ((72f - yC) / 56f).coerceIn(0f, 0.78f)
                    r = (r + (sr / sw - r) * denoiseBlend).toInt()
                    g = (g + (sg / sw - g) * denoiseBlend).toInt()
                    b = (b + (sb / sw - b) * denoiseBlend).toInt()
                }

                val yClean = 0.299f * r + 0.587f * g + 0.114f * b

                // 2. Local HDR Tone Mapping guidato dalla mappa di illuminazione a larga scala
                val ic = illumPx[idx]
                val yIllum = 0.299f * ((ic ushr 16) and 0xFF) + 0.587f * ((ic ushr 8) and 0xFF) + 0.114f * (ic and 0xFF)

                // Apre le ombre in aree realmente scure senza ingrigire i neri puri
                val shadowMask = ((118f - yIllum) / 118f).coerceIn(0f, 1f)
                val blackAnchor = (yClean / 28f).coerceIn(0f, 1f)
                val hdrShadowDelta = shadowMask * shadowMask * blackAnchor * shadowLiftStrength * 52f

                // Recupera le luci alte sovraesposte (finestre, cieli, lampade)
                val hiMask = ((yIllum - 172f) / 83f).coerceIn(0f, 1f)
                val hdrHiDelta = -hiMask * hiMask * highlightRecover * 42f

                // 3. Clarity a media scala (struttura e tridimensionalità)
                val mc = medPx[idx]
                val yMed = 0.299f * ((mc ushr 16) and 0xFF) + 0.587f * ((mc ushr 8) and 0xFF) + 0.114f * (mc and 0xFF)
                val medDetail = (yClean - yMed).coerceIn(-24f, 24f)

                // 4. Micro-nitidezza ottica fine (solo luminanza, bloccata sulle ombre rumorose e sugli estremi bianchi)
                var fineDetail = 0f
                if (yClean > 64f) {
                    val pN = src[idx - w]
                    val pS = src[idx + w]
                    val pE = src[idx + 1]
                    val pW = src[idx - 1]
                    val yN = 0.299f * ((pN ushr 16) and 0xFF) + 0.587f * ((pN ushr 8) and 0xFF) + 0.114f * (pN and 0xFF)
                    val yS = 0.299f * ((pS ushr 16) and 0xFF) + 0.587f * ((pS ushr 8) and 0xFF) + 0.114f * (pS and 0xFF)
                    val yE = 0.299f * ((pE ushr 16) and 0xFF) + 0.587f * ((pE ushr 8) and 0xFF) + 0.114f * (pE and 0xFF)
                    val yW = 0.299f * ((pW ushr 16) and 0xFF) + 0.587f * ((pW ushr 8) and 0xFF) + 0.114f * (pW and 0xFF)

                    val d = yClean - (yN + yS + yE + yW) * 0.25f
                    val absD = abs(d)
                    if (absD > noiseGate) {
                        val sign = if (d >= 0f) 1f else -1f
                        val ramp = ((yClean - 64f) / 36f).coerceIn(0f, 1f)
                        val hiGuard = if (yClean > 225f) ((255f - yClean) / 30f).coerceIn(0.15f, 1f) else 1f
                        fineDetail = sign * (absD - noiseGate).coerceAtMost(22f) * fineSharpGain * ramp * hiGuard
                    }
                }

                val clarityGuard = if (yClean < 55f) (yClean / 55f) else if (yClean > 230f) ((255f - yClean) / 25f).coerceIn(0.2f, 1f) else 1f
                val sDelta = sCurveDelta[yClean.toInt().coerceIn(0, 255)]
                val totalLumaDelta = hdrShadowDelta + hdrHiDelta + sDelta + (medDetail * clarityGain * clarityGuard) + fineDetail

                var fr = (r + totalLumaDelta).coerceIn(0f, 255f)
                var fg = (g + totalLumaDelta).coerceIn(0f, 255f)
                var fb = (b + totalLumaDelta).coerceIn(0f, 255f)

                // 5. Vibrance intelligente (ravviva colori spenti, evita saturazione eccessiva e protegge la pelle)
                val maxCh = max(fr, max(fg, fb))
                val minCh = min(fr, min(fg, fb))
                val sat = if (maxCh > 1f) (maxCh - minCh) / maxCh else 0f
                val avg = (fr + fg + fb) * 0.3333f
                // Protezione toni pelle (rosso dominante caldo)
                val isSkinLike = fr > fg && fg > fb && (fr - fb) > 15f && (fr - fg) < 70f
                val effVibrance = vibranceBoost * (1f - sat) * (if (isSkinLike) 0.45f else 1.0f)
                fr = (avg + (fr - avg) * (1f + effVibrance)).coerceIn(0f, 255f)
                fg = (avg + (fg - avg) * (1f + effVibrance)).coerceIn(0f, 255f)
                fb = (avg + (fb - avg) * (1f + effVibrance)).coerceIn(0f, 255f)

                dst[idx] = (0xFF shl 24) or (fr.toInt() shl 16) or (fg.toInt() shl 8) or fb.toInt()
            }
        }

        // Bordi 1px
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

    private fun alignAndMergeBurstFast(base: Bitmap, extras: List<Bitmap>): Bitmap {
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
        val patchSize = min(128, min(w, h) / 4).coerceAtMost(min(w, h) - margin * 2 - 4)
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

            if (samples == 0 || (bestSad / samples) > 14) continue
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
                    val diffLimit = if (yB < 95) 17 else 12
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
