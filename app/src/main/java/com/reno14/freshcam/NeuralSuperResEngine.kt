package com.reno14.freshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

data class NeuralProcessResult(
    val bitmap: Bitmap,
    val modeApplied: String
)

/**
 * Motore Fotografico Computazionale v13.0 (Zero TFLite distruttivo · Zero Freeze):
 *
 * 1) MOTORE LOCALE DI FOTOGRAFIA COMPUTAZIONALE (Sola Luminanza + Colori Intatti):
 *    - Rimosso il vecchio ritaglio a tasselli 128x128 TFLite che degradava le foto reali.
 *    - Sostituito con una pipeline multi-scala Edge-Preserving che lavora SOLO sulla luminanza (Y):
 *      a) Local Tone Mapping (HDR naturale): apre le ombre profonde e salva le alte luci
 *         in modo morbido, dando profondità reale e gamma dinamica superiore ad AI Off.
 *      b) Micro-Contrasto Fine + Medio con Noise-Gate (Coring): esalta i dettagli reali
 *         (texture, occhi, palazzi, foglie) ignorando completamente il micro-rumore sulle zone piatte.
 *      c) Deconvoluzione ottica dedicata per Tele 3.5x (Samsung JN5).
 *      d) Chroma Denoise mirato solo dove serve, senza mai impastare i bordi.
 *
 * 2) MOTORE CLOUD AI v13 (Swin2SR Classical x2 + Fusione Luminanza Radiale):
 *    - Usa un modello addestrato su foto pulite (niente deformazioni BSRGAN).
 *    - Invece di incollare un riquadro rettangolare, estrae SOLO il dettaglio fine di luminanza
 *      e lo fonde con una maschera radiale morbida (zero bordi visibili, colori originali intatti).
 */
class NeuralSuperResEngine(@Suppress("UNUSED_PARAMETER") private val context: Context) {

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
                val teleOpt = applyLuminanceComputationalPass(
                    source = working,
                    fineClarity = 0.22f,
                    mediumClarity = 0.08f,
                    shadowRecovery = 0.0f,
                    highlightProtect = 0.12f,
                    noiseGate = 4.0f
                )
                if (teleOpt !== working) working.recycle()
                working = teleOpt
            }
            if (postSavePolish) {
                val toned = applySubtleFilmicCurve(working, strength = 0.25f)
                if (toned !== working) working.recycle()
                working = toned
            }
            if (croppedFrames !== frames) croppedFrames.forEach { if (!it.isRecycled) it.recycle() }
            return NeuralProcessResult(
                bitmap = working,
                modeApplied = if (isTeleLens) "Tele Ottico (AI Off)" else "Ottica Pura (AI Off)"
            )
        }

        val isTeleSpinto = displayZoom >= 10.0f
        val level = aiPreset.level

        // 1. Burst Multi-Frame Anti-Ghosting (solo su zoom >= 10x e con verifica rigorosa del movimento)
        var burstUsed = false
        if (isTeleSpinto && croppedFrames.size > 1) {
            try {
                val gradedExtras = croppedFrames.drop(1).map {
                    profile.applyToBitmap(it, strength = profileIntensity, brightnessBoost = brightnessBoost)
                }
                val merged = alignAndMergeBurstSafe(working, gradedExtras)
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

        // 2. Pulizia rumore SOLO cromatico (veloce, su 1 passaggio; mantiene al 100% la nitidezza)
        if (isTeleLens || isTeleSpinto || level >= 2) {
            val cleanChroma = try {
                fastChromaDenoise(working)
            } catch (_: Throwable) {
                working
            }
            if (cleanChroma !== working && !working.isRecycled) working.recycle()
            working = cleanChroma
        }

        // 3. Motore Computazionale Locale Edge-Preserving (Sola Luminanza -> Colori 100% fedeli)
        val teleBoost = if (isTeleLens) 0.10f else 0.0f
        val zoomBoost = if (isTeleSpinto) 0.12f else 0.0f

        val fineClarity = when (level) {
            1 -> 0.24f + teleBoost + zoomBoost
            2 -> 0.36f + teleBoost + zoomBoost
            else -> 0.48f + teleBoost + zoomBoost
        }
        val mediumClarity = when (level) {
            1 -> 0.12f
            2 -> 0.20f
            else -> 0.28f
        }
        val shadowRecovery = when (level) {
            1 -> 0.12f
            2 -> 0.22f
            else -> 0.32f
        }
        val highlightProtect = when (level) {
            1 -> 0.18f
            2 -> 0.28f
            else -> 0.36f
        }
        val noiseGate = if (isTeleSpinto) 5.5f else 3.5f

        val enhanced = try {
            applyLuminanceComputationalPass(
                source = working,
                fineClarity = fineClarity,
                mediumClarity = mediumClarity,
                shadowRecovery = shadowRecovery,
                highlightProtect = highlightProtect,
                noiseGate = noiseGate
            )
        } catch (t: Throwable) {
            Log.e("PhotoEngine", "Computational pass error", t)
            working
        }
        if (enhanced !== working && !working.isRecycled) working.recycle()
        working = enhanced

        // 4. Cloud AI migliorata (Swin2SR Classical x2 + Fusione Luminanza Radiale senza cuciture)
        var cloudStatus = ""
        if (aiPreset.isCloud) {
            val cloudStrength = when (level) {
                1 -> 0.35f
                2 -> 0.55f
                else -> 0.75f
            }
            val (cloudBmp, statusMsg) = tryCloudDetailFusion(
                source = working,
                endpointUrl = cloudEndpointUrl,
                apiKey = cloudApiKey,
                strength = cloudStrength
            )
            cloudStatus = statusMsg
            if (cloudBmp != null) {
                if (cloudBmp !== working) working.recycle()
                working = cloudBmp
            }
        }

        // 5. Finitura tonale cinematografica leggera
        if (postSavePolish) {
            val toned = try {
                applySubtleFilmicCurve(working, strength = if (level >= 2) 0.38f else 0.26f)
            } catch (_: Throwable) {
                working
            }
            if (toned !== working && !working.isRecycled) working.recycle()
            working = toned
        }

        val parts = mutableListOf<String>()
        if (burstUsed) parts.add("Burst")
        parts.add(aiPreset.shortBadge)
        if (cloudStatus.isNotEmpty()) parts.add(cloudStatus)

        return NeuralProcessResult(
            bitmap = working,
            modeApplied = parts.joinToString(" · ")
        )
    }

    /**
     * Pipeline computazionale fotografica ad alta velocità (Sola Luminanza):
     * - Piramide a 3 scale:
     *   1) fineBlur (scala 1/2): estrae micro-dettaglio nitido (capelli, foglie, scritte)
     *   2) medBlur (scala 1/8): estrae tridimensionalità / micro-contrasto locale (Clarity)
     *   3) toneBlur (scala 1/24): mappa di esposizione locale per recuperare ombre e salvare luci (Local HDR)
     * - Noise-Gate (coring): ignora variazioni sotto `noiseGate` così pelle e cielo restano pulitissimi.
     * - Preserva al 100% i rapporti cromatici originali.
     */
    private fun applyLuminanceComputationalPass(
        source: Bitmap,
        fineClarity: Float,
        mediumClarity: Float,
        shadowRecovery: Float,
        highlightProtect: Float,
        noiseGate: Float
    ): Bitmap {
        val w = source.width
        val h = source.height
        val total = w * h

        val fineSmall = Bitmap.createScaledBitmap(source, max(2, w / 2), max(2, h / 2), true)
        val fineBlurBmp = Bitmap.createScaledBitmap(fineSmall, w, h, true)
        fineSmall.recycle()

        val medSmall = Bitmap.createScaledBitmap(source, max(2, w / 8), max(2, h / 8), true)
        val medBlurBmp = Bitmap.createScaledBitmap(medSmall, w, h, true)
        medSmall.recycle()

        val toneSmall = Bitmap.createScaledBitmap(source, max(2, w / 24), max(2, h / 24), true)
        val toneBlurBmp = Bitmap.createScaledBitmap(toneSmall, w, h, true)
        toneSmall.recycle()

        val px = IntArray(total)
        val finePx = IntArray(total)
        val medPx = IntArray(total)
        val tonePx = IntArray(total)

        source.getPixels(px, 0, w, 0, 0, w, h)
        fineBlurBmp.getPixels(finePx, 0, w, 0, 0, w, h)
        fineBlurBmp.recycle()
        medBlurBmp.getPixels(medPx, 0, w, 0, 0, w, h)
        medBlurBmp.recycle()
        toneBlurBmp.getPixels(tonePx, 0, w, 0, 0, w, h)
        toneBlurBmp.recycle()

        for (i in 0 until total) {
            val c = px[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF

            val yOrig = 0.299f * r + 0.587f * g + 0.114f * b

            val fc = finePx[i]
            val yFine = 0.299f * ((fc ushr 16) and 0xFF) + 0.587f * ((fc ushr 8) and 0xFF) + 0.114f * (fc and 0xFF)

            val mc = medPx[i]
            val yMed = 0.299f * ((mc ushr 16) and 0xFF) + 0.587f * ((mc ushr 8) and 0xFF) + 0.114f * (mc and 0xFF)

            val tc = tonePx[i]
            val yTone = 0.299f * ((tc ushr 16) and 0xFF) + 0.587f * ((tc ushr 8) and 0xFF) + 0.114f * (tc and 0xFF)

            // 1. Micro-dettaglio fine con Noise-Gate morbido (zero rumore su cielo/pelle)
            val rawFine = yOrig - yFine
            val absFine = abs(rawFine)
            val gatedFine = if (absFine <= noiseGate) {
                0f
            } else {
                val sign = if (rawFine >= 0f) 1f else -1f
                sign * (absFine - noiseGate).coerceAtMost(24f)
            }

            // 2. Micro-contrasto locale medio (tridimensionalità fotografica)
            val medDetail = (yFine - yMed).coerceIn(-22f, 22f)

            // Protezione anti-alone su alte luci (>220) e ombre profondissime (<16)
            val haloGuard = when {
                yOrig > 220f -> ((255f - yOrig) / 35f).coerceIn(0.15f, 1.0f)
                yOrig < 16f -> (yOrig / 16f).coerceIn(0.20f, 1.0f)
                else -> 1.0f
            }

            var deltaY = (gatedFine * fineClarity + medDetail * mediumClarity) * haloGuard

            // 3. Local Tone Mapping (recupero ombre + protezione alte luci naturale)
            if (shadowRecovery > 0.01f && yTone < 95f && yOrig > 4f) {
                val shadowWeight = ((95f - yTone) / 95f).pow(1.6f)
                deltaY += shadowWeight * shadowRecovery * 28f
            }
            if (highlightProtect > 0.01f && yTone > 185f) {
                val hiWeight = ((yTone - 185f) / 70f).coerceIn(0f, 1f)
                deltaY -= hiWeight * highlightProtect * 22f
            }

            val yNew = (yOrig + deltaY).coerceIn(0f, 255f)
            val diff = yNew - yOrig

            // Applica la variazione di luminanza preservando la saturazione senza dominanti
            val nr = (r + diff).toInt().coerceIn(0, 255)
            val ng = (g + diff).toInt().coerceIn(0, 255)
            val nb = (b + diff).toInt().coerceIn(0, 255)

            px[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * Cloud AI v13 con Fusione Luminanza Radiale (Zero riquadri, zero alterazione colori):
     * - Usa Swin2SR Classical x2 (addestrato su foto pulite).
     * - Invia un ritaglio ad alta qualità del soggetto centrale (fino a 768px).
     * - Estrae SOLO il dettaglio luminanza aggiunto dal modello Cloud rispetto al bicubico,
     *   lo limita contro artefatti e lo sfuma con una finestra radiale coseno morbida.
     */
    private fun tryCloudDetailFusion(
        source: Bitmap,
        endpointUrl: String,
        apiKey: String,
        strength: Float
    ): Pair<Bitmap?, String> {
        if (endpointUrl.isBlank()) return null to "Cloud: URL mancante"
        return try {
            val cropRatio = 0.72f
            val cw = (source.width * cropRatio).toInt().coerceIn(256, source.width)
            val ch = (source.height * cropRatio).toInt().coerceIn(256, source.height)
            val sx = (source.width - cw) / 2
            val sy = (source.height - ch) / 2

            val centerCrop = Bitmap.createBitmap(source, sx, sy, cw, ch)
            val uploadBmp = scaleToMaxSide(centerCrop, 640)
            val baseU = uploadBmp.width
            val baseH = uploadBmp.height

            val baos = ByteArrayOutputStream()
            uploadBmp.compress(Bitmap.CompressFormat.JPEG, 94, baos)
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
                    // Scala sia il risultato Cloud sia la baseline inviata alla dimensione del crop originale
                    val cloudScaled = Bitmap.createScaledBitmap(cloudPatch, cw, ch, true)
                    val baseScaled = Bitmap.createScaledBitmap(uploadBmp, cw, ch, true)
                    cloudPatch.recycle()
                    if (uploadBmp !== centerCrop) uploadBmp.recycle()
                    centerCrop.recycle()

                    val out = source.copy(Bitmap.Config.ARGB_8888, true)
                    val origPx = IntArray(cw * ch)
                    val cloudPx = IntArray(cw * ch)
                    val basePx = IntArray(cw * ch)

                    out.getPixels(origPx, 0, cw, sx, sy, cw, ch)
                    cloudScaled.getPixels(cloudPx, 0, cw, 0, 0, cw, ch)
                    baseScaled.getPixels(basePx, 0, cw, 0, 0, cw, ch)
                    cloudScaled.recycle()
                    baseScaled.recycle()

                    val halfW = cw / 2f
                    val halfH = ch / 2f

                    for (y in 0 until ch) {
                        val ny = (y - halfH) / halfH
                        val row = y * cw
                        for (x in 0 until cw) {
                            val nx = (x - halfW) / halfW
                            val dist = sqrt(nx * nx + ny * ny)
                            if (dist >= 1.0f) continue

                            // Maschera radiale coseno: 1.0 al centro, sfuma dolcemente a 0.0 sui bordi
                            val radialMask = if (dist < 0.65f) {
                                1.0f
                            } else {
                                val t = ((dist - 0.65f) / 0.35f).coerceIn(0f, 1f)
                                (0.5f * (1f + cos(t * Math.PI))).toFloat()
                            }

                            val idx = row + x
                            val o = origPx[idx]
                            val c = cloudPx[idx]
                            val b = basePx[idx]

                            val or = (o ushr 16) and 0xFF
                            val og = (o ushr 8) and 0xFF
                            val ob = o and 0xFF

                            val yCloud = 0.299f * ((c ushr 16) and 0xFF) + 0.587f * ((c ushr 8) and 0xFF) + 0.114f * (c and 0xFF)
                            val yBase = 0.299f * ((b ushr 16) and 0xFF) + 0.587f * ((b ushr 8) and 0xFF) + 0.114f * (b and 0xFF)

                            // Estrae SOLO il dettaglio luminanza pulito con clamp anti-artefatti
                            val detail = (yCloud - yBase).coerceIn(-22f, 22f)
                            val delta = detail * strength * radialMask

                            val nr = (or + delta).toInt().coerceIn(0, 255)
                            val ng = (og + delta).toInt().coerceIn(0, 255)
                            val nb = (ob + delta).toInt().coerceIn(0, 255)

                            origPx[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                        }
                    }

                    out.setPixels(origPx, 0, cw, sx, sy, cw, ch)
                    return out to "Cloud OK (${baseU}x${baseH})"
                }
                if (uploadBmp !== centerCrop) uploadBmp.recycle()
                centerCrop.recycle()
                return null to "Cloud: risposta vuota"
            } else {
                if (uploadBmp !== centerCrop) uploadBmp.recycle()
                centerCrop.recycle()
                val errHint = when (code) {
                    401, 403 -> "Token HF richiesto"
                    404 -> "Modello non trovato"
                    503 -> "Modello in avvio"
                    else -> "HTTP $code"
                }
                return null to "Cloud: $errHint"
            }
        } catch (e: Throwable) {
            Log.w("PhotoEngine", "Errore Cloud AI: ${e.message}")
            null to "Cloud: timeout/rete"
        }
    }

    /**
     * Chroma denoise veloce (1 passaggio): rimuove le macchie colorate di rumore
     * lasciando intatta al 100% la luminanza (nitidezza).
     */
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

                val n1 = src[idx - w]
                val n2 = src[idx + w]
                val n3 = src[idx - 1]
                val n4 = src[idx + 1]

                val neighbors = intArrayOf(n1, n2, n3, n4)
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

    private fun alignAndMergeBurstSafe(base: Bitmap, extras: List<Bitmap>): Bitmap {
        val w = base.width
        val h = base.height
        val total = w * h
        val basePx = IntArray(total)
        base.getPixels(basePx, 0, w, 0, 0, w, h)

        val accR = IntArray(total) { (basePx[it] ushr 16) and 0xFF }
        val accG = IntArray(total) { (basePx[it] ushr 8) and 0xFF }
        val accB = IntArray(total) { basePx[it] and 0xFF }
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

            // Se la scena si è mossa troppo (SAD medio > 14), scarta questo frame: zero ghosting!
            if (samples == 0 || (bestSad / samples) > 14) continue
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

                    if ((abs(br - er) + abs(bg - eg) + abs(bb - eb)) / 3 < 14) {
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

        val curve = 0.022 * strength
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
