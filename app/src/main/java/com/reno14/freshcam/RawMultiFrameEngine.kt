package com.reno14.freshcam

import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow

class RawMultiFrameEngine {

    fun processRawBurst(
        payload: RawBurstPayload,
        profile: CameraProfile,
        profileIntensity: Float,
        brightnessBoost: Float,
        digitalZoomRatio: Float,
        keepCroppedForNeuralZoom: Boolean,
        activeLens: LensType,
        postSavePolish: Boolean
    ): NeuralProcessResult {
        val frames = payload.frames
        val meta = payload.meta
        val w = frames.first().width
        val h = frames.first().height

        val (mergedRaw, mergedCount) = if (frames.size > 1) {
            alignAndMergeLinearBayer(frames, meta.whiteLevel)
        } else {
            frames.first().data to 1
        }

        var effectiveWhite = meta.whiteLevel.toFloat().coerceAtLeast(255f)
        var maxObserved = 0
        run {
            var i = 0
            while (i < mergedRaw.size) {
                val v = mergedRaw[i].toInt() and 0xFFFF
                if (v > maxObserved) maxObserved = v
                i += 97
            }
        }
        val blScale = if (maxObserved > effectiveWhite * 1.8f && effectiveWhite < 20000f) {
            val mult = (65535f / effectiveWhite).coerceAtLeast(1f)
            effectiveWhite = 65535f
            mult
        } else {
            1f
        }

        val blackF = FloatArray(4) { (meta.blackLevels[it] * blScale).coerceAtLeast(0f) }
        val invRange = FloatArray(4) { 1f / (effectiveWhite - blackF[it]).coerceAtLeast(64f) }

        val cfaMap = when (meta.cfaArrangement) {
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> intArrayOf(1, 0, 2, 1)
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> intArrayOf(1, 2, 0, 1)
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> intArrayOf(2, 1, 1, 0)
            else -> intArrayOf(0, 1, 1, 2)
        }

        val wb = meta.wbGains ?: estimateGrayWorldWb(mergedRaw, w, h, cfaMap, blackF, invRange)
        val exposureGain = estimateAutoExposureGain(mergedRaw, w, h, cfaMap, blackF, invRange, wb)

        val ccm = meta.ccm ?: floatArrayOf(
            1.36f, -0.26f, -0.10f,
            -0.14f, 1.24f, -0.10f,
            -0.06f, -0.28f, 1.34f
        )

        val lutSize = 4095
        val srgbLut = IntArray(lutSize + 1) { idx ->
            val lin = idx.toFloat() / lutSize.toFloat()
            val mapped = (lin * (1.0f + 0.14f * lin)) / (1.0f + 0.14f * lin * lin + 0.08f * lin)
            val clamped = mapped.coerceIn(0f, 1f)
            val srgb = if (clamped <= 0.0031308f) {
                12.92f * clamped
            } else {
                1.055f * clamped.pow(1f / 2.4f) - 0.055f
            }
            (srgb * 255f + 0.5f).toInt().coerceIn(0, 255)
        }

        val argb = IntArray(w * h)
        val rGain = wb[0] * exposureGain
        val gGain = wb[1] * exposureGain
        val bGain = wb[2] * exposureGain

        val c00 = ccm[0]; val c01 = ccm[1]; val c02 = ccm[2]
        val c10 = ccm[3]; val c11 = ccm[4]; val c12 = ccm[5]
        val c20 = ccm[6]; val c21 = ccm[7]; val c22 = ccm[8]

        fun normPx(idx: Int, phase: Int): Float {
            val raw = (mergedRaw[idx].toInt() and 0xFFFF).toFloat()
            val v = (raw - blackF[phase]) * invRange[phase]
            return if (v > 0f) v else 0f
        }

        for (y in 1 until h - 1) {
            val row = y * w
            val rMod = (y and 1) shl 1
            val rModOpp = ((y + 1) and 1) shl 1
            for (x in 1 until w - 1) {
                val idx = row + x
                val cMod = x and 1
                val cModOpp = (x + 1) and 1
                val phase = rMod or cMod
                val ch = cfaMap[phase]

                val phaseH = rMod or cModOpp
                val phaseV = rModOpp or cMod
                val phaseD = rModOpp or cModOpp

                var linR: Float
                var linG: Float
                var linB: Float

                if (ch == 1) {
                    linG = normPx(idx, phase)
                    val hCh = cfaMap[phaseH]
                    val hVal = 0.5f * (normPx(idx - 1, phaseH) + normPx(idx + 1, phaseH))
                    val vVal = 0.5f * (normPx(idx - w, phaseV) + normPx(idx + w, phaseV))
                    if (hCh == 0) {
                        linR = hVal
                        linB = vVal
                    } else {
                        linB = hVal
                        linR = vVal
                    }
                } else {
                    val centerVal = normPx(idx, phase)
                    val gL = normPx(idx - 1, phaseH)
                    val gR = normPx(idx + 1, phaseH)
                    val gU = normPx(idx - w, phaseV)
                    val gD = normPx(idx + w, phaseV)

                    val gradH = abs(gL - gR)
                    val gradV = abs(gU - gD)
                    linG = when {
                        gradH + 0.008f < gradV -> 0.5f * (gL + gR)
                        gradV + 0.008f < gradH -> 0.5f * (gU + gD)
                        else -> 0.25f * (gL + gR + gU + gD)
                    }

                    val diagVal = 0.25f * (
                        normPx(idx - w - 1, phaseD) +
                            normPx(idx - w + 1, phaseD) +
                            normPx(idx + w - 1, phaseD) +
                            normPx(idx + w + 1, phaseD)
                        )
                    if (ch == 0) {
                        linR = centerVal
                        linB = diagVal
                    } else {
                        linB = centerVal
                        linR = diagVal
                    }
                }

                val wr = linR * rGain
                val wg = linG * gGain
                val wbVal = linB * bGain

                val cr = (c00 * wr + c01 * wg + c02 * wbVal).coerceIn(0f, 1f)
                val cg = (c10 * wr + c11 * wg + c12 * wbVal).coerceIn(0f, 1f)
                val cb = (c20 * wr + c21 * wg + c22 * wbVal).coerceIn(0f, 1f)

                val outR = srgbLut[(cr * lutSize).toInt()]
                val outG = srgbLut[(cg * lutSize).toInt()]
                val outB = srgbLut[(cb * lutSize).toInt()]
                argb[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }

        for (x in 0 until w) {
            argb[x] = argb[w + x]
            argb[(h - 1) * w + x] = argb[(h - 2) * w + x]
        }
        for (y in 0 until h) {
            argb[y * w] = argb[y * w + 1]
            argb[y * w + (w - 1)] = argb[y * w + (w - 2)]
        }

        var working = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        working.setPixels(argb, 0, w, 0, 0, w, h)

        if (meta.sensorOrientation != 0) {
            val m = Matrix().apply { postRotate(meta.sensorOrientation.toFloat()) }
            val rot = Bitmap.createBitmap(working, 0, 0, working.width, working.height, m, true)
            working.recycle()
            working = rot
        }

        if (digitalZoomRatio > 1.03f) {
            val f = digitalZoomRatio.coerceIn(1.0f, 34.0f)
            val newW = (working.width / f).toInt().coerceAtLeast(160)
            val newH = (working.height / f).toInt().coerceAtLeast(160)
            val sx = (working.width - newW) / 2
            val sy = (working.height - newH) / 2
            val cropped = Bitmap.createBitmap(working, sx, sy, newW, newH)
            if (keepCroppedForNeuralZoom) {
                // Consegna i pixel ottici puri alla rete neurale 4x senza sfocarli con upscaling bilineare!
                working.recycle()
                working = cropped
            } else {
                val scaled = Bitmap.createScaledBitmap(cropped, working.width, working.height, true)
                cropped.recycle()
                working.recycle()
                working = scaled
            }
        }

        if (!keepCroppedForNeuralZoom) {
            val cleaned = shadowAwareFinish(
                source = working,
                isTele = activeLens == LensType.TELE_50MP
            )
            if (cleaned !== working) {
                working.recycle()
                working = cleaned
            }
        }

        if (profile != CameraProfile.NATURAL || abs(brightnessBoost) >= 0.005f) {
            val graded = profile.applyToBitmap(working, strength = profileIntensity, brightnessBoost = brightnessBoost)
            working.recycle()
            working = graded
        }

        if (postSavePolish) {
            val polished = applySubtleScurve(working)
            if (polished !== working) {
                working.recycle()
                working = polished
            }
        }

        val lensTag = if (activeLens == LensType.TELE_50MP) "Tele RAW ${mergedCount}x" else "RAW ${mergedCount}x"
        val isoTag = meta.iso?.let { " · ISO $it" } ?: ""
        return NeuralProcessResult(
            bitmap = working,
            modeApplied = "$lensTag → JPG$isoTag"
        )
    }

    private fun alignAndMergeLinearBayer(frames: List<RawFrameData>, whiteLevel: Int): Pair<ShortArray, Int> {
        val base = frames.first()
        val w = base.width
        val h = base.height
        val total = w * h
        val baseData = base.data

        val accSum = IntArray(total) { baseData[it].toInt() and 0xFFFF }
        val accCount = ByteArray(total) { 1 }

        val margin = 16
        val patch = min(160, min(w, h) / 4) and 0xFFFE
        if (patch < 32) return baseData to 1
        val cx = ((w - patch) / 2) and 0xFFFE
        val cy = ((h - patch) / 2) and 0xFFFE

        val ghostThreshold = (whiteLevel * 0.055f).toInt().coerceIn(48, 3600)
        val sadRejectLimit = (whiteLevel * 0.065f).toLong().coerceIn(55L, 4200L)

        var mergedFrames = 1

        for (fIdx in 1 until frames.size) {
            val extra = frames[fIdx]
            if (extra.width != w || extra.height != h) continue
            val extData = extra.data

            var bestDx = 0
            var bestDy = 0
            var bestSad = Long.MAX_VALUE
            var samples = 0

            fun evalShift(dx: Int, dy: Int, step: Int): Pair<Long, Int> {
                var sad = 0L
                var cnt = 0
                var py = 0
                while (py < patch) {
                    val rowB = (cy + py) * w
                    val rowE = (cy + py + dy) * w
                    var px = 0
                    while (px < patch) {
                        val bv = baseData[rowB + cx + px].toInt() and 0xFFFF
                        val ev = extData[rowE + cx + px + dx].toInt() and 0xFFFF
                        sad += abs(bv - ev)
                        cnt++
                        px += step
                    }
                    py += step
                }
                return sad to cnt
            }

            for (dy in -12..12 step 4) {
                for (dx in -12..12 step 4) {
                    val (sad, cnt) = evalShift(dx, dy, 6)
                    if (sad < bestSad) {
                        bestSad = sad
                        bestDx = dx
                        bestDy = dy
                        samples = cnt
                    }
                }
            }

            val cDx = bestDx
            val cDy = bestDy
            bestSad = Long.MAX_VALUE
            for (dy in (cDy - 2)..(cDy + 2) step 2) {
                for (dx in (cDx - 2)..(cDx + 2) step 2) {
                    if (abs(dx) > 14 || abs(dy) > 14) continue
                    val (sad, cnt) = evalShift(dx, dy, 4)
                    if (sad < bestSad) {
                        bestSad = sad
                        bestDx = dx
                        bestDy = dy
                        samples = cnt
                    }
                }
            }

            if (samples == 0 || (bestSad / samples) > sadRejectLimit) continue
            mergedFrames++

            for (y in margin until h - margin) {
                val rowB = y * w
                val rowE = (y + bestDy) * w
                for (x in margin until w - margin) {
                    val idxB = rowB + x
                    val bv = baseData[idxB].toInt() and 0xFFFF
                    val ev = extData[rowE + x + bestDx].toInt() and 0xFFFF
                    val localLimit = ghostThreshold + (bv shr 4)
                    if (abs(bv - ev) <= localLimit) {
                        accSum[idxB] += ev
                        accCount[idxB] = (accCount[idxB] + 1).toByte()
                    }
                }
            }
        }

        if (mergedFrames <= 1) return baseData to 1

        val merged = ShortArray(total) { i ->
            val cnt = accCount[i].toInt() and 0xFF
            ((accSum[i] + (cnt shr 1)) / cnt).coerceIn(0, 65535).toShort()
        }
        return merged to mergedFrames
    }

    private fun estimateGrayWorldWb(
        raw: ShortArray,
        w: Int,
        h: Int,
        cfaMap: IntArray,
        blackF: FloatArray,
        invRange: FloatArray
    ): FloatArray {
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var cntR = 0
        var cntG = 0
        var cntB = 0

        var y = h / 8
        val yEnd = (h * 7) / 8
        while (y < yEnd) {
            val row = y * w
            val rMod = (y and 1) shl 1
            var x = w / 8
            val xEnd = (w * 7) / 8
            while (x < xEnd) {
                val phase = rMod or (x and 1)
                val v = ((raw[row + x].toInt() and 0xFFFF) - blackF[phase]) * invRange[phase]
                if (v in 0.03f..0.85f) {
                    when (cfaMap[phase]) {
                        0 -> { sumR += v; cntR++ }
                        1 -> { sumG += v; cntG++ }
                        else -> { sumB += v; cntB++ }
                    }
                }
                x += 7
            }
            y += 7
        }

        if (cntR < 50 || cntG < 50 || cntB < 50) return floatArrayOf(1.65f, 1.0f, 1.45f)
        val avgR = (sumR / cntR).toFloat().coerceAtLeast(0.001f)
        val avgG = (sumG / cntG).toFloat().coerceAtLeast(0.001f)
        val avgB = (sumB / cntB).toFloat().coerceAtLeast(0.001f)

        return floatArrayOf(
            (avgG / avgR).coerceIn(0.8f, 3.2f),
            1.0f,
            (avgG / avgB).coerceIn(0.8f, 3.2f)
        )
    }

    private fun estimateAutoExposureGain(
        raw: ShortArray,
        w: Int,
        h: Int,
        cfaMap: IntArray,
        blackF: FloatArray,
        invRange: FloatArray,
        wb: FloatArray
    ): Float {
        val hist = IntArray(256)
        var total = 0
        var y = h / 10
        val yEnd = (h * 9) / 10
        while (y < yEnd) {
            val row = y * w
            val rMod = (y and 1) shl 1
            var x = w / 10
            val xEnd = (w * 9) / 10
            while (x < xEnd) {
                val phase = rMod or (x and 1)
                if (cfaMap[phase] == 1) {
                    val v = (((raw[row + x].toInt() and 0xFFFF) - blackF[phase]) * invRange[phase] * wb[1]).coerceIn(0f, 1f)
                    val bin = (v * 255f).toInt().coerceIn(0, 255)
                    hist[bin]++
                    total++
                }
                x += 8
            }
            y += 8
        }
        if (total < 100) return 1.15f

        val target50 = (total * 0.50).toInt()
        val target98 = (total * 0.98).toInt()
        var acc = 0
        var p50 = 0.18f
        var p98 = 0.85f
        var found50 = false
        for (i in 0..255) {
            acc += hist[i]
            if (!found50 && acc >= target50) {
                p50 = (i + 0.5f) / 255f
                found50 = true
            }
            if (acc >= target98) {
                p98 = (i + 0.5f) / 255f
                break
            }
        }

        val desiredGain = (0.185f / p50.coerceAtLeast(0.03f)).coerceIn(0.85f, 2.35f)
        val highlightCap = (0.94f / p98.coerceAtLeast(0.15f)).coerceIn(0.85f, 2.5f)
        return min(desiredGain, highlightCap)
    }

    private fun shadowAwareFinish(source: Bitmap, isTele: Boolean): Bitmap {
        val w = source.width
        val h = source.height
        val src = IntArray(w * h)
        val dst = IntArray(w * h)
        source.getPixels(src, 0, w, 0, 0, w, h)

        val sharpGain = if (isTele) 0.34f else 0.24f
        val noiseGate = 4.2f

        for (y in 2 until h - 2) {
            val row = y * w
            for (x in 2 until w - 2) {
                val idx = row + x
                val c = src[idx]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val yC = (77 * r + 150 * g + 29 * b) shr 8

                if (yC < 88) {
                    var sumR = r * 3
                    var sumG = g * 3
                    var sumB = b * 3
                    var sumW = 3

                    val offsets = intArrayOf(-w, w, -1, 1, -2 * w, 2 * w, -2, 2, -w - 1, -w + 1, w - 1, w + 1)
                    val lumaThreshold = if (yC < 45) 16 else 12
                    for (off in offsets) {
                        val n = src[idx + off]
                        val nr = (n ushr 16) and 0xFF
                        val ng = (n ushr 8) and 0xFF
                        val nb = n and 0xFF
                        val yN = (77 * nr + 150 * ng + 29 * nb) shr 8
                        if (abs(yC - yN) <= lumaThreshold) {
                            sumR += nr
                            sumG += ng
                            sumB += nb
                            sumW++
                        }
                    }
                    val blend = ((88 - yC) / 68f).coerceIn(0f, 0.85f)
                    val outR = (r + (sumR / sumW - r) * blend).toInt().coerceIn(0, 255)
                    val outG = (g + (sumG / sumW - g) * blend).toInt().coerceIn(0, 255)
                    val outB = (b + (sumB / sumW - b) * blend).toInt().coerceIn(0, 255)
                    dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                } else {
                    val n = src[idx - w]
                    val s = src[idx + w]
                    val e = src[idx + 1]
                    val west = src[idx - 1]
                    val yN = 0.299f * ((n ushr 16) and 0xFF) + 0.587f * ((n ushr 8) and 0xFF) + 0.114f * (n and 0xFF)
                    val yS = 0.299f * ((s ushr 16) and 0xFF) + 0.587f * ((s ushr 8) and 0xFF) + 0.114f * (s and 0xFF)
                    val yE = 0.299f * ((e ushr 16) and 0xFF) + 0.587f * ((e ushr 8) and 0xFF) + 0.114f * (e and 0xFF)
                    val yW = 0.299f * ((west ushr 16) and 0xFF) + 0.587f * ((west ushr 8) and 0xFF) + 0.114f * (west and 0xFF)

                    val yAvg = (yN + yS + yE + yW) * 0.25f
                    val detail = yC - yAvg
                    val absD = abs(detail)
                    val gated = if (absD <= noiseGate) 0f else {
                        val sign = if (detail >= 0f) 1f else -1f
                        sign * (absD - noiseGate).coerceAtMost(18f)
                    }
                    val shadowRamp = ((yC - 88f) / 36f).coerceIn(0f, 1f)
                    val hiGuard = if (yC > 224) ((255f - yC) / 31f).coerceIn(0.15f, 1f) else 1f
                    val delta = gated * sharpGain * shadowRamp * hiGuard

                    val outR = (r + delta).toInt().coerceIn(0, 255)
                    val outG = (g + delta).toInt().coerceIn(0, 255)
                    val outB = (b + delta).toInt().coerceIn(0, 255)
                    dst[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
        }

        for (y in 0 until h) {
            if (y < 2 || y >= h - 2) {
                System.arraycopy(src, y * w, dst, y * w, w)
            } else {
                dst[y * w] = src[y * w]
                dst[y * w + 1] = src[y * w + 1]
                dst[y * w + w - 2] = src[y * w + w - 2]
                dst[y * w + w - 1] = src[y * w + w - 1]
            }
        }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(dst, 0, w, 0, 0, w, h)
        return out
    }

    private fun applySubtleScurve(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val px = IntArray(w * h)
        bitmap.getPixels(px, 0, w, 0, 0, w, h)
        val lut = IntArray(256) { v ->
            val n = v / 255.0
            val s = n - 0.016 * kotlin.math.sin(2.0 * Math.PI * n)
            (s * 255.0).toInt().coerceIn(0, 255)
        }
        for (i in px.indices) {
            val p = px[i]
            px[i] = (0xFF shl 24) or
                (lut[(p ushr 16) and 0xFF] shl 16) or
                (lut[(p ushr 8) and 0xFF] shl 8) or
                lut[p and 0xFF]
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }
}
