package com.reno14.freshcam

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.FlashAuto
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.HdrAuto
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.delay
import kotlin.math.abs

private enum class SheetPage {
    NONE,
    PROFILE_FORMAT,
    FOCUS,
    SETTINGS
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FreshCameraScreen(lifecycleOwner: LifecycleOwner) {
    val context = LocalContext.current
    var state by remember { mutableStateOf(FreshCamState()) }
    var sheet by remember { mutableStateOf(SheetPage.NONE) }
    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var focusRing by remember { mutableStateOf<Offset?>(null) }
    var flashPulse by remember { mutableStateOf(false) }

    val engine = remember(context, lifecycleOwner) { FreshCameraEngine(context, lifecycleOwner) }

    LaunchedEffect(Unit) {
        val lenses = engine.discoverLenses()
        val hasModel = engine.checkNeuralModelReady()
        state = state.copy(detectedLenses = lenses, isNeuralModelReady = hasModel)
    }

    LaunchedEffect(state.bannerMessage) {
        if (state.bannerMessage != null) {
            delay(2500)
            state = state.copy(bannerMessage = null)
        }
    }

    LaunchedEffect(flashPulse) {
        if (flashPulse) {
            delay(85)
            flashPulse = false
        }
    }

    LaunchedEffect(focusRing, state.isSubjectLocked) {
        if (focusRing != null && !state.isSubjectLocked) {
            delay(1200)
            focusRing = null
        }
    }

    LaunchedEffect(
        previewView,
        state.activeLens,
        state.captureFormat,
        state.aiMode,
        state.stabilizationEnabled,
        state.isoValue,
        state.shutterSpeedDenom,
        state.evCompensation,
        state.focusMode,
        state.manualFocusDistance,
        state.shootingMode,
        state.flashMode
    ) {
        val pv = previewView ?: return@LaunchedEffect
        engine.bind(
            previewView = pv,
            state = state,
            onZoomBounds = { _, maxZ ->
                state = state.copy(maxLensZoomRatio = maxZ.coerceAtLeast(10.0f))
            },
            onBanner = { msg ->
                state = state.copy(bannerMessage = msg)
            }
        )
    }

    LaunchedEffect(state.zoomRatio) {
        engine.setZoom(state.zoomRatio)
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(state.focusMode) {
                    detectTapGestures { offset ->
                        focusRing = offset
                        if (state.focusMode == FocusMode.AI_LOCK) {
                            engine.applyFocusPoint(offset.x, offset.y, size.width.toFloat(), size.height.toFloat(), keepLocked = true)
                            state = state.copy(isSubjectLocked = true, bannerMessage = "AI Lock agganciato sull'oggetto")
                        } else if (state.focusMode == FocusMode.AUTO) {
                            engine.applyFocusPoint(offset.x, offset.y, size.width.toFloat(), size.height.toFloat(), keepLocked = false)
                            state = state.copy(isSubjectLocked = false, bannerMessage = "Autofocus sul punto")
                        }
                    }
                }
        ) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                        previewView = this
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                if (state.gridEnabled) {
                    val w3 = size.width / 3f
                    val h3 = size.height / 3f
                    val gCol = Color.White.copy(alpha = 0.20f)
                    drawLine(gCol, Offset(w3, 0f), Offset(w3, size.height), 1f)
                    drawLine(gCol, Offset(w3 * 2f, 0f), Offset(w3 * 2f, size.height), 1f)
                    drawLine(gCol, Offset(0f, h3), Offset(size.width, h3), 1f)
                    drawLine(gCol, Offset(0f, h3 * 2f), Offset(size.width, h3 * 2f), 1f)
                }

                focusRing?.let { pt ->
                    if (state.isSubjectLocked) {
                        val box = 110f
                        drawRect(
                            color = Color(0xFFFFCA28),
                            topLeft = Offset(pt.x - box / 2f, pt.y - box / 2f),
                            size = androidx.compose.ui.geometry.Size(box, box),
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3.5f)
                        )
                    } else {
                        drawCircle(
                            color = state.profile.accentColor,
                            radius = 44f,
                            center = pt,
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3f)
                        )
                    }
                }
            }

            if (flashPulse) {
                Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.4f)))
            }
        }

        // Top Bar con tasto SR: ON / OFF
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                modifier = Modifier.horizontalScroll(rememberScrollState()),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = {
                        val next = when (state.flashMode) {
                            FlashMode.OFF -> FlashMode.AUTO
                            FlashMode.AUTO -> FlashMode.ON
                            FlashMode.ON -> FlashMode.TORCH
                            FlashMode.TORCH -> FlashMode.OFF
                        }
                        state = state.copy(flashMode = next, bannerMessage = "Flash: ${next.label}")
                    }
                ) {
                    Icon(
                        imageVector = when (state.flashMode) {
                            FlashMode.OFF -> Icons.Default.FlashOff
                            FlashMode.AUTO -> Icons.Default.FlashAuto
                            FlashMode.ON, FlashMode.TORCH -> Icons.Default.FlashOn
                        },
                        contentDescription = "Flash",
                        tint = if (state.flashMode == FlashMode.OFF) Color.White.copy(alpha = 0.75f) else state.profile.accentColor
                    )
                }

                IconButton(
                    onClick = {
                        val next = when (state.hdrMode) {
                            HdrMode.OFF -> HdrMode.NATURAL
                            HdrMode.NATURAL -> HdrMode.AUTO
                            HdrMode.AUTO -> HdrMode.OFF
                        }
                        state = state.copy(hdrMode = next, bannerMessage = next.label)
                    }
                ) {
                    Icon(
                        imageVector = Icons.Default.HdrAuto,
                        contentDescription = "HDR",
                        tint = if (state.hdrMode == HdrMode.OFF) Color.White.copy(alpha = 0.75f) else state.profile.accentColor
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            if (state.neuralSuperResEnabled) Color(0xFF00ACC1).copy(alpha = 0.90f)
                            else Color.Black.copy(alpha = 0.65f)
                        )
                        .border(
                            1.dp,
                            if (state.neuralSuperResEnabled) Color.White else Color.White.copy(alpha = 0.35f),
                            RoundedCornerShape(16.dp)
                        )
                        .clickable {
                            val nextSr = !state.neuralSuperResEnabled
                            val msg = if (nextSr) "Super-Resolution SR: ON" else "Super-Resolution SR: OFF"
                            state = state.copy(neuralSuperResEnabled = nextSr, bannerMessage = msg)
                        }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(13.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = if (state.neuralSuperResEnabled) "SR: ON" else "SR: OFF",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Spacer(modifier = Modifier.width(6.dp))

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.Black.copy(alpha = 0.65f))
                        .border(1.dp, state.profile.accentColor.copy(alpha = 0.65f), RoundedCornerShape(16.dp))
                        .clickable { sheet = SheetPage.PROFILE_FORMAT }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = "${state.profile.displayName} · ${state.captureFormat.badge}",
                        color = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Spacer(modifier = Modifier.width(6.dp))

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.Black.copy(alpha = 0.65f))
                        .clickable { sheet = SheetPage.FOCUS }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = state.focusMode.shortLabel,
                        color = if (state.isSubjectLocked) Color(0xFFFFCA28) else state.profile.accentColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            IconButton(
                onClick = { sheet = SheetPage.SETTINGS },
                modifier = Modifier.clip(CircleShape).background(Color.Black.copy(alpha = 0.55f))
            ) {
                Icon(Icons.Default.Settings, contentDescription = "Impostazioni", tint = Color.White)
            }
        }

        if (state.isSubjectLocked) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .statusBarsPadding()
                    .padding(top = 58.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color(0xFFFFCA28))
                    .clickable {
                        engine.clearFocusLock()
                        focusRing = null
                        state = state.copy(isSubjectLocked = false, bannerMessage = "Fuoco sbloccato")
                    }
                    .padding(horizontal = 14.dp, vertical = 6.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Lock, contentDescription = null, tint = Color.Black, modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Oggetto bloccato · Tocca per sbloccare", color = Color.Black, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        } else {
            state.bannerMessage?.let { msg ->
                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .statusBarsPadding()
                        .padding(top = 58.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color.Black.copy(alpha = 0.78f))
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Text(msg, color = Color.White, fontSize = 12.sp)
                }
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .background(Color.Black.copy(alpha = 0.68f))
                .padding(top = 8.dp, bottom = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (state.activeLens == LensType.TELE_50MP) {
                val eqZoom = 3.5f * state.zoomRatio
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Tele ${String.format("%.1f", eqZoom)}x",
                        color = state.profile.accentColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.width(72.dp)
                    )
                    Slider(
                        value = state.zoomRatio,
                        valueRange = 1.0f..state.maxLensZoomRatio,
                        onValueChange = { z -> state = state.copy(zoomRatio = z) },
                        modifier = Modifier.weight(1f),
                        colors = SliderDefaults.colors(
                            thumbColor = state.profile.accentColor,
                            activeTrackColor = state.profile.accentColor
                        )
                    )
                    Text("MAX", color = Color.White.copy(alpha = 0.75f), fontSize = 10.sp, fontWeight = FontWeight.Bold)
                }
            }

            val presets = listOf(
                Triple("0.6x", LensType.ULTRAWIDE, 1.0f),
                Triple("1x", LensType.MAIN_50MP, 1.0f),
                Triple("3.5x", LensType.TELE_50MP, 1.0f),
                Triple("10x", LensType.TELE_50MP, 2.85f),
                Triple("20x", LensType.TELE_50MP, 5.7f),
                Triple("MAX", LensType.TELE_50MP, state.maxLensZoomRatio)
            )
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                presets.forEach { (label, lens, targetZoom) ->
                    val selected = state.activeLens == lens && abs(state.zoomRatio - targetZoom) < 0.45f
                    Box(
                        modifier = Modifier
                            .padding(horizontal = 3.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(if (selected) state.profile.accentColor else Color.DarkGray.copy(alpha = 0.72f))
                            .clickable {
                                state = state.copy(
                                    activeLens = lens,
                                    zoomRatio = targetZoom,
                                    bannerMessage = "${lens.label} ($label)"
                                )
                            }
                            .padding(horizontal = 9.dp, vertical = 5.dp)
                    ) {
                        Text(label, color = if (selected) Color.Black else Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(18.dp, Alignment.CenterHorizontally)
            ) {
                ShootingMode.entries.forEach { mode ->
                    val selected = state.shootingMode == mode
                    Text(
                        text = mode.label,
                        color = if (selected) state.profile.accentColor else Color.White.copy(alpha = 0.72f),
                        fontSize = 13.sp,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                        modifier = Modifier
                            .clickable {
                                val nextLens = if (mode == ShootingMode.TELE_50MP) LensType.TELE_50MP else state.activeLens
                                state = state.copy(shootingMode = mode, activeLens = nextLens, bannerMessage = "Modalità ${mode.label}")
                            }
                            .padding(vertical = 3.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 28.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .clip(CircleShape)
                        .background(Color.White.copy(alpha = 0.14f))
                        .clickable {
                            openOppoGallery(context, state.lastCapturedUri) { err ->
                                state = state.copy(bannerMessage = err)
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.PhotoLibrary, contentDescription = "Galleria Oppo", tint = Color.White, modifier = Modifier.size(22.dp))
                }

                Box(
                    modifier = Modifier
                        .size(74.dp)
                        .clip(CircleShape)
                        .border(4.dp, Color.White, CircleShape)
                        .padding(6.dp)
                        .clip(CircleShape)
                        .background(
                            if (state.shootingMode == ShootingMode.VIDEO) {
                                if (state.isRecordingVideo) Color.Red else Color(0xFFFF5252)
                            } else {
                                state.profile.accentColor
                            }
                        )
                        .clickable {
                            if (state.shootingMode == ShootingMode.VIDEO) {
                                engine.toggleVideo(
                                    state = state,
                                    onRecording = { rec -> state = state.copy(isRecordingVideo = rec) },
                                    onBanner = { msg -> state = state.copy(bannerMessage = msg) }
                                )
                            } else {
                                flashPulse = true
                                val snap = state
                                engine.shootInstant(snap) { uri, msg ->
                                    state = state.copy(
                                        lastCapturedUri = uri ?: state.lastCapturedUri,
                                        bannerMessage = msg
                                    )
                                }
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = when {
                            state.shootingMode == ShootingMode.VIDEO && state.isRecordingVideo -> Icons.Default.Stop
                            state.shootingMode == ShootingMode.VIDEO -> Icons.Default.Videocam
                            else -> Icons.Default.Camera
                        },
                        contentDescription = "Scatta",
                        tint = if (state.shootingMode == ShootingMode.VIDEO) Color.White else Color.Black,
                        modifier = Modifier.size(28.dp)
                    )
                }

                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .clip(CircleShape)
                        .background(Color.White.copy(alpha = 0.14f))
                        .clickable {
                            val nextLens = if (state.activeLens == LensType.FRONT) LensType.MAIN_50MP else LensType.FRONT
                            state = state.copy(activeLens = nextLens, zoomRatio = 1.0f, bannerMessage = nextLens.label)
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Cameraswitch, contentDescription = "Cambia camera", tint = Color.White, modifier = Modifier.size(22.dp))
                }
            }
        }

        if (sheet == SheetPage.PROFILE_FORMAT) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp)) {
                    Text("Tipo di scatto", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CaptureFormat.entries.forEach { fmt ->
                            FilterChip(
                                selected = state.captureFormat == fmt,
                                onClick = { state = state.copy(captureFormat = fmt) },
                                label = { Text(fmt.label) }
                            )
                        }
                    }

                    if (state.captureFormat == CaptureFormat.PRO) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Text("ISO ${state.isoValue}", fontSize = 12.sp, color = state.profile.accentColor)
                        Slider(
                            value = state.isoValue.toFloat(),
                            valueRange = 50f..3200f,
                            onValueChange = { state = state.copy(isoValue = (it.toInt() / 50) * 50) }
                        )
                        Text("Tempo 1/${state.shutterSpeedDenom}s", fontSize = 12.sp, color = state.profile.accentColor)
                        Slider(
                            value = state.shutterSpeedDenom.toFloat(),
                            valueRange = 4f..2000f,
                            onValueChange = { state = state.copy(shutterSpeedDenom = it.toInt().coerceAtLeast(4)) }
                        )
                    }

                    Spacer(modifier = Modifier.height(14.dp))
                    Text("Profili colore", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(CameraProfile.entries) { prof ->
                            val active = state.profile == prof
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(14.dp))
                                    .background(if (active) prof.accentColor.copy(alpha = 0.25f) else Color.White.copy(alpha = 0.08f))
                                    .border(1.dp, if (active) prof.accentColor else Color.White.copy(alpha = 0.2f), RoundedCornerShape(14.dp))
                                    .clickable { state = state.copy(profile = prof, bannerMessage = "Profilo ${prof.displayName}") }
                                    .padding(horizontal = 12.dp, vertical = 8.dp)
                            ) {
                                Text(prof.displayName, color = if (active) prof.accentColor else Color.White, fontSize = 12.sp, fontWeight = if (active) FontWeight.Bold else FontWeight.Normal)
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }

        if (sheet == SheetPage.FOCUS) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp)) {
                    Text("Messa a fuoco", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FocusMode.entries.forEach { fm ->
                            FilterChip(
                                selected = state.focusMode == fm,
                                onClick = {
                                    if (fm != FocusMode.AI_LOCK) engine.clearFocusLock()
                                    state = state.copy(focusMode = fm, isSubjectLocked = false, bannerMessage = fm.label)
                                },
                                label = { Text(fm.label) }
                            )
                        }
                    }
                    if (state.focusMode == FocusMode.MANUAL) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Text("Distanza focale manuale (Macro ↔ Infinito)", fontSize = 12.sp)
                        Slider(value = state.manualFocusDistance, onValueChange = { state = state.copy(manualFocusDistance = it) })
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }

        if (sheet == SheetPage.SETTINGS) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp)
                ) {
                    Text("Impostazioni Reno14 FreshCam", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text(
                        text = if (state.isNeuralModelReady) "Motore SR: Ibrido ESRGAN + Denoise attivo" else "Motore SR: Ibrido Denoise + Dettagli attivo",
                        fontSize = 12.sp,
                        color = Color(0xFF4FC3F7)
                    )

                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(8.dp))

                    ToggleRow(
                        title = "Super-Resolution Ibrida (ON / OFF)",
                        subtitle = "Pulisce prima il rumore del teleobiettivo e fonde il dettaglio in modo naturale",
                        checked = state.neuralSuperResEnabled,
                        onChange = { state = state.copy(neuralSuperResEnabled = it) }
                    )

                    ToggleRow(
                        title = "Ottimizzazione al Salvataggio (disattivabile)",
                        subtitle = "Bilancia ombre e alte luci in background senza bloccare lo scatto",
                        checked = state.postSavePolishEnabled,
                        onChange = { state = state.copy(postSavePolishEnabled = it) }
                    )

                    ToggleRow(
                        title = "Stabilizzazione Avanzata (OIS + EIS)",
                        subtitle = "Stabilizzazione ottica ed elettronica su foto e video",
                        checked = state.stabilizationEnabled,
                        onChange = { state = state.copy(stabilizationEnabled = it) }
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    val lightPct = (state.brightnessBoost * 100).toInt()
                    Text("Luminosità foto: ${if (lightPct >= 0) "+$lightPct%" else "$lightPct%"}", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                    Slider(value = state.brightnessBoost, valueRange = -0.10f..0.30f, onValueChange = { state = state.copy(brightnessBoost = it) })

                    Text("Esposizione (EV): ${if (state.evCompensation >= 0) "+${state.evCompensation}" else "${state.evCompensation}"}", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                    Slider(value = state.evCompensation.toFloat(), valueRange = -2f..3f, onValueChange = { state = state.copy(evCompensation = it.toInt()) })

                    Text("Intensità Profilo (${state.profile.displayName}): ${(state.profileIntensity * 100).toInt()}%", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                    Slider(value = state.profileIntensity, valueRange = 0.2f..1.5f, onValueChange = { state = state.copy(profileIntensity = it) })

                    ToggleRow(
                        title = "Griglia 3x3",
                        subtitle = "Guide composizione sul mirino",
                        checked = state.gridEnabled,
                        onChange = { state = state.copy(gridEnabled = it) }
                    )

                    Spacer(modifier = Modifier.height(14.dp))
                    Button(
                        onClick = { sheet = SheetPage.NONE },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = state.profile.accentColor, contentColor = Color.Black)
                    ) {
                        Text("Chiudi", fontWeight = FontWeight.Bold)
                    }
                    Spacer(modifier = Modifier.height(20.dp))
                }
            }
        }
    }
}

@Composable
private fun ToggleRow(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 12.dp)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text(subtitle, fontSize = 12.sp, color = Color.White.copy(alpha = 0.65f))
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

private fun openOppoGallery(context: Context, lastUri: String?, onError: (String) -> Unit) {
    val pkgs = listOf("com.coloros.gallery3d", "com.oplus.gallery3d", "com.oneplus.gallery", "com.google.android.apps.photos")
    val target = lastUri?.let { Uri.parse(it) } ?: MediaStore.Images.Media.EXTERNAL_CONTENT_URI

    for (pkg in pkgs) {
        val launch = context.packageManager.getLaunchIntentForPackage(pkg)
        if (launch != null) {
            try {
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(target, "image/*")
                    setPackage(pkg)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(intent)
                return
            } catch (_: Exception) {
                try {
                    launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(launch)
                    return
                } catch (_: Exception) {
                }
            }
        }
    }

    try {
        val fallback = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(target, "image/*")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(fallback)
    } catch (_: Exception) {
        onError("Galleria non trovata")
    }
}
