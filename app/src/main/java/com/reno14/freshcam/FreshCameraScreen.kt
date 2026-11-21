package com.reno14.freshcam

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.filled.LocationOff
import androidx.compose.material.icons.filled.LocationOn
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
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.delay
import java.util.Locale
import kotlin.math.abs

private enum class SheetPage {
    NONE,
    PROFILE_FORMAT,
    FOCUS,
    SETTINGS
}

private fun resolveLensAndRatio(displayZoom: Float, currentLens: LensType, maxTeleRatio: Float): Pair<LensType, Float> {
    if (currentLens == LensType.FRONT) {
        return LensType.FRONT to displayZoom.coerceIn(1.0f, 2.0f)
    }
    return when {
        displayZoom < 0.95f -> LensType.ULTRAWIDE to 1.0f
        displayZoom < 3.45f -> LensType.MAIN_50MP to displayZoom.coerceIn(1.0f, 3.45f)
        else -> {
            val ratio = (displayZoom / 3.5f).coerceIn(1.0f, maxTeleRatio)
            LensType.TELE_50MP to ratio
        }
    }
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

    val locationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val granted = (perms[Manifest.permission.ACCESS_FINE_LOCATION] == true) ||
            (perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true)
        state = state.copy(
            hasLocationPermission = granted,
            locationTagEnabled = granted,
            bannerMessage = if (granted) "Posizione GPS attivata nelle foto" else "Permesso posizione negato"
        )
    }

    LaunchedEffect(Unit) {
        val lenses = engine.discoverLenses()
        val hasModel = engine.checkNeuralModelReady()
        val hasLoc = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        state = state.copy(
            detectedLenses = lenses,
            isNeuralModelReady = hasModel,
            loadedModelName = engine.getLoadedModelName(),
            hasLocationPermission = hasLoc
        )
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
                state = state.copy(maxLensZoomRatio = maxZ.coerceAtLeast(17.2f))
            },
            onBanner = { msg ->
                state = state.copy(bannerMessage = msg)
            }
        )
    }

    LaunchedEffect(state.lensZoomRatio) {
        engine.setZoom(state.lensZoomRatio)
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

                IconButton(
                    onClick = {
                        if (!state.locationTagEnabled) {
                            if (state.hasLocationPermission) {
                                state = state.copy(locationTagEnabled = true, bannerMessage = "Posizione GPS: ON")
                            } else {
                                locationPermissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.ACCESS_FINE_LOCATION,
                                        Manifest.permission.ACCESS_COARSE_LOCATION
                                    )
                                )
                            }
                        } else {
                            state = state.copy(locationTagEnabled = false, bannerMessage = "Posizione GPS: OFF")
                        }
                    }
                ) {
                    Icon(
                        imageVector = if (state.locationTagEnabled) Icons.Default.LocationOn else Icons.Default.LocationOff,
                        contentDescription = "Posizione GPS",
                        tint = if (state.locationTagEnabled) state.profile.accentColor else Color.White.copy(alpha = 0.75f)
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
                            val msg = if (nextSr) "Rete Neurale AI: ON (${state.aiMode.shortLabel})" else "Rete Neurale AI: OFF"
                            state = state.copy(neuralSuperResEnabled = nextSr, bannerMessage = msg)
                        }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(13.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = if (state.neuralSuperResEnabled) "AI: ON" else "AI: OFF",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                if (state.neuralSuperResEnabled) {
                    Spacer(modifier = Modifier.width(5.dp))
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .background(Color(0xFF1A2634).copy(alpha = 0.90f))
                            .border(1.dp, Color(0xFF26C6DA), RoundedCornerShape(16.dp))
                            .clickable {
                                val nextLevel = when (state.aiMode) {
                                    AiMode.LOW -> AiMode.MEDIUM
                                    AiMode.MEDIUM -> AiMode.AGGRESSIVE
                                    AiMode.AGGRESSIVE -> AiMode.LOW
                                }
                                state = state.copy(aiMode = nextLevel, bannerMessage = "Livello Rete Neurale: ${nextLevel.shortLabel}")
                            }
                            .padding(horizontal = 9.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = state.aiMode.shortLabel,
                            color = Color(0xFF80DEEA),
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
            if (state.activeLens != LensType.FRONT) {
                val maxGlobalZoom = 60.0f
                val maxTeleRatio = maxGlobalZoom / 3.5f
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "${String.format(Locale.US, "%.1f", state.displayZoom)}x",
                        color = state.profile.accentColor,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.width(52.dp)
                    )
                    Slider(
                        value = state.displayZoom.coerceIn(0.6f, maxGlobalZoom),
                        valueRange = 0.6f..maxGlobalZoom,
                        onValueChange = { targetDisplay ->
                            val (resolvedLens, resolvedRatio) = resolveLensAndRatio(
                                displayZoom = targetDisplay,
                                currentLens = state.activeLens,
                                maxTeleRatio = maxTeleRatio
                            )
                            state = state.copy(
                                displayZoom = targetDisplay,
                                activeLens = resolvedLens,
                                lensZoomRatio = resolvedRatio
                            )
                        },
                        modifier = Modifier.weight(1f),
                        colors = SliderDefaults.colors(
                            thumbColor = state.profile.accentColor,
                            activeTrackColor = state.profile.accentColor
                        )
                    )
                    Text(
                        text = if (state.neuralSuperResEnabled) "NN ${state.aiMode.shortLabel.uppercase()}" else "AI OFF",
                        color = if (state.neuralSuperResEnabled) Color(0xFF26C6DA) else Color.White.copy(alpha = 0.65f),
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(start = 6.dp)
                    )
                }
            }

            val presets = listOf(
                Pair("0.6x", 0.6f),
                Pair("1x", 1.0f),
                Pair("3.5x", 3.5f),
                Pair("10x", 10.0f),
                Pair("30x", 30.0f),
                Pair("60x", 60.0f)
            )
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                presets.forEach { (label, targetDisplayZoom) ->
                    val selected = state.activeLens != LensType.FRONT && abs(state.displayZoom - targetDisplayZoom) < 0.45f
                    Box(
                        modifier = Modifier
                            .padding(horizontal = 3.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(if (selected) state.profile.accentColor else Color.DarkGray.copy(alpha = 0.72f))
                            .clickable {
                                val (resolvedLens, resolvedRatio) = resolveLensAndRatio(
                                    displayZoom = targetDisplayZoom,
                                    currentLens = LensType.MAIN_50MP,
                                    maxTeleRatio = 60.0f / 3.5f
                                )
                                state = state.copy(
                                    displayZoom = targetDisplayZoom,
                                    activeLens = resolvedLens,
                                    lensZoomRatio = resolvedRatio,
                                    bannerMessage = "${resolvedLens.label} ($label)"
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
                                if (mode == ShootingMode.TELE_50MP) {
                                    state = state.copy(
                                        shootingMode = mode,
                                        activeLens = LensType.TELE_50MP,
                                        displayZoom = 3.5f,
                                        lensZoomRatio = 1.0f,
                                        bannerMessage = "Modalità Tele 50MP (3.5x)"
                                    )
                                } else {
                                    state = state.copy(shootingMode = mode, bannerMessage = "Modalità ${mode.label}")
                                }
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
                            if (state.activeLens == LensType.FRONT) {
                                state = state.copy(
                                    activeLens = LensType.MAIN_50MP,
                                    displayZoom = 1.0f,
                                    lensZoomRatio = 1.0f,
                                    bannerMessage = LensType.MAIN_50MP.label
                                )
                            } else {
                                state = state.copy(
                                    activeLens = LensType.FRONT,
                                    displayZoom = 1.0f,
                                    lensZoomRatio = 1.0f,
                                    bannerMessage = LensType.FRONT.label
                                )
                            }
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
                        text = if (state.isNeuralModelReady) "Rete Neurale: ${state.loadedModelName} attiva (1x → 60x)" else "Attenzione: file .tflite non rilevato",
                        fontSize = 12.sp,
                        color = if (state.isNeuralModelReady) Color(0xFF4FC3F7) else Color(0xFFFF7043)
                    )

                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(8.dp))

                    ToggleRow(
                        title = "Rete Neurale AI (ON / OFF)",
                        subtitle = "Discreta su 1x, forte su zoom spinto fino a 60x con ancoraggio colori anti-artefatti",
                        checked = state.neuralSuperResEnabled,
                        onChange = { state = state.copy(neuralSuperResEnabled = it) }
                    )

                    if (state.neuralSuperResEnabled) {
                        Text("Intensità Rete Neurale: ${state.aiMode.shortLabel}", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF80DEEA))
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            AiMode.entries.forEach { mode ->
                                FilterChip(
                                    selected = state.aiMode == mode,
                                    onClick = { state = state.copy(aiMode = mode) },
                                    label = { Text(mode.shortLabel) }
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                    }

                    ToggleRow(
                        title = "Informazioni di Posizione (GPS EXIF)",
                        subtitle = "Salva le coordinate geografiche nei metadati di foto e video",
                        checked = state.locationTagEnabled,
                        onChange = { enable ->
                            if (enable && !state.hasLocationPermission) {
                                locationPermissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.ACCESS_FINE_LOCATION,
                                        Manifest.permission.ACCESS_COARSE_LOCATION
                                    )
                                )
                            } else {
                                state = state.copy(locationTagEnabled = enable)
                            }
                        }
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
