package com.reno14.freshcam

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.FlashAuto
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sin

private enum class SheetPage {
    NONE,
    AI_SELECTOR,
    PROFILE_FORMAT,
    FOCUS,
    SETTINGS
}

private fun resolveLensAndRatio(displayZoom: Float, currentLens: LensType): Pair<LensType, Float> {
    if (currentLens == LensType.FRONT) {
        return LensType.FRONT to displayZoom.coerceIn(1.0f, 2.0f)
    }
    return when {
        displayZoom < 0.95f -> LensType.ULTRAWIDE to 1.0f
        displayZoom < 3.45f -> LensType.MAIN_50MP to displayZoom.coerceIn(1.0f, 3.45f)
        else -> {
            val ratio = (displayZoom / 3.5f).coerceIn(1.0f, 120.0f / 3.5f)
            LensType.TELE_50MP to ratio
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FreshCameraScreen(lifecycleOwner: LifecycleOwner) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf(FreshCamState()) }
    var sheet by remember { mutableStateOf(SheetPage.NONE) }
    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var focusRing by remember { mutableStateOf<Offset?>(null) }
    var flashPulse by remember { mutableStateOf(false) }

    var showZoomDial by remember { mutableStateOf(false) }
    var dialHideJob by remember { mutableStateOf<Job?>(null) }
    var lastThumbBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val engine = remember(context, lifecycleOwner) { FreshCameraEngine(context, lifecycleOwner) }

    fun pingZoomDial() {
        showZoomDial = true
        dialHideJob?.cancel()
        dialHideJob = scope.launch {
            delay(2500)
            showZoomDial = false
        }
    }

    val locationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val granted = (perms[Manifest.permission.ACCESS_FINE_LOCATION] == true) ||
            (perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true)
        state = state.copy(
            hasLocationPermission = granted,
            locationTagEnabled = granted,
            bannerMessage = if (granted) "Posizione GPS attivata" else "Permesso posizione negato"
        )
    }

    LaunchedEffect(state.lastCapturedUri) {
        val uriStr = state.lastCapturedUri
        val loaded = withContext(Dispatchers.IO) {
            loadThumbnailBitmap(context, uriStr)
        }
        if (loaded != null) {
            lastThumbBitmap = loaded
        }
    }

    LaunchedEffect(Unit) {
        val (lenses, sensorReport) = withContext(Dispatchers.IO) {
            val l = engine.discoverLenses()
            val r = engine.inspectSensorCapabilities()
            l to r
        }
        val hasLoc = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        state = state.copy(
            detectedLenses = lenses,
            sensorCapabilitiesReport = sensorReport,
            hasLocationPermission = hasLoc
        )
    }

    // Inizializza in background il modello neurale ONNX locale per mostrare subito lo stato reale (NNAPI/NPU vs CPU)
    LaunchedEffect(state.localNeuralModel, state.preferNnapiNpu) {
        val model = state.localNeuralModel
        val pref = state.preferNnapiNpu
        val status = withContext(Dispatchers.IO) {
            engine.probeLocalAiStatus(model, pref)
        }
        state = state.copy(localAiBackendStatus = status)
    }

    LaunchedEffect(state.bannerMessage) {
        if (state.bannerMessage != null) {
            delay(4200)
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
        state.aiPreset,
        state.forceLogicalHwExtensions,
        state.videoResolution,
        state.stabilizationEnabled,
        state.isoValue,
        state.shutterSpeedDenom,
        state.focusMode,
        state.manualFocusDistance,
        state.shootingMode
    ) {
        val pv = previewView ?: return@LaunchedEffect
        delay(90)
        engine.bind(
            previewView = pv,
            state = state,
            onZoomBounds = { _, maxZ ->
                state = state.copy(maxLensZoomRatio = maxZ.coerceAtLeast(10.0f))
            },
            onExtensionUpdate = { activeExt, report ->
                state = state.copy(
                    activeHardwareExtension = activeExt,
                    hwExtensionsReport = report
                )
            },
            onBanner = { msg ->
                state = state.copy(bannerMessage = msg)
            }
        )
    }

    LaunchedEffect(state.evCompensation) {
        if (state.captureFormat != CaptureFormat.PRO) {
            engine.applyExposureCompensation(state.evCompensation)
        }
    }

    LaunchedEffect(state.flashMode) {
        engine.applyFlashLive(state.flashMode)
    }

    LaunchedEffect(state.lensZoomRatio, state.displayZoom, state.activeHardwareExtension) {
        val targetZoom = if (state.activeHardwareExtension != null && state.forceLogicalHwExtensions) {
            state.displayZoom
        } else {
            state.lensZoomRatio
        }
        engine.setZoom(targetZoom)
    }

    val latestState by rememberUpdatedState(state)

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures { offset ->
                        focusRing = offset
                        val st = latestState
                        if (st.focusMode == FocusMode.AI_LOCK) {
                            engine.applyFocusPoint(offset.x, offset.y, size.width.toFloat(), size.height.toFloat(), keepLocked = true)
                            state = st.copy(isSubjectLocked = true, bannerMessage = "AI Lock agganciato")
                        } else if (st.focusMode == FocusMode.AUTO) {
                            engine.applyFocusPoint(offset.x, offset.y, size.width.toFloat(), size.height.toFloat(), keepLocked = false)
                            state = st.copy(isSubjectLocked = false, bannerMessage = "Autofocus sul punto")
                        }
                    }
                }
                .pointerInput(Unit) {
                    detectTransformGestures { _, _, zoomChange, _ ->
                        if (abs(zoomChange - 1f) > 0.008f && latestState.activeLens != LensType.FRONT) {
                            val st = latestState
                            val nextZoom = (st.displayZoom * zoomChange).coerceIn(0.6f, 120.0f)
                            val (resolvedLens, resolvedRatio) = resolveLensAndRatio(nextZoom, st.activeLens)
                            pingZoomDial()
                            state = st.copy(
                                displayZoom = nextZoom,
                                activeLens = resolvedLens,
                                lensZoomRatio = resolvedRatio
                            )
                        }
                    }
                }
        ) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FIT_CENTER
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

            if (state.countdownRemaining > 0) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "${state.countdownRemaining}",
                        color = Color.White,
                        fontSize = 92.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // BARRA SUPERIORE
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                modifier = Modifier
                    .weight(1f)
                    .horizontalScroll(rememberScrollState()),
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

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            if (state.timerMode != TimerMode.OFF) state.profile.accentColor.copy(alpha = 0.85f)
                            else Color.Black.copy(alpha = 0.60f)
                        )
                        .clickable {
                            val next = when (state.timerMode) {
                                TimerMode.OFF -> TimerMode.SEC_3
                                TimerMode.SEC_3 -> TimerMode.SEC_5
                                TimerMode.SEC_5 -> TimerMode.SEC_10
                                TimerMode.SEC_10 -> TimerMode.OFF
                            }
                            state = state.copy(timerMode = next, bannerMessage = "Autoscatto: ${next.label}")
                        }
                        .padding(horizontal = 8.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.Timer,
                            contentDescription = "Autoscatto",
                            tint = if (state.timerMode != TimerMode.OFF) Color.Black else Color.White,
                            modifier = Modifier.size(14.dp)
                        )
                        if (state.timerMode != TimerMode.OFF) {
                            Spacer(modifier = Modifier.width(3.dp))
                            Text(state.timerMode.label, color = Color.Black, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                Spacer(modifier = Modifier.width(6.dp))

                val rawInRamActiveNow = state.rawInRamEnabled &&
                    state.shootingMode == ShootingMode.PHOTO &&
                    state.captureFormat == CaptureFormat.NORMAL &&
                    state.activeHardwareExtension == null &&
                    (state.activeLens == LensType.MAIN_50MP || state.activeLens == LensType.TELE_50MP)

                val aiButtonText = when {
                    rawInRamActiveNow && state.aiPreset.isEnabled && !state.aiPreset.isCloud -> "RAW+${state.localNeuralModel.shortName}"
                    rawInRamActiveNow && state.aiPreset.isCloud -> "RAW+Cloud"
                    rawInRamActiveNow -> "RAW→JPG"
                    state.activeHardwareExtension != null -> state.activeHardwareExtension!!
                    !state.aiPreset.isEnabled -> "AI Off"
                    state.aiPreset.isCloud -> state.aiPreset.shortBadge
                    else -> "${state.localNeuralModel.shortName} ${state.aiPreset.shortBadge.removePrefix("AI ")}"
                }
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            when {
                                state.aiPreset.isCloud -> Color(0xFF7E57C2).copy(alpha = 0.92f)
                                rawInRamActiveNow && state.aiPreset.isEnabled -> Color(0xFF00838F).copy(alpha = 0.95f)
                                rawInRamActiveNow -> Color(0xFF00897B).copy(alpha = 0.94f)
                                state.activeHardwareExtension != null -> Color(0xFF2E7D32).copy(alpha = 0.92f)
                                !state.aiPreset.isEnabled -> Color.Black.copy(alpha = 0.65f)
                                else -> Color(0xFF0288D1).copy(alpha = 0.94f)
                            }
                        )
                        .border(
                            1.dp,
                            if (rawInRamActiveNow || state.aiPreset.isEnabled || state.activeHardwareExtension != null) Color.White else Color.White.copy(alpha = 0.35f),
                            RoundedCornerShape(16.dp)
                        )
                        .clickable { sheet = SheetPage.AI_SELECTOR }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(13.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = aiButtonText,
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
                        .padding(horizontal = 9.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = state.focusMode.shortLabel,
                        color = if (state.isSubjectLocked) Color(0xFFFFCA28) else state.profile.accentColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.width(6.dp))

            IconButton(
                onClick = { sheet = SheetPage.SETTINGS },
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF1A222D).copy(alpha = 0.90f))
                    .border(1.dp, Color.White.copy(alpha = 0.35f), CircleShape)
            ) {
                Icon(Icons.Default.Settings, contentDescription = "Impostazioni Foto e Video", tint = Color.White)
            }
        }

        // FIX v18: Indicatori superiori INDIPENDENTI! AI Lock NON nasconde più i banner di salvataggio/errore!
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(top = 56.dp, start = 12.dp, end = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (state.backgroundJobsCount > 0) {
                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color(0xFF004D40).copy(alpha = 0.92f))
                        .border(1.dp, Color(0xFF4DB6AC), RoundedCornerShape(20.dp))
                        .padding(horizontal = 12.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(
                        color = Color(0xFF80CBC4),
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(12.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    val label = if (state.backgroundJobsCount == 1) {
                        "AI & Multi-Frame in background (1 foto)..."
                    } else {
                        "AI & Multi-Frame in background (${state.backgroundJobsCount} foto)..."
                    }
                    Text(label, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                }
                Spacer(modifier = Modifier.height(6.dp))
            }

            if (state.isSubjectLocked) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color(0xFFFFCA28))
                        .clickable {
                            engine.clearFocusLock()
                            focusRing = null
                            state = state.copy(isSubjectLocked = false, bannerMessage = "Fuoco sbloccato")
                        }
                        .padding(horizontal = 14.dp, vertical = 5.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Lock, contentDescription = null, tint = Color.Black, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Oggetto bloccato · Tocca per sbloccare", color = Color.Black, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(modifier = Modifier.height(6.dp))
            }

            val activeStatusText = state.persistentStatus ?: state.bannerMessage
            if (activeStatusText != null) {
                val isError = activeStatusText.contains("ERR", ignoreCase = true) ||
                    activeStatusText.contains("Errore", ignoreCase = true) ||
                    activeStatusText.contains("inserisci", ignoreCase = true)
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(if (isError) Color(0xFF5D1313).copy(alpha = 0.94f) else Color(0xFF0E1621).copy(alpha = 0.92f))
                        .border(1.dp, if (isError) Color(0xFFFF5252) else Color(0xFF26C6DA).copy(alpha = 0.70f), RoundedCornerShape(20.dp))
                        .padding(horizontal = 14.dp, vertical = 7.dp)
                ) {
                    Text(activeStatusText, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }

        // DIALOG ESPLICITO SE CLOUD GRATUITO SELEZIONATO SENZA TOKEN/URL
        if (state.showCloudSetupDialog) {
            AlertDialog(
                onDismissRequest = { state = state.copy(showCloudSetupDialog = false) },
                containerColor = Color(0xFF1A1D22),
                titleContentColor = Color.White,
                textContentColor = Color.White.copy(alpha = 0.85f),
                title = { Text("Cloud Gratuito non configurato", fontWeight = FontWeight.Bold) },
                text = {
                    Text(
                        "Per usare il Cloud gratuito devi inserire il tuo token gratuito Hugging Face (hf_..., nessuna carta richiesta) oppure un URL di un tuo endpoint gratuito nelle Impostazioni.

In alternativa puoi usare la nuova AI Neurale Locale ONNX (100% offline e gratuita, già attiva!)."
                    )
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            state = state.copy(showCloudSetupDialog = false)
                            sheet = SheetPage.SETTINGS
                        }
                    ) {
                        Text("Apri Impostazioni", color = Color(0xFF4FC3F7), fontWeight = FontWeight.Bold)
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = {
                            state = state.copy(showCloudSetupDialog = false)
                        }
                    ) {
                        Text("Annulla", color = Color.White.copy(alpha = 0.7f))
                    }
                }
            )
        }

        // BLOCCO INFERIORE
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .background(Color.Black.copy(alpha = 0.72f))
                .padding(top = 6.dp, bottom = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (state.activeLens != LensType.FRONT) {
                AnimatedVisibility(
                    visible = showZoomDial,
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    RotatingZoomDial(
                        currentZoom = state.displayZoom,
                        accentColor = state.profile.accentColor,
                        onZoomChange = { newZoom ->
                            pingZoomDial()
                            val (resolvedLens, resolvedRatio) = resolveLensAndRatio(newZoom, state.activeLens)
                            state = state.copy(
                                displayZoom = newZoom,
                                activeLens = resolvedLens,
                                lensZoomRatio = resolvedRatio
                            )
                        }
                    )
                }

                val presets = listOf(
                    Pair("0.6x", 0.6f),
                    Pair("1x", 1.0f),
                    Pair("3.5x", 3.5f),
                    Pair("7x", 7.0f),
                    Pair("20x", 20.0f),
                    Pair("60x", 60.0f),
                    Pair("120x", 120.0f)
                )
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                        .padding(horizontal = 10.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    presets.forEach { (label, targetDisplayZoom) ->
                        val selected = abs(state.displayZoom - targetDisplayZoom) < (if (targetDisplayZoom >= 20f) 4f else 0.45f)
                        Box(
                            modifier = Modifier
                                .padding(horizontal = 3.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .background(if (selected) state.profile.accentColor else Color.DarkGray.copy(alpha = 0.72f))
                                .clickable {
                                    pingZoomDial()
                                    val (resolvedLens, resolvedRatio) = resolveLensAndRatio(targetDisplayZoom, LensType.MAIN_50MP)
                                    state = state.copy(
                                        displayZoom = targetDisplayZoom,
                                        activeLens = resolvedLens,
                                        lensZoomRatio = resolvedRatio
                                    )
                                }
                                .padding(horizontal = 9.dp, vertical = 5.dp)
                        ) {
                            Text(label, color = if (selected) Color.Black else Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(24.dp, Alignment.CenterHorizontally)
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
                                state = state.copy(shootingMode = mode, bannerMessage = "Modalità ${mode.label}")
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
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(Color.White.copy(alpha = 0.14f))
                        .border(1.5.dp, Color.White.copy(alpha = 0.6f), CircleShape)
                        .clickable {
                            openOppoGallery(context, state.lastCapturedUri) { err ->
                                state = state.copy(bannerMessage = err)
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    val bmp = lastThumbBitmap
                    if (bmp != null) {
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Ultima foto in Galleria",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize().clip(CircleShape)
                        )
                    } else {
                        Icon(Icons.Default.PhotoLibrary, contentDescription = "Galleria Oppo", tint = Color.White, modifier = Modifier.size(22.dp))
                    }
                }

                Box(
                    modifier = Modifier
                        .size(74.dp)
                        .clip(CircleShape)
                        .border(4.dp, if (state.isCapturing) Color.Gray else Color.White, CircleShape)
                        .padding(6.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                state.isCapturing -> Color.Gray
                                state.shootingMode == ShootingMode.VIDEO -> if (state.isRecordingVideo) Color.Red else Color(0xFFFF5252)
                                else -> state.profile.accentColor
                            }
                        )
                        .clickable(enabled = !state.isCapturing && state.countdownRemaining == 0) {
                            if (state.shootingMode == ShootingMode.VIDEO) {
                                engine.toggleVideo(
                                    state = state,
                                    onRecording = { rec -> state = state.copy(isRecordingVideo = rec) },
                                    onBanner = { msg -> state = state.copy(bannerMessage = msg) }
                                )
                            } else {
                                val triggerCapture = {
                                    flashPulse = true
                                    val snap = state.copy(isCapturing = true)
                                    state = snap
                                    engine.shootInstant(
                                        snapshot = snap,
                                        onStatus = { busyMsg ->
                                            state = state.copy(persistentStatus = busyMsg)
                                        },
                                        onCaptureUnlocked = {
                                            state = state.copy(isCapturing = false, persistentStatus = null)
                                        },
                                        onBackgroundQueueChanged = { count ->
                                            state = state.copy(backgroundJobsCount = count)
                                        },
                                        onBackendUpdate = { beStatus ->
                                            state = state.copy(localAiBackendStatus = beStatus)
                                        },
                                        onResult = { uri, msg ->
                                            state = state.copy(
                                                lastCapturedUri = uri ?: state.lastCapturedUri,
                                                bannerMessage = msg
                                            )
                                        }
                                    )
                                }

                                if (state.timerMode.seconds > 0) {
                                    scope.launch {
                                        for (sec in state.timerMode.seconds downTo 1) {
                                            state = state.copy(countdownRemaining = sec)
                                            delay(1000)
                                        }
                                        state = state.copy(countdownRemaining = 0)
                                        triggerCapture()
                                    }
                                } else {
                                    triggerCapture()
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
                        .size(48.dp)
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

        // MENU UNIFICATO AI NEURALE LOCALE & RAW-IN-RAM
        if (sheet == SheetPage.AI_SELECTOR) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp, vertical = 12.dp)
                ) {
                    Text("Vera AI Neurale Locale & RAW (v18)", fontSize = 17.sp, fontWeight = FontWeight.Bold)
                    Text(
                        text = "Motore ONNX: ${state.localAiBackendStatus}",
                        fontSize = 11.sp,
                        color = Color(0xFF81C784)
                    )
                    Spacer(modifier = Modifier.height(10.dp))

                    ToggleRow(
                        title = "Fusione RAW-in-RAM → JPEG leggero",
                        subtitle = "Fonde 4 RAW lineari in background prima dell'AI (salva SOLO un normale JPG leggero)",
                        checked = state.rawInRamEnabled,
                        onChange = { enabled ->
                            state = state.copy(
                                rawInRamEnabled = enabled,
                                bannerMessage = if (enabled) "RAW-in-RAM → JPEG attivato" else "RAW-in-RAM disattivato"
                            )
                        }
                    )

                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("Rete Neurale Offline (100% Locale · Gratis)", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF4FC3F7))
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        LocalNeuralModel.entries.forEach { m ->
                            FilterChip(
                                selected = state.localNeuralModel == m,
                                onClick = {
                                    state = state.copy(localNeuralModel = m, bannerMessage = "Modello AI: ${m.shortName}")
                                },
                                label = { Text(if (m == LocalNeuralModel.CLEAR_REALITY_SPAN) "SPAN 4x (Consigliato)" else "Real-ESRGAN v3") }
                            )
                        }
                    }
                    Text(
                        text = state.localNeuralModel.description,
                        fontSize = 11.sp,
                        color = Color.White.copy(alpha = 0.65f)
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    Text("Intensità AI Neurale Locale", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF4FC3F7))
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.horizontalScroll(rememberScrollState())) {
                        listOf(
                            UnifiedAiPreset.OFF,
                            UnifiedAiPreset.LOCAL_LOW,
                            UnifiedAiPreset.LOCAL_MED,
                            UnifiedAiPreset.LOCAL_HIGH
                        ).forEach { preset ->
                            FilterChip(
                                selected = state.aiPreset == preset,
                                onClick = {
                                    state = state.copy(aiPreset = preset, bannerMessage = preset.fullLabel)
                                    sheet = SheetPage.NONE
                                },
                                label = { Text(preset.shortBadge) }
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Cloud Gratuito Opzionale (Token gratuito HF)", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFFB39DDB))
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(UnifiedAiPreset.CLOUD_FREE_MED, UnifiedAiPreset.CLOUD_FREE_HIGH).forEach { preset ->
                            FilterChip(
                                selected = state.aiPreset == preset,
                                onClick = {
                                    val needsHfToken = state.cloudEndpointUrl.contains("huggingface.co") && state.cloudApiKey.isBlank()
                                    val missingUrl = state.cloudEndpointUrl.isBlank()
                                    if (needsHfToken || missingUrl) {
                                        sheet = SheetPage.NONE
                                        state = state.copy(showCloudSetupDialog = true)
                                    } else {
                                        state = state.copy(aiPreset = preset, bannerMessage = preset.fullLabel)
                                        sheet = SheetPage.NONE
                                    }
                                },
                                label = { Text(preset.shortBadge) }
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }

        // PROFILI COLORE & FORMATO
        if (sheet == SheetPage.PROFILE_FORMAT) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp)) {
                    Text("Tipo di scatto (JPEG leggero di default)", fontSize = 15.sp, fontWeight = FontWeight.Bold)
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
                        val isoIdx = PRO_ISO_STOPS.indexOf(state.isoValue).let { if (it >= 0) it else 5 }
                        val shutterIdx = PRO_SHUTTER_DENOMS.indexOf(state.shutterSpeedDenom).let { if (it >= 0) it else 6 }

                        Spacer(modifier = Modifier.height(10.dp))
                        Text("ISO ${state.isoValue}", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = state.profile.accentColor)
                        Slider(
                            value = isoIdx.toFloat(),
                            valueRange = 0f..(PRO_ISO_STOPS.size - 1).toFloat(),
                            steps = PRO_ISO_STOPS.size - 2,
                            onValueChange = { idx ->
                                val picked = PRO_ISO_STOPS[idx.roundToInt().coerceIn(0, PRO_ISO_STOPS.lastIndex)]
                                state = state.copy(isoValue = picked)
                            }
                        )
                        Text("Tempo 1/${state.shutterSpeedDenom}s", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = state.profile.accentColor)
                        Slider(
                            value = shutterIdx.toFloat(),
                            valueRange = 0f..(PRO_SHUTTER_DENOMS.size - 1).toFloat(),
                            steps = PRO_SHUTTER_DENOMS.size - 2,
                            onValueChange = { idx ->
                                val picked = PRO_SHUTTER_DENOMS[idx.roundToInt().coerceIn(0, PRO_SHUTTER_DENOMS.lastIndex)]
                                state = state.copy(shutterSpeedDenom = picked)
                            }
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

        // MESSA A FUOCO
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

        // IMPOSTAZIONI FOTO, VIDEO & AI NEURALE v18
        if (sheet == SheetPage.SETTINGS) {
            ModalBottomSheet(
                onDismissRequest = { sheet = SheetPage.NONE },
                containerColor = Color(0xFF141619),
                contentColor = Color.White
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp, vertical = 12.dp)
                ) {
                    Text("Impostazioni Foto & AI Neurale (v18)", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("AI Locale ONNX: ${state.localAiBackendStatus}", fontSize = 11.sp, color = Color(0xFFFFCA28))
                    Text("Estensioni HW: ${state.hwExtensionsReport}", fontSize = 11.sp, color = Color(0xFF81C784))
                    Text("Sensori: ${state.sensorCapabilitiesReport}", fontSize = 11.sp, color = Color(0xFF4FC3F7))

                    Spacer(modifier = Modifier.height(14.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("VERA AI NEURALE LOCALE (OFFLINE · GRATIS)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFFFFCA28))
                    Spacer(modifier = Modifier.height(6.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        LocalNeuralModel.entries.forEach { m ->
                            FilterChip(
                                selected = state.localNeuralModel == m,
                                onClick = { state = state.copy(localNeuralModel = m) },
                                label = { Text(m.displayName, fontSize = 12.sp) }
                            )
                        }
                    }
                    ToggleRow(
                        title = "Accelerazione Hardware NPU/DSP (Android NNAPI)",
                        subtitle = "Prova a delegare la rete neurale alla NPU Dimensity tramite NNAPI FP16 (con fallback automatico su CPU 4-Thread)",
                        checked = state.preferNnapiNpu,
                        onChange = { state = state.copy(preferNnapiNpu = it) }
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("MOTORE RAW & MULTI-FRAME IN BACKGROUND", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF4DB6AC))
                    ToggleRow(
                        title = "Fusione RAW-in-RAM → JPEG leggero",
                        subtitle = "Fonde 4 RAW grezzi su 1x e Tele 3.5x in spazio lineare e salva solo un normale JPG (zero DNG pesanti)",
                        checked = state.rawInRamEnabled,
                        onChange = { state = state.copy(rawInRamEnabled = it) }
                    )
                    ToggleRow(
                        title = "Elaborazione in Background",
                        subtitle = "Sblocca subito lo scatto mentre RAW e rete neurale completano la foto in coda",
                        checked = state.backgroundProcessingEnabled,
                        onChange = { state = state.copy(backgroundProcessingEnabled = it) }
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("ESTENSIONI HARDWARE (NOTTE / RITRATTO)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF81C784))
                    ToggleRow(
                        title = "Forza Camera Logica per Estensioni HW",
                        subtitle = "Usa la camera logica Oppo per HW Notte e HW Bokeh",
                        checked = state.forceLogicalHwExtensions,
                        onChange = { state = state.copy(forceLogicalHwExtensions = it) }
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("IMPOSTAZIONI VIDEO", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF80DEEA))
                    Spacer(modifier = Modifier.height(6.dp))
                    Text("Risoluzione Video", fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        VideoResolution.entries.forEach { res ->
                            FilterChip(
                                selected = state.videoResolution == res,
                                onClick = { state = state.copy(videoResolution = res, bannerMessage = "Video: ${res.label}") },
                                label = { Text(res.label) }
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("IMPOSTAZIONI FOTO & OTTICA", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF80DEEA))
                    Spacer(modifier = Modifier.height(6.dp))

                    ToggleRow(
                        title = "Stabilizzazione (OIS in Foto · OIS+EIS in Video)",
                        subtitle = "In modalità Foto usa solo OIS ottico puro senza ritaglio video EIS",
                        checked = state.stabilizationEnabled,
                        onChange = { state = state.copy(stabilizationEnabled = it) }
                    )

                    Text("Qualità JPEG: ${state.jpegQuality}%", fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    Slider(
                        value = state.jpegQuality.toFloat(),
                        valueRange = 85f..100f,
                        onValueChange = { state = state.copy(jpegQuality = it.toInt()) }
                    )

                    ToggleRow(
                        title = "Curva Tonale Cinematografica",
                        subtitle = "Aggiunge una leggera curva a S al salvataggio",
                        checked = state.postSavePolishEnabled,
                        onChange = { state = state.copy(postSavePolishEnabled = it) }
                    )

                    ToggleRow(
                        title = "Posizione GPS nei metadati (EXIF)",
                        subtitle = "Salva le coordinate geografiche in foto e video",
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
                        title = "Griglia 3x3",
                        subtitle = "Guide composizione sul mirino",
                        checked = state.gridEnabled,
                        onChange = { state = state.copy(gridEnabled = it) }
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                    val lightPct = (state.brightnessBoost * 100).toInt()
                    Text("Luminosità foto: ${if (lightPct >= 0) "+$lightPct%" else "$lightPct%"}", fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    Slider(value = state.brightnessBoost, valueRange = -0.15f..0.20f, onValueChange = { state = state.copy(brightnessBoost = it) })

                    Text("Esposizione (EV): ${if (state.evCompensation >= 0) "+${state.evCompensation}" else "${state.evCompensation}"}", fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    Slider(value = state.evCompensation.toFloat(), valueRange = -2f..3f, onValueChange = { state = state.copy(evCompensation = it.toInt()) })

                    Spacer(modifier = Modifier.height(10.dp))
                    HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(modifier = Modifier.height(10.dp))

                    Text("CLOUD GRATUITO (OPZIONALE · ZERO ABBONAMENTI)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFFB39DDB))
                    Spacer(modifier = Modifier.height(6.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        FreeCloudPreset.entries.forEach { preset ->
                            FilterChip(
                                selected = state.freeCloudPreset == preset,
                                onClick = {
                                    val newUrl = if (preset == FreeCloudPreset.CUSTOM_FREE) state.cloudEndpointUrl else preset.url
                                    state = state.copy(freeCloudPreset = preset, cloudEndpointUrl = newUrl)
                                },
                                label = { Text(preset.label, fontSize = 12.sp) }
                            )
                        }
                    }

                    OutlinedTextField(
                        value = state.cloudApiKey,
                        onValueChange = { state = state.copy(cloudApiKey = it) },
                        label = { Text("Token gratuito Hugging Face (hf_...)") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
                    )
                    OutlinedTextField(
                        value = state.cloudEndpointUrl,
                        onValueChange = { state = state.copy(cloudEndpointUrl = it, freeCloudPreset = FreeCloudPreset.CUSTOM_FREE) },
                        label = { Text("URL Endpoint Gratuito") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
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
private fun RotatingZoomDial(
    currentZoom: Float,
    accentColor: Color,
    onZoomChange: (Float) -> Unit
) {
    val minZ = 0.6f
    val maxZ = 120.0f
    val lnMin = ln(minZ.toDouble())
    val lnMax = ln(maxZ.toDouble())

    val latestZoom by rememberUpdatedState(currentZoom)
    val latestOnChange by rememberUpdatedState(onZoomChange)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(86.dp)
            .pointerInput(Unit) {
                detectHorizontalDragGestures { change, dragAmount ->
                    change.consume()
                    val cur = latestZoom.coerceIn(minZ, maxZ)
                    val norm = ((ln(cur.toDouble()) - lnMin) / (lnMax - lnMin)).toFloat()
                    val delta = -dragAmount / 320f
                    val nextNorm = (norm + delta).coerceIn(0f, 1f)
                    val nextZoom = exp(lnMin + nextNorm * (lnMax - lnMin)).toFloat().coerceIn(minZ, maxZ)
                    latestOnChange(nextZoom)
                }
            },
        contentAlignment = Alignment.BottomCenter
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cx = size.width / 2f
            val radius = size.width * 0.72f
            val cy = radius + 28f

            val norm = ((ln(currentZoom.coerceIn(minZ, maxZ).toDouble()) - lnMin) / (lnMax - lnMin)).toFloat()
            val phase = (norm * 44f) % 2.2f

            for (i in -20..20) {
                val angleDeg = i * 2.2f - phase
                val rad = (angleDeg - 90.0) * PI / 180.0
                val isMajor = (i % 5 == 0)
                val innerR = radius - if (isMajor) 24f else 13f
                val start = Offset(cx + (radius * cos(rad)).toFloat(), cy + (radius * sin(rad)).toFloat())
                val end = Offset(cx + (innerR * cos(rad)).toFloat(), cy + (innerR * sin(rad)).toFloat())
                drawLine(
                    color = if (isMajor) Color.White.copy(alpha = 0.90f) else Color.White.copy(alpha = 0.35f),
                    start = start,
                    end = end,
                    strokeWidth = if (isMajor) 2.5f else 1.5f,
                    cap = StrokeCap.Round
                )
            }

            drawLine(
                color = accentColor,
                start = Offset(cx, 22f),
                end = Offset(cx, 54f),
                strokeWidth = 4f,
                cap = StrokeCap.Round
            )
        }

        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black.copy(alpha = 0.78f))
                .border(1.dp, accentColor, RoundedCornerShape(12.dp))
                .padding(horizontal = 12.dp, vertical = 2.dp)
        ) {
            Text(
                text = "${String.format(Locale.US, "%.1f", currentZoom)}x",
                color = accentColor,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

private fun loadThumbnailBitmap(context: Context, uriString: String?): Bitmap? {
    return try {
        val targetUri: Uri? = if (uriString != null) {
            Uri.parse(uriString)
        } else {
            val projection = arrayOf(MediaStore.Images.Media._ID)
            val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"
            context.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                null,
                null,
                sortOrder
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val id = cursor.getLong(0)
                    Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id.toString())
                } else null
            }
        }
        if (targetUri == null) return null
        val opts = BitmapFactory.Options().apply { inSampleSize = 8 }
        context.contentResolver.openInputStream(targetUri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, opts)
        }
    } catch (_: Throwable) {
        null
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
