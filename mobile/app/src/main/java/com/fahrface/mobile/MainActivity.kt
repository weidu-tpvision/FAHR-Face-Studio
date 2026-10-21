package com.fahrface.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random

private val BgObsidian = Color(0xFF0B0F19)
private val SurfaceSlate = Color(0xFF111827)
private val CardSlate = Color(0xFF1E293B)
private val BorderSubtle = Color(0xFF26334D)
private val SkyAccent = Color(0xFF38BDF8)
private val IndigoAccent = Color(0xFF6366F1)
private val EmeraldGood = Color(0xFF34D399)
private val AmberWarn = Color(0xFFFBBF24)
private val TextPrimary = Color(0xFFF1F5F9)
private val TextSecondary = Color(0xFF94A3B8)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    background = BgObsidian,
                    surface = SurfaceSlate,
                    primary = SkyAccent,
                    secondary = IndigoAccent
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = BgObsidian
                ) {
                    FAHRFaceMobileApp()
                }
            }
        }
    }
}

@Composable
fun FAHRFaceMobileApp() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedTab by remember { mutableIntStateOf(0) }
    var isLoading by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf<String?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var analysisResult by remember { mutableStateOf<AnalyzeResult?>(null) }
    var selectedFaceIdx by remember { mutableIntStateOf(0) }
    var chronoAgeText by remember { mutableStateOf("42") }
    var showLandmarks by remember { mutableStateOf(true) }
    var selectedDetector by remember { mutableStateOf("retinaface") }
    var lastAnalyzedBytes by remember { mutableStateOf<ByteArray?>(null) }

    // Tab 2 state
    var heatmapOpacity by remember { mutableFloatStateOf(0.55f) }
    var showPatchGrid by remember { mutableStateOf(true) }

    // Tab 3 state
    var maeResult by remember { mutableStateOf<MaeResult?>(null) }
    var isMaeLoading by remember { mutableStateOf(false) }

    val activeFace = analysisResult?.faces?.getOrNull(selectedFaceIdx)

    fun triggerAnalysis(bytes: ByteArray, detector: String = selectedDetector) {
        lastAnalyzedBytes = bytes
        scope.launch {
            isLoading = true
            errorMessage = null
            val detLabel = if (detector == "retinaface") "RetinaFace (118M)" else "YuNet (Fast)"
            statusText = "Running On-Device AI ($detLabel + FAHR-FaceAge + MAE)..."
            try {
                val res = OnDeviceAiEngine.analyzePhotoOnDevice(context, bytes, detectorBackend = detector)
                analysisResult = res
                selectedFaceIdx = 0
                if (res.faces.isNotEmpty()) {
                    maeResult = OnDeviceAiEngine.runMaeOnDevice(context, 0, 42)
                }
            } catch (e: Exception) {
                errorMessage = e.message ?: "Analysis failed."
            } finally {
                isLoading = false
                statusText = null
            }
        }
    }

    fun triggerMae(seed: Int = Random.nextInt(100000)) {
        val face = activeFace ?: return
        scope.launch {
            isMaeLoading = true
            try {
                maeResult = OnDeviceAiEngine.runMaeOnDevice(context, face.faceId, seed)
            } catch (e: Exception) {
                errorMessage = "MAE error: ${e.message}"
            } finally {
                isMaeLoading = false
            }
        }
    }

    // Automatically run the bundled sample portrait on first launch
    LaunchedEffect(Unit) {
        try {
            val sampleBytes = OnDeviceAiEngine.loadBundledSampleBytes(context)
            triggerAnalysis(sampleBytes)
        } catch (_: Exception) {
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bmp ->
        if (bmp != null) {
            triggerAnalysis(OnDeviceAiEngine.bitmapToJpegBytes(bmp))
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                triggerAnalysis(bytes)
            }
        }
    }

    Scaffold(
        containerColor = BgObsidian,
        topBar = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(SurfaceSlate)
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(34.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(
                                    Brush.linearGradient(listOf(SkyAccent, IndigoAccent))
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "FF",
                                color = Color.White,
                                fontWeight = FontWeight.ExtraBold,
                                fontSize = 14.sp
                            )
                        }
                        Spacer(modifier = Modifier.width(10.dp))
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    text = "FAHR-Face",
                                    color = TextPrimary,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 15.sp
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "On-Device AI",
                                    color = EmeraldGood,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(6.dp))
                                        .background(EmeraldGood.copy(alpha = 0.16f))
                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }
                            Text(
                                text = "AIM-Harvard · Standalone Pixel 9 Edition",
                                color = TextSecondary,
                                fontSize = 11.sp
                            )
                        }
                    }

                    OutlinedButton(
                        onClick = {
                            scope.launch {
                                isLoading = true
                                errorMessage = null
                                try {
                                    val sampleBytes = OnDeviceAiEngine.loadBundledSampleBytes(context)
                                    triggerAnalysis(sampleBytes)
                                } catch (e: Exception) {
                                    isLoading = false
                                    errorMessage = e.message ?: "Failed to load sample portrait."
                                }
                            }
                        },
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        border = BorderStroke(1.dp, SkyAccent.copy(alpha = 0.4f)),
                        shape = RoundedCornerShape(10.dp),
                        modifier = Modifier.height(32.dp)
                    ) {
                        Text("✨ Sample", color = SkyAccent, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        },
        bottomBar = {
            NavigationBar(
                containerColor = Color(0xFF080B12),
                tonalElevation = 8.dp
            ) {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Text("👤", fontSize = 16.sp) },
                    label = { Text("Scan & Age", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Text("🔥", fontSize = 16.sp) },
                    label = { Text("AI Heatmap", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) }
                )
                NavigationBarItem(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    icon = { Text("🧩", fontSize = 16.sp) },
                    label = { Text("MAE Rebuild", fontSize = 11.sp, fontWeight = FontWeight.SemiBold) }
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (isLoading) {
                LinearProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                    color = SkyAccent,
                    trackColor = CardSlate
                )
                statusText?.let { st ->
                    Text(
                        text = st,
                        color = SkyAccent,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            errorMessage?.let { msg ->
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF4C0519)),
                    border = BorderStroke(1.dp, Color(0xFFF43F5E).copy(alpha = 0.5f)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(10.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = msg,
                            color = Color(0xFFFECDD3),
                            fontSize = 11.sp,
                            modifier = Modifier.weight(1f)
                        )
                        TextButton(onClick = { errorMessage = null }) {
                            Text("✕", color = Color.White)
                        }
                    }
                }
            }

            when (selectedTab) {
                0 -> ScanAndAgeTab(
                    analysisResult = analysisResult,
                    activeFace = activeFace,
                    selectedFaceIdx = selectedFaceIdx,
                    onSelectFace = { idx ->
                        selectedFaceIdx = idx
                        triggerMae(42)
                    },
                    chronoAgeText = chronoAgeText,
                    onChronoAgeChange = { chronoAgeText = it },
                    showLandmarks = showLandmarks,
                    onToggleLandmarks = { showLandmarks = it },
                    selectedDetector = selectedDetector,
                    onSelectDetector = { det ->
                        selectedDetector = det
                        lastAnalyzedBytes?.let { triggerAnalysis(it, det) }
                    },
                    onTakeSelfie = { cameraLauncher.launch(null) },
                    onPickGallery = { galleryLauncher.launch("image/*") }
                )

                1 -> AttentionHeatmapTab(
                    activeFace = activeFace,
                    opacity = heatmapOpacity,
                    onOpacityChange = { heatmapOpacity = it },
                    showGrid = showPatchGrid,
                    onToggleGrid = { showPatchGrid = it }
                )

                2 -> MaeRebuildTab(
                    maeResult = maeResult,
                    maskRatio = 0.75f,
                    onMaskRatioChange = {},
                    useOnDeviceAi = true,
                    isLoading = isMaeLoading,
                    onReroll = { triggerMae(Random.nextInt(100000)) }
                )
            }
        }
    }
}

@Composable
fun ScanAndAgeTab(
    analysisResult: AnalyzeResult?,
    activeFace: FaceAnalysis?,
    selectedFaceIdx: Int,
    onSelectFace: (Int) -> Unit,
    chronoAgeText: String,
    onChronoAgeChange: (String) -> Unit,
    showLandmarks: Boolean,
    onToggleLandmarks: (Boolean) -> Unit,
    selectedDetector: String,
    onSelectDetector: (String) -> Unit,
    onTakeSelfie: () -> Unit,
    onPickGallery: () -> Unit
) {
    // Action Row: Take Selfie & Photo Gallery
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Button(
            onClick = onTakeSelfie,
            modifier = Modifier
                .weight(1f)
                .height(44.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = SkyAccent)
        ) {
            Text("📷 Take Selfie", color = BgObsidian, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }

        OutlinedButton(
            onClick = onPickGallery,
            modifier = Modifier
                .weight(1f)
                .height(44.dp),
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(1.dp, BorderSubtle)
        ) {
            Text("🖼️ Photo Gallery", color = TextPrimary, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
        }
    }

    // Detector Model Switcher (RetinaFace vs YuNet)
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("Detector:", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        FilterChip(
            selected = selectedDetector == "retinaface",
            onClick = { onSelectDetector("retinaface") },
            label = { Text("RetinaFace (Upstream)", fontSize = 11.sp, fontWeight = FontWeight.Medium) },
            colors = FilterChipDefaults.filterChipColors(
                selectedContainerColor = SkyAccent.copy(alpha = 0.2f),
                selectedLabelColor = SkyAccent
            )
        )
        FilterChip(
            selected = selectedDetector == "yunet",
            onClick = { onSelectDetector("yunet") },
            label = { Text("YuNet (Fast)", fontSize = 11.sp, fontWeight = FontWeight.Medium) },
            colors = FilterChipDefaults.filterChipColors(
                selectedContainerColor = SkyAccent.copy(alpha = 0.2f),
                selectedLabelColor = SkyAccent
            )
        )
    }

    // Portrait & Landmarks Canvas Card
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceSlate),
        border = BorderStroke(1.dp, BorderSubtle),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(270.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xFF06090F)),
                contentAlignment = Alignment.Center
            ) {
                val bmp = analysisResult?.originalBitmap
                if (bmp != null) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val scale = min(
                            size.width / analysisResult.imageWidth.toFloat(),
                            size.height / analysisResult.imageHeight.toFloat()
                        )
                        val drawW = analysisResult.imageWidth * scale
                        val drawH = analysisResult.imageHeight * scale
                        val offX = (size.width - drawW) / 2f
                        val offY = (size.height - drawH) / 2f

                        drawImage(
                            image = bmp,
                            dstOffset = IntOffset(offX.roundToInt(), offY.roundToInt()),
                            dstSize = IntSize(drawW.roundToInt(), drawH.roundToInt())
                        )

                        analysisResult.faces.forEachIndexed { idx, f ->
                            val isSel = idx == selectedFaceIdx
                            val bx = offX + f.bbox[0] * scale
                            val by = offY + f.bbox[1] * scale
                            val bw = f.bbox[2] * scale
                            val bh = f.bbox[3] * scale

                            drawRect(
                                color = if (isSel) SkyAccent else Color.LightGray,
                                topLeft = Offset(bx, by),
                                size = Size(bw, bh),
                                style = Stroke(width = if (isSel) 6f else 3f)
                            )

                            if (showLandmarks && f.landmarks != null) {
                                val lm = f.landmarks
                                val re = Offset(offX + lm.rightEye.first * scale, offY + lm.rightEye.second * scale)
                                val le = Offset(offX + lm.leftEye.first * scale, offY + lm.leftEye.second * scale)
                                val nose = Offset(offX + lm.nose.first * scale, offY + lm.nose.second * scale)
                                val rm = Offset(offX + lm.rightMouth.first * scale, offY + lm.rightMouth.second * scale)
                                val lmPt = Offset(offX + lm.leftMouth.first * scale, offY + lm.leftMouth.second * scale)

                                drawLine(color = SkyAccent, start = re, end = le, strokeWidth = 3.5f)
                                drawCircle(color = SkyAccent, radius = 6f, center = re)
                                drawCircle(color = SkyAccent, radius = 6f, center = le)
                                drawCircle(color = AmberWarn, radius = 6f, center = nose)
                                drawCircle(color = Color(0xFFF472B6), radius = 6f, center = rm)
                                drawCircle(color = Color(0xFFF472B6), radius = 6f, center = lmPt)
                            }
                        }
                    }
                } else {
                    Text(
                        text = "Tap 'Take Selfie', 'Photo Gallery', or '✨ Sample' above to analyze a portrait.",
                        color = TextSecondary,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(24.dp)
                    )
                }

                // Landmarks toggle chip
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(8.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(BgObsidian.copy(alpha = 0.85f))
                        .clickable { onToggleLandmarks(!showLandmarks) }
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = if (showLandmarks) "☑ 5-Pt Landmarks" else "☐ 5-Pt Landmarks",
                        color = TextPrimary,
                        fontSize = 10.sp
                    )
                }
            }

            // Multi-face selector chips if > 1 face
            if ((analysisResult?.faces?.size ?: 0) > 1) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    analysisResult?.faces?.forEachIndexed { idx, f ->
                        FilterChip(
                            selected = idx == selectedFaceIdx,
                            onClick = { onSelectFace(idx) },
                            label = {
                                Text(
                                    "Face #${idx + 1} (${String.format("%.1f", f.predictedAge)}y)",
                                    fontSize = 11.sp
                                )
                            }
                        )
                    }
                }
            }
        }
    }

    // Concise FAHR-FaceAge Biomarker Card
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceSlate),
        border = BorderStroke(1.dp, BorderSubtle),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "FAHR-FACEAGE BIOMARKER",
                    color = TextSecondary,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = if (activeFace != null) "${activeFace.inferenceMs} ms (${activeFace.device.uppercase()})" else "Ready",
                    color = TextSecondary,
                    fontSize = 10.sp
                )
            }

            val predAge = activeFace?.predictedAge
            val actualAge = chronoAgeText.toFloatOrNull()
            val ageGap = if (predAge != null && actualAge != null && actualAge > 0f) predAge - actualAge else null

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Predicted FaceAge
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .background(CardSlate.copy(alpha = 0.6f))
                        .padding(10.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("FaceAge", color = TextSecondary, fontSize = 10.sp)
                    Text(
                        text = if (predAge != null) String.format("%.1f", predAge) else "—",
                        color = SkyAccent,
                        fontSize = 24.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text("years", color = TextSecondary, fontSize = 10.sp)
                }

                // Actual Chronological Age
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .background(CardSlate.copy(alpha = 0.6f))
                        .padding(10.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("Actual Age", color = TextSecondary, fontSize = 10.sp)
                    OutlinedTextField(
                        value = chronoAgeText,
                        onValueChange = onChronoAgeChange,
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier
                            .width(72.dp)
                            .height(44.dp),
                        textStyle = LocalTextStyle.current.copy(
                            color = TextPrimary,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            textAlign = TextAlign.Center
                        )
                    )
                }

                // Age Gap (ΔAge)
                val gapColor = when {
                    ageGap == null -> TextSecondary
                    ageGap <= -2.5f -> EmeraldGood
                    ageGap >= 2.5f -> AmberWarn
                    else -> SkyAccent
                }
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .background(CardSlate.copy(alpha = 0.6f))
                        .padding(10.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("Age Gap (Δ)", color = TextSecondary, fontSize = 10.sp)
                    Text(
                        text = if (ageGap != null) "${if (ageGap > 0) "+" else ""}${String.format("%.1f", ageGap)}" else "—",
                        color = gapColor,
                        fontSize = 24.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text("years", color = TextSecondary, fontSize = 10.sp)
                }
            }

            // Compact Telemetry Footer
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = if (activeFace != null) "Det: ${activeFace.detectorBackend.uppercase()}" else "Det: —",
                    color = SkyAccent,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = if (activeFace != null) "Conf: ${String.format("%.1f%%", activeFace.confidence * 100f)}" else "Conf: —",
                    color = TextSecondary,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = if (activeFace != null) "Tilt: ${String.format("%+.2f°", activeFace.alignmentAngleDeg)}" else "Tilt: —",
                    color = TextSecondary,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = if (activeFace?.interocularPx != null) "IOD: ${activeFace.interocularPx}px" else "IOD: —",
                    color = TextSecondary,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    }
}

@Composable
fun AttentionHeatmapTab(
    activeFace: FaceAnalysis?,
    opacity: Float,
    onOpacityChange: (Float) -> Unit,
    showGrid: Boolean,
    onToggleGrid: (Boolean) -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceSlate),
        border = BorderStroke(1.dp, BorderSubtle),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("12L × 12H VIT ATTENTION ROLLOUT", color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    Text("Facial patches driving CLS age regression", color = TextSecondary, fontSize = 11.sp)
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.clickable { onToggleGrid(!showGrid) }
                ) {
                    Checkbox(checked = showGrid, onCheckedChange = onToggleGrid)
                    Text("7×7 Grid", color = TextPrimary, fontSize = 11.sp)
                }
            }

            Box(
                modifier = Modifier
                    .size(220.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.Black)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center
            ) {
                if (activeFace?.alignedBitmap != null) {
                    Image(
                        bitmap = activeFace.alignedBitmap,
                        contentDescription = "Aligned 112x112 Crop",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                    if (activeFace.heatmapBitmap != null) {
                        Image(
                            bitmap = activeFace.heatmapBitmap,
                            contentDescription = "Attention Heatmap",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .fillMaxSize()
                                .alpha(opacity)
                        )
                    }
                    if (showGrid) {
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val step = size.width / 7f
                            for (i in 1..6) {
                                val pos = i * step
                                drawLine(
                                    color = Color.White.copy(alpha = 0.3f),
                                    start = Offset(pos, 0f),
                                    end = Offset(pos, size.height),
                                    strokeWidth = 1.5f
                                )
                                drawLine(
                                    color = Color.White.copy(alpha = 0.3f),
                                    start = Offset(0f, pos),
                                    end = Offset(size.width, pos),
                                    strokeWidth = 1.5f
                                )
                            }
                        }
                    }
                } else {
                    Text("Scan a face on Tab 1 first", color = TextSecondary, fontSize = 12.sp)
                }
            }

            Column(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Aligned Crop (112×112)", color = TextSecondary, fontSize = 11.sp)
                    Text("Heatmap: ${(opacity * 100).roundToInt()}%", color = SkyAccent, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
                Slider(
                    value = opacity,
                    onValueChange = onOpacityChange,
                    valueRange = 0f..1f
                )
            }
        }
    }

    // Anatomical Zone Attribution Bars
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceSlate),
        border = BorderStroke(1.dp, BorderSubtle),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("ANATOMICAL ZONE ATTRIBUTION", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            if (activeFace != null) {
                activeFace.regionalAttribution.forEach { (zone, pct) ->
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(zone, color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                            Text("${String.format("%.1f", pct)}%", color = SkyAccent, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                        }
                        LinearProgressIndicator(
                            progress = { (pct / 45f).coerceIn(0f, 1f) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(7.dp)
                                .clip(CircleShape),
                            color = SkyAccent,
                            trackColor = CardSlate
                        )
                    }
                }
            } else {
                Text("No face analyzed yet.", color = TextSecondary, fontSize = 12.sp)
            }
        }
    }
}

@Composable
fun MaeRebuildTab(
    maeResult: MaeResult?,
    maskRatio: Float,
    onMaskRatioChange: (Float) -> Unit,
    useOnDeviceAi: Boolean,
    isLoading: Boolean,
    onReroll: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceSlate),
        border = BorderStroke(1.dp, BorderSubtle),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("MAE PATCH RECONSTRUCTION", color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    Text("fahr_face · WebFace42M Foundation Model", color = TextSecondary, fontSize = 11.sp)
                }
                Button(
                    onClick = onReroll,
                    enabled = !isLoading && maeResult != null,
                    colors = ButtonDefaults.buttonColors(containerColor = IndigoAccent),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Text(if (isLoading) "..." else "🎲 Re-Mask", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }

            Column {
                val pct = (maskRatio * 100).roundToInt()
                val maskedPatches = (maskRatio * 49).roundToInt()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        if (useOnDeviceAi) "On-Device ONNX Masking" else "Mask Ratio",
                        color = TextSecondary,
                        fontSize = 11.sp
                    )
                    Text("$pct% ($maskedPatches/49 patches)", color = IndigoAccent, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
                if (!useOnDeviceAi) {
                    Slider(
                        value = maskRatio,
                        onValueChange = onMaskRatioChange,
                        valueRange = 0.25f..0.90f
                    )
                }
            }

            // 2x2 Grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                MaePanelCard(
                    title = "1. Aligned 112×112",
                    subtitle = "49 patches (7×7)",
                    bitmap = maeResult?.origBitmap,
                    modifier = Modifier.weight(1f)
                )
                MaePanelCard(
                    title = "2. Masked Input",
                    subtitle = "Only ${((1f - maskRatio) * 100).roundToInt()}% visible",
                    bitmap = maeResult?.maskedBitmap,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                MaePanelCard(
                    title = "3. MAE Inpainted",
                    subtitle = "Visible + Rebuilt",
                    bitmap = maeResult?.compositeBitmap,
                    highlight = true,
                    modifier = Modifier.weight(1f)
                )
                MaePanelCard(
                    title = "4. Raw Decoder",
                    subtitle = "All 49 patches",
                    bitmap = maeResult?.fullBitmap,
                    modifier = Modifier.weight(1f)
                )
            }

            if (maeResult != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Pixel MSE Loss: ${String.format("%.4f", maeResult.loss)}", color = IndigoAccent, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    Text("MAE Latency: ${maeResult.inferenceMs} ms", color = TextSecondary, fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
fun MaePanelCard(
    title: String,
    subtitle: String,
    bitmap: ImageBitmap?,
    highlight: Boolean = false,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(CardSlate.copy(alpha = 0.55f))
            .border(
                1.dp,
                if (highlight) IndigoAccent.copy(alpha = 0.6f) else BorderSubtle,
                RoundedCornerShape(14.dp)
            )
            .padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(10.dp))
                .background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap,
                    contentDescription = title,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Text("—", color = TextSecondary, fontSize = 12.sp)
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = title,
            color = if (highlight) Color(0xFFA5B4FC) else TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
        Text(text = subtitle, color = TextSecondary, fontSize = 10.sp)
    }
}
