package com.fahrface.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

data class LandmarkPoints(
    val rightEye: Pair<Float, Float>,
    val leftEye: Pair<Float, Float>,
    val nose: Pair<Float, Float>,
    val rightMouth: Pair<Float, Float>,
    val leftMouth: Pair<Float, Float>
)

data class FaceAnalysis(
    val faceId: Int,
    val bbox: IntArray,
    val confidence: Float,
    val alignmentAngleDeg: Float,
    val interocularPx: Float?,
    val detectorBackend: String = "RetinaFace",
    val landmarks: LandmarkPoints?,
    val predictedAge: Float,
    val inferenceMs: Float,
    val device: String,
    val regionalAttribution: List<Pair<String, Float>>,
    val alignedBitmap: ImageBitmap?,
    val heatmapBitmap: ImageBitmap?
)

data class AnalyzeResult(
    val imageWidth: Int,
    val imageHeight: Int,
    val originalBitmap: ImageBitmap?,
    val faces: List<FaceAnalysis>,
    val device: String
)

data class MaeResult(
    val maskRatio: Float,
    val numMasked: Int,
    val totalPatches: Int,
    val loss: Float,
    val inferenceMs: Float,
    val origBitmap: ImageBitmap?,
    val maskedBitmap: ImageBitmap?,
    val compositeBitmap: ImageBitmap?,
    val fullBitmap: ImageBitmap?
)

object OnDeviceAiEngine {

    private const val YUNET_ASSET = "face_detection_yunet_2023mar.onnx"
    private const val RETINAFACE_ASSET = "retinaface_int8.onnx"
    private const val AGE_ASSET = "fahr_faceage_int8.onnx"
    private const val MAE_ASSET = "fahr_face_mae_int8.onnx"
    private const val SAMPLE_ASSET = "sample_face.jpg"

    private val initMutex = Mutex()
    private var ortEnv: OrtEnvironment? = null
    private var yunetSession: OrtSession? = null
    private var retinafaceSession: OrtSession? = null
    private var ageSession: OrtSession? = null
    private var maeSession: OrtSession? = null

    // Last aligned 112x112 bitmaps cached by faceId so MAE tab can re-roll instantly
    private val cachedAlignedBitmaps = mutableMapOf<Int, Bitmap>()

    fun bitmapToJpegBytes(bmp: Bitmap): ByteArray {
        val out = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 94, out)
        return out.toByteArray()
    }

    fun loadBundledSampleBytes(context: Context): ByteArray {
        return context.assets.open(SAMPLE_ASSET).use { it.readBytes() }
    }

    private fun ensureAssetFile(context: Context, assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 100_000L) {
            return outFile
        }
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    output.write(buf, 0, read)
                }
                output.flush()
            }
        }
        return outFile
    }

    private suspend fun ensureSessions(context: Context) {
        if (yunetSession != null && retinafaceSession != null && ageSession != null && maeSession != null) return
        initMutex.withLock {
            if (yunetSession != null && retinafaceSession != null && ageSession != null && maeSession != null) return
            val env = OrtEnvironment.getEnvironment()
            ortEnv = env
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val yunetFile = ensureAssetFile(context, YUNET_ASSET)
            val retinafaceFile = ensureAssetFile(context, RETINAFACE_ASSET)
            val ageFile = ensureAssetFile(context, AGE_ASSET)
            val maeFile = ensureAssetFile(context, MAE_ASSET)

            yunetSession = env.createSession(yunetFile.absolutePath, opts)
            retinafaceSession = env.createSession(retinafaceFile.absolutePath, opts)
            ageSession = env.createSession(ageFile.absolutePath, opts)
            maeSession = env.createSession(maeFile.absolutePath, opts)
        }
    }

    private fun decodeBitmapWithExif(imageBytes: ByteArray): Bitmap {
        val rawBmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: throw IllegalArgumentException("Could not decode image bytes")
        val exif = try {
            ExifInterface(ByteArrayInputStream(imageBytes))
        } catch (_: Exception) {
            return rawBmp
        }
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            else -> return rawBmp
        }
        return Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, matrix, true)
    }

    private data class RawFaceDet(
        val x1: Float,
        val y1: Float,
        val w: Float,
        val h: Float,
        val rightEye: Pair<Float, Float>,
        val leftEye: Pair<Float, Float>,
        val nose: Pair<Float, Float>,
        val rightMouth: Pair<Float, Float>,
        val leftMouth: Pair<Float, Float>,
        val score: Float
    )

    private fun iou(a: RawFaceDet, b: RawFaceDet): Float {
        val xx1 = max(a.x1, b.x1)
        val yy1 = max(a.y1, b.y1)
        val xx2 = min(a.x1 + a.w, b.x1 + b.w)
        val yy2 = min(a.y1 + a.h, b.y1 + b.h)
        val interW = max(0f, xx2 - xx1)
        val interH = max(0f, yy2 - yy1)
        val interArea = interW * interH
        val unionArea = a.w * a.h + b.w * b.h - interArea + 1e-6f
        return interArea / unionArea
    }

    private fun detectFacesYuNet(bmp: Bitmap): List<RawFaceDet> {
        val env = ortEnv!!
        val session = yunetSession!!

        val inputSize = 640
        val resized = Bitmap.createScaledBitmap(bmp, inputSize, inputSize, true)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val planeSize = inputSize * inputSize
        val floatBuf = FloatBuffer.allocate(3 * planeSize)
        // YuNet expects BGR [0..255]
        for (i in 0 until planeSize) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF).toFloat()
            val g = ((px shr 8) and 0xFF).toFloat()
            val b = (px and 0xFF).toFloat()
            floatBuf.put(i, b)
            floatBuf.put(planeSize + i, g)
            floatBuf.put(2 * planeSize + i, r)
        }
        floatBuf.rewind()

        val scaleX = bmp.width.toFloat() / inputSize.toFloat()
        val scaleY = bmp.height.toFloat() / inputSize.toFloat()

        val candidates = mutableListOf<RawFaceDet>()
        val strides = intArrayOf(8, 16, 32)

        OnnxTensor.createTensor(env, floatBuf, longArrayOf(1, 3, 640, 640)).use { tensor ->
            session.run(mapOf("input" to tensor)).use { results ->
                for (sIdx in strides.indices) {
                    val s = strides[sIdx]
                    val gw = inputSize / s
                    @Suppress("UNCHECKED_CAST")
                    val clsArr = (results[sIdx].value as Array<Array<FloatArray>>)[0]
                    @Suppress("UNCHECKED_CAST")
                    val objArr = (results[sIdx + 3].value as Array<Array<FloatArray>>)[0]
                    @Suppress("UNCHECKED_CAST")
                    val bboxArr = (results[sIdx + 6].value as Array<Array<FloatArray>>)[0]
                    @Suppress("UNCHECKED_CAST")
                    val kpsArr = (results[sIdx + 9].value as Array<Array<FloatArray>>)[0]

                    for (idx in clsArr.indices) {
                        val cVal = clsArr[idx][0].coerceIn(0f, 1f)
                        val oVal = objArr[idx][0].coerceIn(0f, 1f)
                        val score = sqrt(cVal * oVal)
                        if (score < 0.6f) continue

                        val r = idx / gw
                        val c = idx % gw
                        val bb = bboxArr[idx]
                        val kp = kpsArr[idx]

                        val cx = (c + bb[0]) * s
                        val cy = (r + bb[1]) * s
                        val w = exp(bb[2]) * s
                        val h = exp(bb[3]) * s
                        val x1 = (cx - w / 2f) * scaleX
                        val y1 = (cy - h / 2f) * scaleY
                        val bw = w * scaleX
                        val bh = h * scaleY

                        fun pt(k: Int): Pair<Float, Float> {
                            val px = (c + kp[2 * k]) * s * scaleX
                            val py = (r + kp[2 * k + 1]) * s * scaleY
                            return Pair(px, py)
                        }

                        candidates.add(
                            RawFaceDet(
                                x1 = x1,
                                y1 = y1,
                                w = bw,
                                h = bh,
                                rightEye = pt(0),
                                leftEye = pt(1),
                                nose = pt(2),
                                rightMouth = pt(3),
                                leftMouth = pt(4),
                                score = score
                            )
                        )
                    }
                }
            }
        }

        // Non-Maximum Suppression (NMS = 0.3)
        candidates.sortByDescending { it.score }
        val kept = mutableListOf<RawFaceDet>()
        for (cand in candidates) {
            var overlap = false
            for (k in kept) {
                if (iou(cand, k) > 0.3f) {
                    overlap = true
                    break
                }
            }
            if (!overlap) kept.add(cand)
        }
        // Sort by bounding box area descending (matches face_pipeline.py)
        return kept.sortedByDescending { it.w * it.h }
    }

    private fun detectFacesRetinaFace(bmp: Bitmap): List<RawFaceDet> {
        val env = ortEnv ?: return emptyList()
        val session = retinafaceSession ?: return emptyList()

        val inputSize = 640
        val resized = Bitmap.createScaledBitmap(bmp, inputSize, inputSize, true)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val planeSize = inputSize * inputSize
        val floatBuf = FloatBuffer.allocate(3 * planeSize)
        // RetinaFace expects RGB [0..255] float32
        for (i in 0 until planeSize) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF).toFloat()
            val g = ((px shr 8) and 0xFF).toFloat()
            val b = (px and 0xFF).toFloat()
            floatBuf.put(i, r)
            floatBuf.put(planeSize + i, g)
            floatBuf.put(2 * planeSize + i, b)
        }
        floatBuf.rewind()

        val scaleX = bmp.width.toFloat() / inputSize.toFloat()
        val scaleY = bmp.height.toFloat() / inputSize.toFloat()
        val candidates = mutableListOf<RawFaceDet>()

        OnnxTensor.createTensor(env, floatBuf, longArrayOf(1, 3, 640, 640)).use { tensor ->
            session.run(mapOf("input" to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val detArr = (results[0].value as Array<Array<FloatArray>>)[0]

                for (det in detArr) {
                    val score = det[14]
                    if (score < 0.6f) continue

                    val x1 = det[0] * scaleX
                    val y1 = det[1] * scaleY
                    val bw = det[2] * scaleX
                    val bh = det[3] * scaleY

                    val rightEye = Pair(det[4] * scaleX, det[5] * scaleY)
                    val leftEye = Pair(det[6] * scaleX, det[7] * scaleY)
                    val nose = Pair(det[8] * scaleX, det[9] * scaleY)
                    val rightMouth = Pair(det[10] * scaleX, det[11] * scaleY)
                    val leftMouth = Pair(det[12] * scaleX, det[13] * scaleY)

                    candidates.add(
                        RawFaceDet(
                            x1 = x1,
                            y1 = y1,
                            w = bw,
                            h = bh,
                            rightEye = rightEye,
                            leftEye = leftEye,
                            nose = nose,
                            rightMouth = rightMouth,
                            leftMouth = leftMouth,
                            score = score
                        )
                    )
                }
            }
        }

        // Non-Maximum Suppression (NMS = 0.4)
        candidates.sortByDescending { it.score }
        val kept = mutableListOf<RawFaceDet>()
        for (cand in candidates) {
            var overlap = false
            for (k in kept) {
                if (iou(cand, k) > 0.4f) {
                    overlap = true
                    break
                }
            }
            if (!overlap) kept.add(cand)
        }
        return kept.sortedByDescending { it.w * it.h }
    }

    private fun alignAndCropFace(
        srcBmp: Bitmap,
        bbox: IntArray,
        rightEyeIn: Pair<Float, Float>,
        leftEyeIn: Pair<Float, Float>
    ): Bitmap {
        val w = srcBmp.width
        val h = srcBmp.height
        val x = bbox[0]
        val y = bbox[1]
        val bw = bbox[2]
        val bh = bbox[3]

        val (rightEye, leftEye) = if (rightEyeIn.first > leftEyeIn.first) {
            Pair(leftEyeIn, rightEyeIn)
        } else {
            Pair(rightEyeIn, leftEyeIn)
        }

        val dx = leftEye.first - rightEye.first
        val dy = leftEye.second - rightEye.second
        val angleDeg = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()

        val eyesCenterX = (rightEye.first + leftEye.first) / 2f
        val eyesCenterY = (rightEye.second + leftEye.second) / 2f

        val rotatedBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(rotatedBmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        val matrix = Matrix().apply {
            // Rotate counter-clockwise by angleDeg around eyesCenter so eyes become horizontal
            postRotate(-angleDeg, eyesCenterX, eyesCenterY)
        }
        canvas.drawBitmap(srcBmp, matrix, paint)

        val boxCx = x + bw / 2f
        val boxCy = y + bh / 2f
        val relX = boxCx - eyesCenterX
        val relY = boxCy - eyesCenterY
        val rad = Math.toRadians(-angleDeg.toDouble())
        val rotCx = (eyesCenterX + relX * cos(rad) - relY * sin(rad)).toFloat()
        val rotCy = (eyesCenterY + relX * sin(rad) + relY * cos(rad)).toFloat()

        var x1 = (rotCx - bw / 2f).roundToInt().coerceAtLeast(0)
        var y1 = (rotCy - bh / 2f).roundToInt().coerceAtLeast(0)
        var x2 = (rotCx + bw / 2f).roundToInt().coerceAtMost(w)
        var y2 = (rotCy + bh / 2f).roundToInt().coerceAtMost(h)

        if (x2 <= x1 || y2 <= y1) {
            x1 = x.coerceAtLeast(0)
            y1 = y.coerceAtLeast(0)
            x2 = (x + bw).coerceAtMost(w)
            y2 = (y + bh).coerceAtMost(h)
        }
        val cropW = (x2 - x1).coerceAtLeast(1)
        val cropH = (y2 - y1).coerceAtLeast(1)
        return Bitmap.createBitmap(rotatedBmp, x1, y1, cropW, cropH)
    }

    private fun bitmap112ToRgb01Buffer(bmp112: Bitmap): FloatBuffer {
        val pixels = IntArray(112 * 112)
        bmp112.getPixels(pixels, 0, 112, 0, 0, 112, 112)
        val plane = 112 * 112
        val buf = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF) / 255f
            val g = ((px shr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f
            buf.put(i, r)
            buf.put(plane + i, g)
            buf.put(2 * plane + i, b)
        }
        buf.rewind()
        return buf
    }

    // Polynomial approximation of Google's Turbo colormap for [0..1]
    private fun turboColor(vIn: Float): Int {
        val x = vIn.coerceIn(0f, 1f)
        val r = (0.13572138f + x * (4.61539260f + x * (-42.66032258f + x * (132.13108234f + x * (-152.94239396f + x * 59.28637943f))))).coerceIn(0f, 1f)
        val g = (0.09140261f + x * (2.19418839f + x * (4.84296658f + x * (-14.18503333f + x * (4.27729857f + x * 2.82956604f))))).coerceIn(0f, 1f)
        val b = (0.10667330f + x * (12.64194608f + x * (-60.58204836f + x * (110.36276771f + x * (-89.90310912f + x * 27.34824973f))))).coerceIn(0f, 1f)
        val ri = (r * 255f).roundToInt()
        val gi = (g * 255f).roundToInt()
        val bi = (b * 255f).roundToInt()
        return (0xFF shl 24) or (ri shl 16) or (gi shl 8) or bi
    }

    private fun renderHeatmapBitmap(grid7x7: Array<FloatArray>, size: Int = 224): Bitmap {
        // First create a 7x7 float grid bitmap and upscale smoothly to 224x224
        val smallBmp = Bitmap.createBitmap(7, 7, Bitmap.Config.ARGB_8888)
        for (r in 0 until 7) {
            for (c in 0 until 7) {
                val gray = (grid7x7[r][c].coerceIn(0f, 1f) * 255f).roundToInt()
                val argb = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
                smallBmp.setPixel(c, r, argb)
            }
        }
        val smoothGray = Bitmap.createScaledBitmap(smallBmp, size, size, true)
        val pixels = IntArray(size * size)
        smoothGray.getPixels(pixels, 0, size, 0, 0, size, size)
        for (i in pixels.indices) {
            val v = (pixels[i] and 0xFF) / 255f
            pixels[i] = turboColor(v)
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    private fun computeRegionalAttribution(grid: Array<FloatArray>): List<Pair<String, Float>> {
        fun meanRows(rStart: Int, rEnd: Int): Float {
            var sum = 0f
            var count = 0
            for (r in rStart until rEnd) {
                for (c in 1 until 6) {
                    sum += grid[r][c]
                    count++
                }
            }
            return sum / count.toFloat()
        }
        val forehead = meanRows(0, 2)
        val eyes = meanRows(2, 4)
        val midface = meanRows(3, 5)
        val lowerface = meanRows(5, 7)
        val total = forehead + eyes + midface + lowerface + 1e-8f

        val list = mutableListOf(
            Pair("Forehead & Glabella", ((forehead / total) * 1000f).roundToInt() / 10f),
            Pair("Periorbital & Eyes", ((eyes / total) * 1000f).roundToInt() / 10f),
            Pair("Cheeks & Nasolabial", ((midface / total) * 1000f).roundToInt() / 10f),
            Pair("Mouth, Chin & Jaw", ((lowerface / total) * 1000f).roundToInt() / 10f)
        )
        list.sortByDescending { it.second }
        return list
    }

    suspend fun analyzePhotoOnDevice(
        context: Context,
        imageBytes: ByteArray,
        detectorBackend: String = "retinaface"
    ): AnalyzeResult = withContext(Dispatchers.Default) {
        ensureSessions(context)

        val origBmp = decodeBitmapWithExif(imageBytes)
        val origW = origBmp.width
        val origH = origBmp.height

        val rawDets = if (detectorBackend.equals("retinaface", ignoreCase = true)) {
            val rf = detectFacesRetinaFace(origBmp)
            if (rf.isNotEmpty()) rf else detectFacesYuNet(origBmp)
        } else {
            detectFacesYuNet(origBmp)
        }
        val actualDetector = if (detectorBackend.equals("retinaface", ignoreCase = true)) "RetinaFace" else "YuNet"
        val facesList = mutableListOf<FaceAnalysis>()
        cachedAlignedBitmaps.clear()

        val env = ortEnv!!
        val ageSess = ageSession!!

        if (rawDets.isNotEmpty()) {
            for ((idx, det) in rawDets.withIndex()) {
                val x = det.x1.roundToInt().coerceAtLeast(0)
                val y = det.y1.roundToInt().coerceAtLeast(0)
                val bw = det.w.roundToInt().coerceAtMost(origW - x)
                val bh = det.h.roundToInt().coerceAtMost(origH - y)
                if (bw < 16 || bh < 16) continue

                val bbox = intArrayOf(x, y, bw, bh)
                val alignedCrop = alignAndCropFace(origBmp, bbox, det.rightEye, det.leftEye)
                val aligned112 = Bitmap.createScaledBitmap(alignedCrop, 112, 112, true)
                cachedAlignedBitmaps[idx] = aligned112

                val t0 = System.nanoTime()
                val rgbBuf = bitmap112ToRgb01Buffer(aligned112)
                var predAge = 0f
                var grid7x7 = Array(7) { FloatArray(7) }

                OnnxTensor.createTensor(env, rgbBuf, longArrayOf(1, 3, 112, 112)).use { tensor ->
                    ageSess.run(mapOf("rgb_01" to tensor)).use { out ->
                        predAge = (out[0].value as FloatArray)[0]
                        @Suppress("UNCHECKED_CAST")
                        grid7x7 = out[1].value as Array<FloatArray>
                    }
                }
                val elapsedMs = ((System.nanoTime() - t0) / 100_000L) / 10f

                val (rePt, lePt) = if (det.rightEye.first > det.leftEye.first) {
                    Pair(det.leftEye, det.rightEye)
                } else {
                    Pair(det.rightEye, det.leftEye)
                }
                val dx = lePt.first - rePt.first
                val dy = lePt.second - rePt.second
                val alignAngle = (Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat() * 100f).roundToInt() / 100f
                val iod = (hypot(dx.toDouble(), dy.toDouble()).toFloat() * 10f).roundToInt() / 10f

                val aligned224 = Bitmap.createScaledBitmap(aligned112, 224, 224, true)
                val heatmap224 = renderHeatmapBitmap(grid7x7, 224)

                facesList.add(
                    FaceAnalysis(
                        faceId = idx,
                        bbox = bbox,
                        confidence = det.score,
                        alignmentAngleDeg = alignAngle,
                        interocularPx = iod,
                        detectorBackend = actualDetector,
                        landmarks = LandmarkPoints(
                            rightEye = det.rightEye,
                            leftEye = det.leftEye,
                            nose = det.nose,
                            rightMouth = det.rightMouth,
                            leftMouth = det.leftMouth
                        ),
                        predictedAge = (predAge * 100f).roundToInt() / 100f,
                        inferenceMs = elapsedMs,
                        device = "Pixel 9 On-Device",
                        regionalAttribution = computeRegionalAttribution(grid7x7),
                        alignedBitmap = aligned224.asImageBitmap(),
                        heatmapBitmap = heatmap224.asImageBitmap()
                    )
                )
            }
        }

        if (facesList.isEmpty()) {
            val aligned112 = Bitmap.createScaledBitmap(origBmp, 112, 112, true)
            cachedAlignedBitmaps[0] = aligned112
            val t0 = System.nanoTime()
            val rgbBuf = bitmap112ToRgb01Buffer(aligned112)
            var predAge = 0f
            var grid7x7 = Array(7) { FloatArray(7) }
            OnnxTensor.createTensor(env, rgbBuf, longArrayOf(1, 3, 112, 112)).use { tensor ->
                ageSess.run(mapOf("rgb_01" to tensor)).use { out ->
                    predAge = (out[0].value as FloatArray)[0]
                    @Suppress("UNCHECKED_CAST")
                    grid7x7 = out[1].value as Array<FloatArray>
                }
            }
            val elapsedMs = ((System.nanoTime() - t0) / 100_000L) / 10f
            val aligned224 = Bitmap.createScaledBitmap(aligned112, 224, 224, true)
            val heatmap224 = renderHeatmapBitmap(grid7x7, 224)
            facesList.add(
                FaceAnalysis(
                    faceId = 0,
                    bbox = intArrayOf(0, 0, origW, origH),
                    confidence = 1.0f,
                    alignmentAngleDeg = 0.0f,
                    interocularPx = null,
                    detectorBackend = "Fallback (Pre-Cropped)",
                    landmarks = null,
                    predictedAge = (predAge * 100f).roundToInt() / 100f,
                    inferenceMs = elapsedMs,
                    device = "Pixel 9 On-Device",
                    regionalAttribution = computeRegionalAttribution(grid7x7),
                    alignedBitmap = aligned224.asImageBitmap(),
                    heatmapBitmap = heatmap224.asImageBitmap()
                )
            )
        }

        AnalyzeResult(
            imageWidth = origW,
            imageHeight = origH,
            originalBitmap = origBmp.asImageBitmap(),
            faces = facesList,
            device = "Pixel 9 On-Device"
        )
    }

    suspend fun runMaeOnDevice(
        context: Context,
        faceId: Int,
        seed: Int
    ): MaeResult = withContext(Dispatchers.Default) {
        ensureSessions(context)
        val aligned112 = cachedAlignedBitmaps[faceId] ?: cachedAlignedBitmaps[0]
            ?: throw IllegalStateException("No aligned face cached for MAE")

        val env = ortEnv!!
        val maeSess = maeSession!!

        val t0 = System.nanoTime()
        val rgbBuf = bitmap112ToRgb01Buffer(aligned112)
        val rng = java.util.Random(seed.toLong())
        val noiseBuf = FloatBuffer.allocate(49)
        for (i in 0 until 49) {
            noiseBuf.put(i, rng.nextFloat())
        }
        noiseBuf.rewind()

        val plane = 112 * 112
        var reconChw = Array(3) { Array(112) { FloatArray(112) } }
        var maskChw = Array(3) { Array(112) { FloatArray(112) } }
        var lossVal = 0f

        OnnxTensor.createTensor(env, rgbBuf, longArrayOf(1, 3, 112, 112)).use { rgbTensor ->
            OnnxTensor.createTensor(env, noiseBuf, longArrayOf(1, 49)).use { noiseTensor ->
                maeSess.run(mapOf("rgb_01" to rgbTensor, "noise" to noiseTensor)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    reconChw = (out[0].value as Array<Array<Array<FloatArray>>>)[0]
                    @Suppress("UNCHECKED_CAST")
                    maskChw = (out[1].value as Array<Array<Array<FloatArray>>>)[0]
                    lossVal = (out[2].value as FloatArray)[0]
                }
            }
        }
        val elapsedMs = ((System.nanoTime() - t0) / 100_000L) / 10f

        val origPixels = IntArray(plane)
        aligned112.getPixels(origPixels, 0, 112, 0, 0, 112, 112)

        val maskedPixels = IntArray(plane)
        val compositePixels = IntArray(plane)
        val fullReconPixels = IntArray(plane)

        for (y in 0 until 112) {
            for (x in 0 until 112) {
                val idx = y * 112 + x
                val oPx = origPixels[idx]
                val oR = ((oPx shr 16) and 0xFF) / 255f
                val oG = ((oPx shr 8) and 0xFF) / 255f
                val oB = (oPx and 0xFF) / 255f

                val rR = reconChw[0][y][x].coerceIn(0f, 1f)
                val rG = reconChw[1][y][x].coerceIn(0f, 1f)
                val rB = reconChw[2][y][x].coerceIn(0f, 1f)

                val m = maskChw[0][y][x].coerceIn(0f, 1f)

                // Masked input visualization with subtle 16x16 patch grid lines
                var mR = ((oR * (1f - m) + 0.14f * m) * 255f).roundToInt()
                var mG = ((oG * (1f - m) + 0.14f * m) * 255f).roundToInt()
                var mB = ((oB * (1f - m) + 0.14f * m) * 255f).roundToInt()
                if ((x > 0 && x % 16 == 0) || (y > 0 && y % 16 == 0)) {
                    mR = (mR + 35).coerceAtMost(255)
                    mG = (mG + 35).coerceAtMost(255)
                    mB = (mB + 35).coerceAtMost(255)
                }
                maskedPixels[idx] = (0xFF shl 24) or (mR shl 16) or (mG shl 8) or mB

                // Composite (visible original + reconstructed masked patches)
                val cR = ((oR * (1f - m) + rR * m) * 255f).roundToInt().coerceIn(0, 255)
                val cG = ((oG * (1f - m) + rG * m) * 255f).roundToInt().coerceIn(0, 255)
                val cB = ((oB * (1f - m) + rB * m) * 255f).roundToInt().coerceIn(0, 255)
                compositePixels[idx] = (0xFF shl 24) or (cR shl 16) or (cG shl 8) or cB

                // Full raw MAE output
                val fR = (rR * 255f).roundToInt().coerceIn(0, 255)
                val fG = (rG * 255f).roundToInt().coerceIn(0, 255)
                val fB = (rB * 255f).roundToInt().coerceIn(0, 255)
                fullReconPixels[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
            }
        }

        val orig224 = Bitmap.createScaledBitmap(aligned112, 224, 224, true)
        val masked112 = Bitmap.createBitmap(maskedPixels, 112, 112, Bitmap.Config.ARGB_8888)
        val masked224 = Bitmap.createScaledBitmap(masked112, 224, 224, false)
        val comp112 = Bitmap.createBitmap(compositePixels, 112, 112, Bitmap.Config.ARGB_8888)
        val comp224 = Bitmap.createScaledBitmap(comp112, 224, 224, true)
        val full112 = Bitmap.createBitmap(fullReconPixels, 112, 112, Bitmap.Config.ARGB_8888)
        val full224 = Bitmap.createScaledBitmap(full112, 224, 224, true)

        MaeResult(
            maskRatio = 0.75f,
            numMasked = 37,
            totalPatches = 49,
            loss = (lossVal * 10000f).roundToInt() / 10000f,
            inferenceMs = elapsedMs,
            origBitmap = orig224.asImageBitmap(),
            maskedBitmap = masked224.asImageBitmap(),
            compositeBitmap = comp224.asImageBitmap(),
            fullBitmap = full224.asImageBitmap()
        )
    }
}
