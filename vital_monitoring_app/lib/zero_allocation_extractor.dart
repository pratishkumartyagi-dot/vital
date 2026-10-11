import 'dart:typed_data';
import 'package:camera/camera.dart';

class ZeroAllocationExtractor {
  // Static memory storage allocation to prevent runtime heap churn
  static final Float64List _rgbLumaOutputBuffer = Float64List(4); 

static Float64List processYUV420Frame(CameraImage image) {
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final int width = image.width;
    final int height = image.height;
    final int yRowStride = yPlane.bytesPerRow;
    final int uvRowStride = uPlane.bytesPerRow;
    final int uvPixelStride = uPlane.bytesPerPixel ?? 2;

    final Uint8List yBytes = yPlane.bytes;
    final Uint8List uBytes = uPlane.bytes;
    final Uint8List vBytes = vPlane.bytes;

    int sumY = 0, sumU = 0, sumV = 0;
    int sampleCount = 0;

    // Improved spatial downsampling: Every 2nd grid row/col pixel (4x more samples)
    // This provides better spatial detail for finger detection while maintaining performance
    for (int y = 0; y < height; y += 2) {
      final int yRowStart = y * yRowStride;
      final int uvRow = (y ~/ 2) * uvRowStride;

      for (int x = 0; x < width; x += 2) {
        sumY += yBytes[yRowStart + x];
        final int uvIndex = uvRow + (x ~/ 2) * uvPixelStride;

        if (uvIndex < uBytes.length) {
          sumU += uBytes[uvIndex];
          sumV += vBytes[uvIndex];
          sampleCount++;
        }
      }
    }

    final double meanY = sumY / sampleCount;
    final double meanU = (sumU / sampleCount) - 128.0;
    final double meanV = (sumV / sampleCount) - 128.0;

    // Fast BT.601 color translation vectors
    _rgbLumaOutputBuffer[0] = (meanY + 1.402 * meanV).clamp(0.0, 255.0);               // R
    _rgbLumaOutputBuffer[1] = (meanY - 0.344136 * meanU - 0.714136 * meanV).clamp(0.0, 255.0); // G
    _rgbLumaOutputBuffer[2] = (meanY + 1.772 * meanU).clamp(0.0, 255.0);                // B
    _rgbLumaOutputBuffer[3] = meanY;                                                    // Y

    return _rgbLumaOutputBuffer;
  }

  /// Process a Y-plane from native Camera2 (Android) API.
  /// Estimates RGB from luminance using physics-based ratios for red LED illumination.
  /// Under red LED (660nm): R channel dominant, G/B minimal.
  /// Y = 0.299R + 0.587G + 0.114B (BT.601 luma)
  /// With red LED: G ≈ 0.1R, B ≈ 0.05R → Y ≈ 0.364R → R ≈ 2.75Y
  static Float64List processYPlane(Uint8List yPlane, int width, int height) {
    int sumY = 0;
    int sampleCount = 0;

    // Sample every 4th pixel for performance (matching 2x2 downsample)
    for (int y = 0; y < height; y += 4) {
      final int rowStart = y * width;
      for (int x = 0; x < width; x += 4) {
        final int idx = rowStart + x;
        if (idx < yPlane.length) {
          sumY += yPlane[idx];
          sampleCount++;
        }
      }
    }

    if (sampleCount == 0) {
      _rgbLumaOutputBuffer[0] = 0;
      _rgbLumaOutputBuffer[1] = 0;
      _rgbLumaOutputBuffer[2] = 0;
      _rgbLumaOutputBuffer[3] = 0;
      return _rgbLumaOutputBuffer;
    }

    final double meanY = sumY / sampleCount;

    // Physics-based estimation for red LED transillumination PPG:
    // Y = 0.299R + 0.587G + 0.114B
    // Under 660nm LED: G ~ 0.1R, B ~ 0.05R (hemoglobin absorption)
    // Y = 0.299R + 0.587(0.1R) + 0.114(0.05R) = 0.36365R
    // R ≈ Y / 0.36365 ≈ 2.75Y
    // G ≈ 0.1R ≈ 0.275Y
    // B ≈ 0.05R ≈ 0.1375Y
    _rgbLumaOutputBuffer[0] = (meanY * 2.75).clamp(0.0, 255.0);   // R
    _rgbLumaOutputBuffer[1] = (meanY * 0.275).clamp(0.0, 255.0);  // G
    _rgbLumaOutputBuffer[2] = (meanY * 0.1375).clamp(0.0, 255.0); // B
    _rgbLumaOutputBuffer[3] = meanY;                              // Y

    return _rgbLumaOutputBuffer;
  }
}