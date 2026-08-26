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

    // Coarse spatial downsampling matrix (Every 4th grid row/col pixel)
    for (int y = 0; y < height; y += 4) {
      final int yRowStart = y * yRowStride;
      final int uvRow = (y ~/ 2) * uvRowStride;
      
      for (int x = 0; x < width; x += 4) {
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
}