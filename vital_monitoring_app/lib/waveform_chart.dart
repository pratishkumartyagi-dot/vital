import 'package:flutter/material.dart';

class WaveformPainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double strokeWidth;

  WaveformPainter({
    required this.data,
    this.color = Colors.red,
    this.strokeWidth = 2.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;

    final Paint paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final Path path = Path();
    
    // Calculate scaling factors
    final double xStep = size.width / (data.length - 1);
    final double yOffset = size.height / 2;
    final double yScale = size.height * 0.4; // Use 40% of height for waveform
    
    // Start path
    path.moveTo(0, yOffset - data[0] * yScale);
    
    // Draw waveform
    for (int i = 1; i < data.length; i++) {
      final double x = i * xStep;
      final double y = yOffset - data[i] * yScale;
      path.lineTo(x, y);
    }
    
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

class WaveformChart extends StatelessWidget {
  final List<double> data;
  final Color color;
  final String? label;

  const WaveformChart({
    Key? key,
    required this.data,
    this.color = Colors.red,
    this.label,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 100,
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (label != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Text(
                label!,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          Expanded(
            child: CustomPaint(
              painter: WaveformPainter(
                data: data,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}