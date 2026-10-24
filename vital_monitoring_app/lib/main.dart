import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';
import 'vital_sign_controller.dart';
import 'waveform_chart.dart';
import 'ml_inference.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await VitalMLInference.init();
  final cameras = await availableCameras();
  runApp(
    ChangeNotifierProvider(
      create: (_) => VitalSignController(),
      child: VitalMonitoringApp(cameras: cameras),
    ),
  );
}

class VitalMonitoringApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const VitalMonitoringApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vital Signs Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0A0A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00FF88),
          secondary: Color(0xFF00BFA5),
          surface: Color(0xFF1A1A1A),
        ),
      ),
      home: VitalHomePage(cameras: cameras),
    );
  }
}

class VitalHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const VitalHomePage({super.key, required this.cameras});

  @override
  State<VitalHomePage> createState() => _VitalHomePageState();
}

class _VitalHomePageState extends State<VitalHomePage> {
  CameraController? _controller;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final controller = CameraController(
      widget.cameras.isNotEmpty ? widget.cameras.first : CameraDescription(
        name: 'default',
        lensDirection: CameraLensDirection.back,
        sensorOrientation: 90,
      ),
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await controller.initialize();
      if (mounted) {
        setState(() {
          _controller = controller;
          _isInitialized = true;
        });
        final ctrl = context.read<VitalSignController>();
        ctrl.enableLiveMode();
        ctrl.attachStreamPipeline(controller);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Camera error: $e')),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Vital Signs Monitor'),
        centerTitle: true,
        actions: [
          Consumer<VitalSignController>(
            builder: (_, ctrl, __) {
              return IconButton(
                icon: Icon(ctrl.cameraMode == CameraMode.rear
                    ? Icons.camera_front
                    : Icons.camera_rear),
                onPressed: () {
                  ctrl.setCameraMode(
                    ctrl.cameraMode == CameraMode.rear
                        ? CameraMode.front
                        : CameraMode.rear,
                  );
                  _switchCamera();
                },
              );
            },
          ),
        ],
      ),
      body: Consumer<VitalSignController>(
        builder: (context, ctrl, child) {
          return Column(
            children: [
              // Camera preview
              Expanded(
                flex: 2,
                child: _isInitialized && _controller != null
                    ? Stack(
                        children: [
                          CameraPreview(_controller!),
                          Positioned.fill(
                            child: Container(
                              color: Colors.black.withOpacity(0.3),
                              child: Center(
                                child: Container(
                                  width: 200,
                                  height: 200,
                                  decoration: BoxDecoration(
                                    border: Border.all(
                                      color: ctrl.signalQuality > 50
                                          ? Colors.green
                                          : Colors.orange,
                                      width: 3,
                                    ),
                                    borderRadius: BorderRadius.circular(100),
                                  ),
                                ),
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 20,
                            left: 0,
                            right: 0,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 24,
                                vertical: 12,
                              ),
                              color: Colors.black.withOpacity(0.6),
                              child: Text(
                                ctrl.toastPrompt,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ),
                        ],
                      )
                    : const Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFF00FF88),
                        ),
                      ),
              ),
              // Vital signs dashboard
              Expanded(
                flex: 3,
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          Expanded(child: _vitalCard('Heart Rate', '${ctrl.heartRate.toStringAsFixed(0)} BPM', Colors.red, Icons.favorite)),
                          const SizedBox(width: 12),
                          Expanded(child: _vitalCard('SpO₂', '${ctrl.spo2Level.toStringAsFixed(0)}%', Colors.green, Icons.air)),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(child: _vitalCard('Hb', '${ctrl.hbLevel.toStringAsFixed(1)} g/dL', Colors.blue, Icons.bloodtype)),
                          const SizedBox(width: 12),
                          Expanded(child: _vitalCard('Resp Rate', '${ctrl.respRate.toStringAsFixed(0)} /min', Colors.orange, Icons.healing)),
                        ],
                      ),
                      const SizedBox(height: 20),
                      // Signal quality
                      _signalQualityCard(ctrl),
                      const SizedBox(height: 20),
                      // Waveform
                      WaveformChart(
                        data: ctrl.waveformData,
                        color: const Color(0xFF00FF88),
                        label: 'PPG Waveform',
                      ),
                      const SizedBox(height: 20),
                      // ML inference
                      _mlCard(ctrl),
                      const SizedBox(height: 20),
                      // Demo toggle
                      Consumer<VitalSignController>(
                        builder: (_, ctl, __) {
                          return SizedBox(
                            width: double.infinity,
                            height: 48,
                            child: ElevatedButton.icon(
                              icon: Icon(ctl.isDemoMode ? Icons.bug_report : Icons.camera_alt),
                              label: Text(ctl.isDemoMode ? 'Enable Live Mode' : 'Demo Mode'),
                              onPressed: ctl.isDemoMode ? ctl.enableLiveMode : ctl.enableDemoMode,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: ctl.isDemoMode
                                    ? Colors.orange
                                    : const Color(0xFF00FF88),
                                foregroundColor: Colors.black,
                              ),
                            ),
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _vitalCard(String label, String value, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  label,
                  style: TextStyle(
                    color: Colors.grey.shade400,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _signalQualityCard(VitalSignController ctrl) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade700),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Signal Quality',
            style: TextStyle(
              color: Colors.grey,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: ctrl.signalQuality / 100,
            backgroundColor: Colors.grey.shade800,
            valueColor: AlwaysStoppedAnimation<Color>(
              ctrl.signalQuality > 50 ? Colors.green : Colors.orange,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${ctrl.signalQuality.toStringAsFixed(0)}%',
            style: TextStyle(
              color: ctrl.signalQuality > 50 ? Colors.green : Colors.orange,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _mlCard(VitalSignController ctrl) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.purple.shade700),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'ML Estimation (TFLite)',
            style: TextStyle(
              color: Colors.purple,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(child: _mlValue('Systolic', '--', Colors.red.shade300)),
              const SizedBox(width: 12),
              Expanded(child: _mlValue('Diastolic', '--', Colors.red.shade300)),
              const SizedBox(width: 12),
              Expanded(child: _mlValue('Glucose', '--', Colors.orange.shade300)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _mlValue(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.grey.shade500,
            fontSize: 11,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  void _switchCamera() async {
    if (_controller == null) return;
    await _controller!.dispose();
    final cameras = widget.cameras;
    final newCamera = cameras.length > 1
        ? (cameras.first.lensDirection == CameraLensDirection.back
            ? cameras.last
            : cameras.first)
        : cameras.first;
    final controller = CameraController(newCamera, ResolutionPreset.medium, enableAudio: false);
    try {
      await controller.initialize();
      if (mounted) {
        setState(() {
          _controller = controller;
        });
        final ctrl = context.read<VitalSignController>();
        ctrl.attachStreamPipeline(controller);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Camera switch error: $e')),
        );
      }
    }
  }
}
