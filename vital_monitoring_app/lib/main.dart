import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'native_hardware_bridge.dart';
import 'vital_sign_controller.dart';
import 'package:provider/provider.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  // Lock hardware ISP before starting the app
  await NativeHardwareBridge.lockHardwareISP();
  runApp(const VitalMonitoringApp());
}

class VitalMonitoringApp extends StatelessWidget {
  const VitalMonitoringApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => VitalSignController(),
      child: MaterialApp(
        title: 'Vital Monitoring App',
        theme: ThemeData(
          primarySwatch: Colors.red,
        ),
        home: const VitalSignPage(),
      ),
    );
  }
}

class VitalSignPage extends StatefulWidget {
  const VitalSignPage({super.key});

  @override
  State<VitalSignPage> createState() => _VitalSignPageState();
}

class _VitalSignPageState extends State<VitalSignPage> {
  late CameraController _controller;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      _cameras.first,
      ResolutionPreset.medium,
      // Enable audio to ensure we get the full frame rate
      enableAudio: false,
    );

    _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {
        _isInitialized = true;
      });
      // Attach the stream pipeline to the controller
      final vitalSignController =
          Provider.of<VitalSignController>(context, listen: false);
      vitalSignController.attachStreamPipeline(_controller);
    }).catchError((Object e) {
      if (mounted) {
        setState(() {
          _isInitialized = false;
        });
      }
      print('Camera initialization error: $e');
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_controller),
          Consumer<VitalSignController>(
            builder: (context, controller, child) => Container(
              color: Colors.black54,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Text(
                      controller.toastPrompt,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  if (controller.currentState == TrackingState.profiling)
                    LinearProgressIndicator(
                      value: controller.gatheringProgress,
                      backgroundColor: Colors.grey[800],
                      valueColor: const AlwaysStoppedAnimation<Color>(Colors.red),
                    ),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}