import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'native_hardware_bridge.dart';
import 'vital_sign_controller.dart';
import 'waveform_chart.dart';
import 'package:provider/provider.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize cameras with timeout to prevent hanging
  try {
    _cameras = await availableCameras().timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        print('Camera initialization timed out - using demo mode');
        return <CameraDescription>[];
      },
    );
  } catch (e) {
    print('Error getting cameras: $e');
    _cameras = [];
  }
  
  // Start the app IMMEDIATELY - don't wait for hardware operations
  runApp(const VitalMonitoringApp());
  
  // Lock hardware ISP in the background (non-blocking)
  try {
    await NativeHardwareBridge.lockHardwareISP().timeout(
      const Duration(seconds: 2),
      onTimeout: () {
        print('Hardware ISP lock timed out - continuing without lock');
        return null;
      },
    );
  } catch (e) {
    print('Hardware ISP lock failed: $e');
  }
}

class VitalMonitoringApp extends StatelessWidget {
  const VitalMonitoringApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => VitalSignController(),
      child: MaterialApp(
        title: 'PPG Vital Signs Monitor - Research Tool',
        theme: ThemeData(
          primarySwatch: Colors.red,
          brightness: Brightness.dark,
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
  bool _showInfoOverlay = true;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) {
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
      return;
    }

    _controller = CameraController(
      _cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {
        _isInitialized = true;
      });
      final vitalSignController =
          Provider.of<VitalSignController>(context, listen: false);
      vitalSignController.attachStreamPipeline(_controller);
    }).catchError((Object e) {
      if (mounted) {
        setState(() {
          _isInitialized = true;
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

  void _toggleInfoOverlay() {
    setState(() {
      _showInfoOverlay = !_showInfoOverlay;
    });
  }

  @override
  Widget build(BuildContext context) {
    final controller = Provider.of<VitalSignController>(context);

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera preview or placeholder
          _isInitialized && _cameras.isNotEmpty
              ? CameraPreview(_controller)
              : Container(
                  color: Colors.black,
                  child: const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.videocam,
                          size: 48,
                          color: Colors.grey,
                        ),
                        SizedBox(height: 16),
                        Text(
                          'Camera Not Available\nRunning in Demo Mode',
                          style: TextStyle(
                            color: Colors.grey,
                            fontSize: 16,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),

          // Overlay UI - FIXED: Use Positioned.fill to cover entire screen
          Positioned.fill(
            child: Container(
              color: Colors.black54,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  // Waveform visualization
                  Container(
                    height: 120,
                    padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                    child: WaveformChart(
                      data: controller.waveformData,
                      color: Colors.redAccent,
                      label: 'PPG Signal',
                    ),
                  ),

                  // Status and prompts
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

                  // Progress indicator during profiling
                  if (controller.currentState == TrackingState.profiling)
                    LinearProgressIndicator(
                      value: controller.gatheringProgress,
                      backgroundColor: Colors.grey[800],
                      valueColor: const AlwaysStoppedAnimation<Color>(Colors.red),
                    ),

                  // Heart rate display when measurement is complete
                  if (controller.currentState == TrackingState.complete)
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(
                        'Heart Rate: ${controller.heartRate.toStringAsFixed(1)} BPM',
                        style: const TextStyle(
                          color: Colors.green,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),

                  // Demo mode indicator
                  if (controller.isDemoMode)
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12.0, vertical: 6.0),
                        decoration: BoxDecoration(
                          color: Colors.orange.withOpacity(0.7),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Text(
                          'DEMO MODE',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),

                  const SizedBox(height: 20),

                  // Info toggle button
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        GestureDetector(
                          onTap: _toggleInfoOverlay,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12.0, vertical: 6.0),
                            decoration: BoxDecoration(
                              color: Colors.blue.withOpacity(0.5),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              _showInfoOverlay ? 'Hide Info' : 'Show Info',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 14,
                              ),
                            ),
                          ),
                        ),
                        Text(
                          'PPG Vital Monitor v1.0\nResearch Tool for Photoplethysmography',
                          style: const TextStyle(
                            color: Colors.grey,
                            fontSize: 12,
                          ),
                          textAlign: TextAlign.right,
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 10),
                ],
              ),
            ),
          ),

          // Info overlay - FIXED: Use Positioned.fill to cover entire screen
          if (_showInfoOverlay)
            Positioned.fill(
              child: Container(
                color: Colors.black87,
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'PPG Vital Signs Monitor',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'What this app does:',
                        style: TextStyle(
                          color: Colors.lightGreen,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '• Uses smartphone camera to detect PPG signals from fingertip\n'
                        '• Measures heart rate by analyzing blood volume changes\n'
                        '• Displays real-time PPG waveform for signal quality assessment\n'
                        '• Applies hardware locks to disable auto white balance/exposure\n'
                        '• Uses zero-allocation processing for efficient operation\n'
                        '• Implements research-grade signal processing pipeline\n'
                        '• Includes Leave-One-Subject-Out validation framework\n',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'How to use for research:',
                        style: TextStyle(
                          color: Colors.lightGreen,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '1. Place fingertip firmly over camera lens and flash\n'
                        '2. Hold steady and wait for signal acquisition\n'
                        '3. Monitor waveform quality - look for clear pulsatile signal\n'
                        '4. Wait for measurement to complete (progress bar fills)\n'
                        '5. Record the displayed heart rate value\n'
                        '6. Repeat measurements for data collection\n'
                        '7. Use demo mode for testing without camera access\n'
                        '8. Export data using the validation script for LOSO analysis\n',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'Technical Notes:',
                        style: TextStyle(
                          color: Colors.lightGreen,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '• Sampling rate: ~30 FPS from camera preview\n'
                        '• Bandpass filter: 0.5-3.5 Hz (40-210 BPM range)\n'
                        '• Perfusion index threshold: 0.5% for signal quality\n'
                        '• Uses 4th-order Butterworth zero-phase filtering\n'
                        '• Spatial downsampling reduces processing load\n'
                        '• Native Android hardware ISP locking via MethodChannel\n'
                        '• Zero-allocation frame processing prevents GC pressure\n'
                        '• Peak detection algorithm for heart rate calculation\n',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 13,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          final vitalSignController =
              Provider.of<VitalSignController>(context, listen: false);
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Measurement reset - place finger over camera to start'),
              duration: Duration(seconds: 2),
            ),
          );
        },
        label: const Text('Reset'),
        icon: const Icon(Icons.refresh),
        backgroundColor: Colors.red,
      ),
    );
  }
}