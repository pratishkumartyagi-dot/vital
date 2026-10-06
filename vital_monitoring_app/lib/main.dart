import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'native_hardware_bridge.dart';
import 'vital_sign_controller.dart'
import 'waveform_chart.dart';
import 'package:provider/provider.dart';
import 'ml_inference.dart';
import 'feature_extractor.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  FlutterError.onError = (FlutterErrorDetails details) {
    debugPrint('Flutter error: ${details.exception}');
    debugPrint('Stack trace: ${details.stack}');
  };

  try {
    _cameras = await availableCameras().timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        debugPrint('Camera initialization timed out - using demo mode');
        return <CameraDescription>[];
      },
    );
  } catch (e) {
    debugPrint('Error getting cameras: $e');
    _cameras = [];
  }

  runApp(const VitalMonitoringApp());
}

class VitalMonitoringApp extends StatelessWidget {
  const VitalMonitoringApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => VitalSignController(),
      child: MaterialApp(
        title: 'PPG Vital Signs Monitor - Research Tool',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.red,
          brightness: Brightness.dark,
          useMaterial3: true,
        ),
        home: const VitalSignPage(),
      ),
    );
  }
}

enum CameraMode { front, rear }
enum MeasurementMode { live, validation }

class VitalSignPage extends StatefulWidget {
  const VitalSignPage({super.key});

  @override
  State<VitalSignPage> createState() => _VitalSignPageState();
}

class _VitalSignPageState extends State<VitalSignPage> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isInitialized = false;
  bool _showInfoOverlay = false;
  String? _cameraError;
  CameraMode _cameraMode = CameraMode.rear;
  MeasurementMode _measurementMode = MeasurementMode.live;
  bool _torchEnabled = true;
  bool _isRecording = false;
  List<Map<String, dynamic>> _validationData = [];
  Map<String, dynamic>? _groundTruth = {};
  Timer? _validationTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    VitalMLInference.init().then((_) => debugPrint('ML model loaded'));
    _initializeCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _validationTimer?.cancel();
    _disableTorch();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    if (state == AppLifecycleState.inactive) {
      _disableTorch();
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera().then((_) {
        if (_controller != null && _controller!.value.isInitialized) {
          final vitalSignController =
              Provider.of<VitalSignController>(context, listen: false);
          vitalSignController.enableLiveMode();
          vitalSignController.attachStreamPipeline(_controller!);
        }
      });
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) {
      if (mounted) {
        setState(() {
          _isInitialized = true;
          _cameraError = 'No cameras found';
        });
      }
      return;
    }

    CameraDescription selectedCamera;
    if (_cameraMode == CameraMode.rear) {
      selectedCamera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );
    } else {
      selectedCamera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras.first,
      );
    }

    try {
      _controller = CameraController(
        selectedCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize().timeout(
        const Duration(seconds: 10),
        onTimeout: () {
          throw Exception('Camera initialization timed out');
        },
      );

      if (!mounted) return;
      
      try {
        await NativeHardwareBridge.lockHardwareISP().timeout(
          const Duration(seconds: 2),
          onTimeout: () {
            debugPrint('Hardware ISP lock timed out - continuing without lock');
          },
        );
      } catch (e) {
        debugPrint('Hardware ISP lock failed: $e');
      }

      try {
        await _applyAdaptivePpgSettings().timeout(
          const Duration(seconds: 3),
          onTimeout: () {
            debugPrint('Adaptive PPG settings timed out - continuing with defaults');
          },
        );
      } catch (e) {
        debugPrint('Adaptive PPG settings failed: $e');
      }

      setState(() {
        _isInitialized = true;
      });

      // Enable torch for rear camera PPG
      if (_cameraMode == CameraMode.rear && _torchEnabled) {
        await _enableTorch();
      } else if (_cameraMode == CameraMode.front) {
        await _disableTorch();
      }

      final vitalSignController =
          Provider.of<VitalSignController>(context, listen: false);
      vitalSignController.enableLiveMode();
      vitalSignController.attachStreamPipeline(_controller!);
    } catch (e) {
      if (mounted) {
        setState(() {
          _isInitialized = true;
          _cameraError = e.toString();
        });
      }
      debugPrint('Camera initialization error: $e');
    }
  }

  Future<void> _switchCameraMode(CameraMode mode) async {
    if (mode == _cameraMode) return;
    
    _cameraMode = mode;
    _disableTorch();
    await _controller?.dispose();
    _controller = null;
    setState(() => _isInitialized = false);
    await _initializeCamera();
  }

  Future<void> _toggleTorch() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    _torchEnabled = !_torchEnabled;
    if (_torchEnabled) {
      await _enableTorch();
    } else {
      await _disableTorch();
    }
    setState(() {});
  }

  Future<void> _enableTorch() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    try {
      final hasFlash = _controller!.value.flashMode != null;
      if (hasFlash) {
        await _controller!.setFlashMode(FlashMode.torch);
        debugPrint('Torch enabled for PPG measurement');
      }
    } catch (e) {
      debugPrint('Torch not available: $e');
    }
  }

  Future<void> _disableTorch() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    try {
      await _controller!.setFlashMode(FlashMode.off);
    } catch (e) {
      debugPrint('Could not disable torch: $e');
    }
  }

  Future<void> _applyAdaptivePpgSettings() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    final facing = _cameraMode == CameraMode.front ? 0 : 1;
    final caps = await NativeHardwareBridge.optimizeSensorForPPG(facing: facing);
    if (caps == null) {
      debugPrint('Adaptive PPG: could not query native capabilities');
      return;
    }

    debugPrint('Adaptive PPG capabilities: $caps');

    final ppgSettings = caps['ppgOptimizedSettings'] as Map<dynamic, dynamic>?;
    if (ppgSettings != null) {
      debugPrint('Adaptive PPG: native optimized settings = $ppgSettings');
      await _applyPpgOptimizedSettings(ppgSettings);
    }

    final aeCompMin = caps['aeCompMin'] is int ? caps['aeCompMin'] as int : 0;
    final aeCompMax = caps['aeCompMax'] is int ? caps['aeCompMax'] as int : 0;
    final flashAvailable = caps['flashAvailable'] is bool ? caps['flashAvailable'] as bool : false;

    // Underexpose to prevent red/green channel clipping above 250
    try {
      if (aeCompMin < 0) {
        await _controller!.setExposureOffset(aeCompMin.toDouble());
        debugPrint('Adaptive PPG: exposure offset set to $aeCompMin');
      }
    } catch (e) {
      debugPrint('Adaptive PPG: setExposureOffset failed: $e');
    }

    // Lock focus if the controller exposes it; otherwise continue
    try {
      final focusMode = _controller!.value.focusMode;
      debugPrint('Adaptive PPG: current focus mode = $focusMode');
    } catch (e) {
      debugPrint('Adaptive PPG: focus mode query failed: $e');
    }

    // Ensure torch is on for rear-camera fingertip PPG
    if (flashAvailable && _cameraMode == CameraMode.rear && _torchEnabled) {
      try {
        await _controller!.setFlashMode(FlashMode.torch);
        debugPrint('Adaptive PPG: torch enabled');
      } catch (e) {
        debugPrint('Adaptive PPG: torch enable failed: $e');
      }
    }
  }

  Future<void> _applyPpgOptimizedSettings(Map<dynamic, dynamic> settings) async {
    try {
      final shouldLockAe = settings['shouldLockAe'] as bool? ?? false;
      final shouldLockAwb = settings['shouldLockAwb'] as bool? ?? false;
      final shouldLockAf = settings['shouldLockAf'] as bool? ?? false;
      final shouldDisableFlash = settings['shouldDisableFlash'] as bool? ?? false;
      final hasFixedFocus = settings['hasFixedFocus'] as bool? ?? false;
      final lensFocusDistance = settings['lensFocusDistance'] as double? ?? 0.0;

      debugPrint('Applying PPG optimized settings: lockAE=$shouldLockAe, lockAWB=$shouldLockAwb, lockAF=$shouldLockAf, disableFlash=$shouldDisableFlash, fixedFocus=$hasFixedFocus, focusDistance=$lensFocusDistance');

      // Disable flash for front camera (rPPG) only; rear camera uses torch
      if (shouldDisableFlash) {
        await _controller!.setFlashMode(FlashMode.off);
        debugPrint('Adaptive PPG: flash disabled');
      }

      // For front camera (rPPG), we want to lock AE/AWB/AF to prevent skin-tone drift
      // For rear camera (PPG), we also lock but rely on torch for illumination
      if (_cameraMode == CameraMode.front) {
        // Front camera: apply exposure compensation for stable skin-tone capture
        // The native side provides targetExposureNs and targetIso but Flutter camera plugin
        // only supports exposureOffset. We use the aeCompMin from capabilities.
        final aeCompMin = settings['aeCompMin'] as int? ?? 0;
        if (aeCompMin < 0) {
          await _controller!.setExposureOffset(aeCompMin.toDouble());
          debugPrint('Adaptive PPG (front): exposure offset set to $aeCompMin');
        }
      }

      // Note: Flutter camera plugin doesn't expose direct AE/AWB/AF lock controls
      // These would require platform-specific implementation or a custom camera plugin
      // The native query provides the optimal values for reference
    } catch (e) {
      debugPrint('Adaptive PPG: applying optimized settings failed: $e');
    }
  }

  void _toggleMeasurementMode() {
    setState(() {
      _measurementMode = _measurementMode == MeasurementMode.live 
          ? MeasurementMode.validation 
          : MeasurementMode.live;
      if (_measurementMode == MeasurementMode.validation) {
        _startValidationMode();
      } else {
        _stopValidationMode();
      }
    });
  }

  void _startValidationMode() {
    _validationData.clear();
    _groundTruth = {
      'bpm': null,
      'spo2': null,
      'hb': null,
      'bpSys': null,
      'bpDia': null,
      'glucose': null,
      'respRate': null,
    };
    
    _validationTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      final controller = Provider.of<VitalSignController>(context, listen: false);
      if (controller.currentState == TrackingState.complete) {
        _validationData.add({
          'timestamp': DateTime.now().toIso8601String(),
          'appBpm': controller.heartRate,
          'appHb': controller.hbLevel,
          'appSpO2': _calculateSpO2(controller),
        });
      }
    });
  }

  void _stopValidationMode() {
    _validationTimer?.cancel();
    _validationTimer = null;
  }

  double? _calculateSpO2(VitalSignController controller) {
    // This would need access to red/blue buffers - placeholder
    return null;
  }

  void _showGroundTruthDialog() {
    final bpmController = TextEditingController(text: _groundTruth?['bpm']?.toString() ?? '');
    final spo2Controller = TextEditingController(text: _groundTruth?['spo2']?.toString() ?? '');
    final hbController = TextEditingController(text: _groundTruth?['hb']?.toString() ?? '');
    final bpSysController = TextEditingController(text: _groundTruth?['bpSys']?.toString() ?? '');
    final bpDiaController = TextEditingController(text: _groundTruth?['bpDia']?.toString() ?? '');
    final glucoseController = TextEditingController(text: _groundTruth?['glucose']?.toString() ?? '');
    final respController = TextEditingController(text: _groundTruth?['respRate']?.toString() ?? '');

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Enter Ground Truth Values'),
        backgroundColor: Colors.grey[900],
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildDialogField('Heart Rate (BPM)', bpmController, TextInputType.number),
              _buildDialogField('SpO₂ (%)', spo2Controller, TextInputType.number),
              _buildDialogField('Hb (g/dL)', hbController, TextInputType.number),
              _buildDialogField('BP Systolic (mmHg)', bpSysController, TextInputType.number),
              _buildDialogField('BP Diastolic (mmHg)', bpDiaController, TextInputType.number),
              _buildDialogField('Glucose (mg/dL)', glucoseController, TextInputType.number),
              _buildDialogField('Resp Rate (/min)', respController, TextInputType.number),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              setState(() {
                _groundTruth = {
                  'bpm': double.tryParse(bpmController.text),
                  'spo2': double.tryParse(spo2Controller.text),
                  'hb': double.tryParse(hbController.text),
                  'bpSys': double.tryParse(bpSysController.text),
                  'bpDia': double.tryParse(bpDiaController.text),
                  'glucose': double.tryParse(glucoseController.text),
                  'respRate': double.tryParse(respController.text),
                };
              });
              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Widget _buildDialogField(String label, TextEditingController controller, TextInputType type) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: TextField(
        controller: controller,
        keyboardType: type,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: Colors.grey),
          enabledBorder: OutlineInputBorder(
            borderSide: const BorderSide(color: Colors.grey),
            borderRadius: BorderRadius.circular(8),
          ),
          focusedBorder: OutlineInputBorder(
            borderSide: const BorderSide(color: Colors.red),
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
    );
  }

  void _exportValidationData() {
    if (_validationData.isEmpty) return;
    
    final csv = StringBuffer();
    csv.writeln('timestamp,app_bpm,app_hb,app_spo2,gt_bpm,gt_spo2,gt_hb,gt_bp_sys,gt_bp_dia,gt_glucose,gt_resp_rate');
    
    for (final point in _validationData) {
      csv.write('${point['timestamp']},');
      csv.write('${point['appBpm'] ?? ''},');
      csv.write('${point['appHb'] ?? ''},');
      csv.write('${point['appSpO2'] ?? ''},');
      csv.write('${_groundTruth?['bpm'] ?? ''},');
      csv.write('${_groundTruth?['spo2'] ?? ''},');
      csv.write('${_groundTruth?['hb'] ?? ''},');
      csv.write('${_groundTruth?['bpSys'] ?? ''},');
      csv.write('${_groundTruth?['bpDia'] ?? ''},');
      csv.write('${_groundTruth?['glucose'] ?? ''},');
      csv.writeln('${_groundTruth?['respRate'] ?? ''}');
    }
    
    // In a real app, you'd use share_plus or path_provider to save the file
    debugPrint('Validation CSV:\n${csv.toString()}');
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Exported ${_validationData.length} validation points (check console)')),
    );
  }

  Future<void> _exportRawSignals(VitalSignController controller) async {
    try {
      // Include ground truth if available
      final groundTruth = _groundTruth?.isNotEmpty == true ? _groundTruth : null;
      
      // Get patient ID from validation data if available
      final patientId = _validationData.isNotEmpty ? 'patient_${_validationData.first['timestamp']?.toString().split('T').first}' : null;
      
      // Export raw signals as JSON
      final jsonString = controller.exportRawSignalsJson(
        groundTruth: groundTruth,
        sessionId: 'session_${DateTime.now().millisecondsSinceEpoch}',
        patientId: patientId,
      );
      
      // Save to file
      final directory = await getApplicationDocumentsDirectory();
      final fileName = 'ppg_signals_${DateTime.now().toIso8601String().replaceAll(':', '-')}.json';
      final file = File('${directory.path}/$fileName');
      await file.writeAsString(jsonString);
      
      debugPrint('Exported raw signals to: ${file.path}');
      
      // Share the file
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'PPG Raw Signals Export - ${controller.cameraMode.name} camera',
        subject: 'PPG Training Data Export',
      );
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Exported raw signals (${(jsonString.length / 1024).toStringAsFixed(1)} KB)'),
          duration: const Duration(seconds: 3),
        ),
      );
    } catch (e) {
      debugPrint('Export raw signals failed: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Export failed: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _toggleInfoOverlay() {
    setState(() {
      _showInfoOverlay = !_showInfoOverlay;
    });
  }

  void _resetMeasurement() {
    final vitalSignController =
        Provider.of<VitalSignController>(context, listen: false);
    vitalSignController.reset();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Measurement reset - place finger over camera to start'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = Provider.of<VitalSignController>(context);

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                _buildHeader(),
                const SizedBox(height: 8),
                Expanded(
                  child: ClipRect(
                    child: _buildCameraPreview(),
                  ),
                ),
                _buildWaveformSection(controller),
                _buildMetricsSection(controller),
                const SizedBox(height: 8),
                _buildControlsSection(controller),
                const SizedBox(height: 16),
              ],
            ),
            if (_showInfoOverlay) _buildInfoOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          const Icon(Icons.favorite, color: Colors.red, size: 28),
          const SizedBox(width: 8),
          const Text(
            'PPG Vital Monitor',
            style: TextStyle(
              color: Colors.red,
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Spacer(),
          _buildModeBadge(),
          const SizedBox(width: 8),
          _buildCameraModeSelector(),
        ],
      ),
    );
  }

  Widget _buildModeBadge() {
    Color color;
    String text;
    IconData icon;
    
    if (_measurementMode == MeasurementMode.validation) {
      color = Colors.orange;
      text = 'VALIDATION';
      icon = Icons.science;
    } else if (controller.isDemoMode) {
      color = Colors.orange;
      text = 'DEMO';
      icon = Icons.science;
    } else {
      color = Colors.green;
      text = 'LIVE';
      icon = Icons.circle;
    }
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 12),
          const SizedBox(width: 4),
          Text(text, style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildCameraModeSelector() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _cameraButton(CameraMode.front, Icons.camera_front, 'Front'),
        const SizedBox(width: 4),
        _cameraButton(CameraMode.rear, Icons.camera_rear, 'Rear'),
      ],
    );
  }

  Widget _cameraButton(CameraMode mode, IconData icon, String label) {
    final isSelected = _cameraMode == mode;
    return GestureDetector(
      onTap: () => _switchCameraMode(mode),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? Colors.red : Colors.grey[800],
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: isSelected ? Colors.red : Colors.grey[700]!),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: isSelected ? Colors.white : Colors.grey, size: 16),
            const SizedBox(width: 4),
            Text(label, style: TextStyle(
              color: isSelected ? Colors.white : Colors.grey,
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (_controller != null && _controller!.value.isInitialized) {
      return Stack(
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: 1.0 / _controller!.value.aspectRatio,
              child: CameraPreview(_controller!),
            ),
          ),
          if (_cameraMode == CameraMode.rear)
            Positioned(
              top: 8,
              right: 8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _torchEnabled ? Icons.flash_on : Icons.flash_off,
                      color: _torchEnabled ? Colors.amber : Colors.grey,
                      size: 16,
                    ),
                    const SizedBox(width: 4),
                    GestureDetector(
                      onTap: _toggleTorch,
                      child: Text(
                        _torchEnabled ? 'Flash ON' : 'Flash OFF',
                        style: TextStyle(
                          color: _torchEnabled ? Colors.amber : Colors.grey,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      );
    }

    return Container(
      color: Colors.black,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _cameraError ?? 'Initializing camera...',
              style: const TextStyle(color: Colors.grey, fontSize: 14),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWaveformSection(VitalSignController controller) {
    return Container(
      height: 100,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
      ),
      child: Stack(
        children: [
          WaveformChart(
            data: controller.waveformData,
            color: Colors.redAccent,
            label: _cameraMode == CameraMode.front ? 'rPPG Signal (Front)' : 'PPG Signal (Rear)',
          ),
          Positioned(
            top: 4,
            right: 4,
            child: _buildSignalQualityIndicator(controller),
          ),
        ],
      ),
    );
  }

  Widget _buildSignalQualityIndicator(VitalSignController controller) {
    final sqi = controller.signalQuality;
    
    Color color;
    if (sqi > 60) color = Colors.green;
    else if (sqi > 30) color = Colors.orange;
    else color = Colors.red;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.signal_cellular_alt, color: color, size: 10),
          const SizedBox(width: 2),
          Text('SQI: ${sqi.toStringAsFixed(0)}%', style: TextStyle(color: color, fontSize: 9, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildMetricsSection(VitalSignController controller) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        children: [
          Text(
            controller.toastPrompt,
            style: const TextStyle(color: Colors.white, fontSize: 14),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          if (controller.currentState == TrackingState.profiling)
            LinearProgressIndicator(
              value: controller.gatheringProgress,
              backgroundColor: Colors.grey[800],
              valueColor: const AlwaysStoppedAnimation<Color>(Colors.red),
            ),
          if (controller.currentState == TrackingState.complete)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Heart Rate: ${controller.heartRate.toStringAsFixed(1)} BPM',
                style: const TextStyle(
                  color: Colors.green,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          if (controller.currentState == TrackingState.complete)
            _buildMetricsGrid(controller),
          const SizedBox(height: 4),
          const Text(
            'Estimates only — not clinical measurements',
            style: TextStyle(color: Colors.orange, fontSize: 10, fontStyle: FontStyle.italic),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsGrid(VitalSignController controller) {
    final isValidation = _measurementMode == MeasurementMode.validation;
    final gtBpm = _groundTruth?['bpm'];
    final gtSpO2 = _groundTruth?['spo2'];
    final gtHb = _groundTruth?['hb'];
    final gtBpSys = _groundTruth?['bpSys'];
    final gtBpDia = _groundTruth?['bpDia'];
    final gtGlucose = _groundTruth?['glucose'];
    final gtResp = _groundTruth?['respRate'];
    
    // Use the controller's computed values from PPGSignalEngine
    final spo2 = controller.spo2Level > 0 ? controller.spo2Level : null;
    final bpSys = controller.bpSys > 0 ? controller.bpSys : null;
    final bpDia = controller.bpDia > 0 ? controller.bpDia : null;
    final glucose = controller.glucoseLevel > 0 ? controller.glucoseLevel : null;
    final respRate = controller.respRate > 0 ? controller.respRate : null;

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: [
        _buildMetricCard(
          'Heart Rate',
          '${controller.heartRate.toStringAsFixed(1)} BPM',
          Icons.favorite,
          Colors.red,
          gtValue: gtBpm,
          unit: 'BPM',
        ),
        _buildMetricCard(
          'SpO₂',
          spo2 != null ? '${spo2.toStringAsFixed(1)}%' : '--',
          Icons.air,
          Colors.cyan,
          gtValue: gtSpO2,
          unit: '%',
        ),
        _buildMetricCard(
          'Hb',
          '${controller.hbLevel.toStringAsFixed(1)} g/dL',
          Icons.bloodtype,
          Colors.pink,
          gtValue: gtHb,
          unit: 'g/dL',
        ),
        _buildMetricCard(
          'BP',
          bpSys != null && bpDia != null ? '$bpSys/$bpDia' : '--',
          Icons.monitor_heart,
          Colors.green,
          gtValue: gtBpSys != null && gtBpDia != null ? '${gtBpSys.toInt()}/${gtBpDia.toInt()}' : null,
          unit: 'mmHg',
        ),
        _buildMetricCard(
          'Glucose',
          glucose != null ? '${glucose.toInt()} mg/dL' : '--',
          Icons.local_dining,
          Colors.amber,
          gtValue: gtGlucose,
          unit: 'mg/dL',
        ),
        _buildMetricCard(
          'Resp Rate',
          respRate != null ? '${respRate.toStringAsFixed(1)} /min' : '--',
          Icons.airwave,
          Colors.blue,
          gtValue: gtResp,
          unit: '/min',
        ),
      ],
    );
  }

  Widget _buildMetricCard(String label, String value, IconData icon, Color color, 
      {dynamic gtValue, String? unit}) {
    String? diff;
    if (gtValue != null) {
      final appValue = _extractNumericValue(value);
      if (appValue != null) {
        final difference = (appValue - gtValue).abs();
        diff = 'Δ${difference.toStringAsFixed(1)}${unit ?? ''}';
      }
    }
    
    return Container(
      width: 140,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(color: Colors.grey, fontSize: 10),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 14),
            textAlign: TextAlign.center,
          ),
          if (diff != null)
            Text(
              diff,
              style: const TextStyle(color: Colors.orange, fontSize: 9),
            ),
          if (isValidation && gtValue != null)
            Text(
              'GT: ${gtValue}${unit != null ? ' $unit' : ''}',
              style: const TextStyle(color: Colors.blue, fontSize: 9),
            ),
        ],
      ),
    );
  }

  double? _extractNumericValue(String value) {
    final match = RegExp(r'([\d.]+)').firstMatch(value);
    return match != null ? double.tryParse(match.group(1)!) : null;
  }

  Widget _buildControlsSection(VitalSignController controller) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _controlButton(
                icon: Icons.refresh,
                label: 'Reset',
                color: Colors.red,
                onPressed: _resetMeasurement,
              ),
              _controlButton(
                icon: _measurementMode == MeasurementMode.validation ? Icons.science : Icons.science_outlined,
                label: 'Validation',
                color: _measurementMode == MeasurementMode.validation ? Colors.orange : Colors.grey,
                onPressed: _toggleMeasurementMode,
              ),
              _controlButton(
                icon: Icons.edit,
                label: 'Ground Truth',
                color: Colors.blue,
                onPressed: _showGroundTruthDialog,
              ),
              if (_measurementMode == MeasurementMode.validation)
                _controlButton(
                  icon: Icons.download,
                  label: 'Export Val.',
                  color: Colors.green,
                  onPressed: _exportValidationData,
                ),
              _controlButton(
                icon: Icons.save_alt,
                label: 'Export Signals',
                color: Colors.purple,
                onPressed: () => _exportRawSignals(controller),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _controlButton(
                icon: _showInfoOverlay ? Icons.info : Icons.info_outline,
                label: _showInfoOverlay ? 'Hide Info' : 'Show Info',
                color: Colors.blue,
                onPressed: _toggleInfoOverlay,
              ),
              Text(
                'PPG Vital Monitor v1.0 • ${_cameraMode == CameraMode.front ? "rPPG (Face)" : "PPG (Finger)"}',
                style: const TextStyle(color: Colors.grey, fontSize: 10),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _controlButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withOpacity(0.5)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color, size: 18),
            const SizedBox(height: 2),
            Text(label, style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black87,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'PPG Vital Signs Monitor',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: _toggleInfoOverlay,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildInfoSection('Camera Modes:', [
                'Front Camera (rPPG): Face-based remote PPG using green channel',
                'Rear Camera + Flash (PPG): Fingertip transmission PPG with LED illumination',
                'Switch modes using the camera selector in the header',
              ]),
              const SizedBox(height: 16),
              _buildInfoSection('Flash Control:', [
                'Rear mode: Flash/LED stays ON for transillumination PPG',
                'Front mode: Flash OFF (uses ambient light)',
                'Tap flash indicator to toggle manually',
                'Auto-exposure compensation prevents saturation',
              ]),
              const SizedBox(height: 16),
              _buildInfoSection('Measurement Modes:', [
                'Live: Real-time monitoring with adaptive signal quality',
                'Validation: Records app vs ground truth for Bland-Altman analysis',
                'Ground Truth: Enter clinical reference values for comparison',
                'Export: CSV export of validation data for statistical analysis',
              ]),
              const SizedBox(height: 16),
              _buildInfoSection('Signal Quality:', [
                'SQI (Signal Quality Index): Real-time perfusion index estimate',
                'Green = Good (>10%), Orange = Fair (5-10%), Red = Poor (<5%)',
                'Adjust finger pressure if signal is flat or saturated',
              ]),
              const SizedBox(height: 16),
              _buildInfoSection('Validation Protocol:', [
                '1. Enter clinical ground truth values',
                '2. Switch to Validation mode',
                '3. Take multiple measurements',
                '4. Export CSV for Bland-Altman analysis',
                '5. Compute mean bias, LOA, percentage within thresholds',
              ]),
              const SizedBox(height: 16),
              _buildInfoSection('Technical Notes:', [
                'Sampling rate: ~30 FPS from camera preview',
                'Bandpass filter: 0.5-3.5 Hz (4th-order Butterworth SOS)',
                'Perfusion index threshold: 0.5% for signal quality',
                'Native Android hardware ISP locking via MethodChannel',
                'Zero-allocation frame processing prevents GC pressure',
                'Peak detection algorithm for heart rate calculation',
                'SpO₂ via Red/Blue AC/DC ratio (not clinical-grade without IR)',
                'BP/Glucose estimated by ML model (requires ground-truth training)',
              ]),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoSection(String title, List<String> items) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            color: Colors.lightGreen,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        ...items.map((item) => Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(
            '• $item',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 13,
              height: 1.4,
            ),
          ),
        )),
      ],
    );
  }
}