# Product Context: Vital Monitoring App

## Why This App Exists

Remote photoplethysmography (rPPG) enables non-contact measurement of vital signs like heart rate and blood oxygen saturation from video. This application provides both a real-time monitoring interface and a complete training pipeline for developing rPPG models.

## Problems It Solves

1. **Research Tooling**: Provides a complete platform for rPPG research
2. **Real-time Monitoring**: Web-based PPG signal acquisition without specialized hardware
3. **Model Training**: Integrated MCD-rPPG training pipeline for model development
4. **Validation**: Cross-validation framework for model evaluation

## How It Works

1. User opens the web app
2. Camera captures video frames
3. Signal processing extracts PPG from facial regions
4. Heart rate and SpO2 are estimated in real-time
5. Training dashboard allows model configuration and monitoring

## Key User Experience Goals

- **Immediate Usability**: Works directly in browser with webcam
- **Research Ready**: Complete training pipeline included
- **Clear Visualization**: Real-time waveform and signal quality indicators
- **Professional UI**: Dark theme with medical/research aesthetic

## Integration Points

- **MCD-rPPG**: Training models based on the MCD-rPPG dataset
- **Signal Processing**: POS algorithm and bandpass filtering
- **Model Evaluation**: 5-fold cross-validation with HR/PPG metrics
