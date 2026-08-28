# System Patterns: Vital Monitoring App

## Architecture Overview

```
/
├── src/
│   └── app/                    # Next.js App Router
│       ├── layout.tsx          # Root layout
│       ├── page.tsx            # Home page
│       ├── globals.css         # Global styles
│       ├── vital-monitoring/
│       │   └── page.tsx        # Real-time PPG monitoring
│       ├── training/
│       │   └── page.tsx        # Training dashboard
│       └── api/
│           └── training/
│               └── route.ts    # Training API
├── training/
│   ├── run_training.py         # Main training script
│   ├── requirements.txt        # Python dependencies
│   └── rppglib/                # MCD-rPPG library
│       ├── models/             # Model implementations
│       ├── train.py            # Training loop
│       ├── processing.py       # Signal processing
│       ├── face_utils.py       # Face detection
│       └── data_utils.py       # Data loading
└── vital_monitoring_app/       # Flutter app (legacy)
```

## Key Design Patterns

### 1. App Router Pattern
- `/` - Home page
- `/vital-monitoring` - Real-time PPG monitoring
- `/training` - Training dashboard
- `/api/training` - Training API

### 2. Client Components
- Vital monitoring page uses "use client" for camera access and real-time processing
- Training dashboard uses "use client" for interactive elements

### 3. Signal Processing Pipeline
```
Camera Frame → ROI Extraction → POS Processing → Bandpass Filter → HR Estimation
```

### 4. Model Training Pipeline
```
Dataset → Preprocessing → Cross-Validation Split → Training → Evaluation → Metrics
```

## Styling Conventions

- Dark theme with red accents for medical/research aesthetic
- Tailwind CSS utility classes
- Responsive design for mobile and desktop
