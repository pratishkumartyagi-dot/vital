# Technical Context: Vital Monitoring App

## Technology Stack

| Technology   | Version | Purpose                         |
| ------------ | ------- | ------------------------------- |
| Next.js      | 16.x    | React framework with App Router |
| React        | 19.x    | UI library                      |
| TypeScript   | 5.9.x   | Type-safe JavaScript            |
| Tailwind CSS | 4.x     | Utility-first CSS               |
| Bun          | Latest  | Package manager & runtime       |
| Drizzle ORM  | 0.45.x  | SQLite database ORM             |
| @kilocode/app-builder-db | 0.0.1 (vendored `vendor/app-builder-db`, 186d1c1 build) | SQLite-via-HTTP Drizzle client |
| PyTorch      | 2.6.0   | Deep learning framework         |
| Python       | 3.10+   | Training pipeline               |

## Development Environment

### Prerequisites
- Bun installed
- Python 3.10+ with pip (for training)

### Commands

```bash
bun install        # Install JS dependencies
bun dev            # Start dev server
bun build          # Production build
bun lint           # Run ESLint
bun typecheck      # Run TypeScript type checking

# Training (separate environment)
cd training
pip install -r requirements.txt
python run_training.py --model SCNN_8rois
```

## Project Configuration

### Next.js Config
- App Router enabled
- Default settings

### TypeScript Config
- Strict mode enabled
- Path alias: `@/*` → `src/*`

### Tailwind CSS 4
- Uses `@tailwindcss/postcss` plugin
- CSS-first configuration

## Training Pipeline

### Models Available
- SCNN_8rois (primary, 8 ROI attention-based CNN)
- POS, PBV, OMIT (baseline methods)
- PhysFormer, RhythmFormer, TSCAN, iBVPNet

### Dataset Format
- Preprocessed `.npy` files with video and PPG
- CSV metadata with fold assignments
- Supports MCD-rPPG, MMPD, SCAMPS, UBFC_rPPG

## File Structure

```
/
├── src/app/                      # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   ├── vital-monitoring/page.tsx
│   ├── training/page.tsx
│   └── api/training/route.ts
├── training/
│   ├── run_training.py
│   ├── requirements.txt
│   └── rppglib/
│       ├── models/
│       ├── train.py
│       ├── processing.py
│       ├── face_utils.py
│       └── data_utils.py
└── vital_monitoring_app/         # Flutter app (legacy)
```
