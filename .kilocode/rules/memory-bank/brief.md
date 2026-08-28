# Project Brief: Vital Monitoring App

## Purpose

This is a research-grade remote photoplethysmography (rPPG) application for measuring vital signs from video. It combines a Next.js web interface for real-time monitoring with the MCD-rPPG training pipeline for model development.

## Target Users

- Researchers working on rPPG and health biomarker estimation
- Developers building health monitoring applications
- Teams needing a standardized platform for PPG signal processing

## Core Use Case

1. Real-time vital sign monitoring via webcam (web app)
2. Model training and evaluation using the MCD-rPPG dataset
3. Research documentation and validation tools

## Key Requirements

### Must Have
- Real-time PPG signal acquisition from webcam
- MCD-rPPG model training pipeline
- Heart rate and SpO2 estimation
- Training dashboard for model management
- Next.js 16 with App Router
- TypeScript for type safety
- Tailwind CSS 4 for styling

### Nice to Have
- Database for storing measurement history
- User authentication
- Model export/deployment tools
- Advanced signal processing visualization

## Success Metrics
- Working real-time PPG monitoring
- Successful model training pipeline
- Clean, maintainable codebase
- Passing type checks and lint

## Constraints
- Framework: Next.js 16 + React 19 + Tailwind CSS 4
- Package manager: Bun
- Python training pipeline requires PyTorch
