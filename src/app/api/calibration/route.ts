import { NextResponse } from 'next/server';
import { CameraCalibration } from '@/lib/camera-calibration';

export async function GET() {
  try {
    // Get device info from headers or query params
    const model = 'Unknown';
    const manufacturer = 'Unknown';
    const apiLevel = 30;
    
    const calibration = CameraCalibration.getCalibrationInfo(model, manufacturer, apiLevel);
    
    return NextResponse.json({
      zlo: calibration.zlo,
      sensorLinearSlope: calibration.sensorLinearSlope,
      model: calibration.model,
      manufacturer: calibration.manufacturer,
      matched: calibration.matched,
      source: calibration.matched ? 'LOCAL' : 'FALLBACK',
      profile: calibration.profile,
    });
  } catch (error) {
    console.error('Calibration API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch calibration', zlo: 16.0, slope: 1.0, source: 'FALLBACK' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { model, manufacturer, apiLevel } = body;
    
    const calibration = CameraCalibration.getCalibrationInfo(
      model || 'Unknown',
      manufacturer || 'Unknown',
      apiLevel || 30
    );
    
    return NextResponse.json({
      zlo: calibration.zlo,
      sensorLinearSlope: calibration.sensorLinearSlope,
      model: calibration.model,
      manufacturer: calibration.manufacturer,
      matched: calibration.matched,
      source: calibration.matched ? 'LOCAL' : 'FALLBACK',
      profile: calibration.profile,
    });
  } catch (error) {
    console.error('Calibration API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch calibration', zlo: 16.0, slope: 1.0, source: 'FALLBACK' },
      { status: 500 }
    );
  }
}