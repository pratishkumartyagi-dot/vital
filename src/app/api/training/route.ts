import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { model, dataset, validFold, testFold, device, batchSize } = body;
    
    return NextResponse.json({
      success: true,
      message: 'Training job started',
      config: {
        model: model || 'SCNN_8rois',
        dataset: dataset || 'mcd_rppg',
        validFold: validFold ?? 1,
        testFold: testFold ?? 0,
        device: device || 'cpu',
        batchSize: batchSize || 36,
      }
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Invalid request body' },
      { status: 400 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'ready',
    models: ['SCNN_8rois', 'POS', 'PBV', 'OMIT', 'PhysFormer', 'RhythmFormer', 'TSCAN', 'iBVPNet'],
    datasets: ['mcd_rppg', 'MMPD', 'SCAMPS', 'UBFC_rPPG']
  });
}
