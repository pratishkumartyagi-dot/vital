import { NextResponse } from "next/server";
import type { FrameData } from "@/lib/rppg-api-client";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { frames, fs = 30.0, useDeepModel = true } = body;

    if (!frames || !Array.isArray(frames) || frames.length < 30) {
      return NextResponse.json(
        { error: "Need at least 30 frames for SpO2 estimation" },
        { status: 400 }
      );
    }

    // Proxy to FastAPI backend (rppg_api.py running on port 8002)
    const apiResponse = await fetch("http://localhost:8002/rppg/spo2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frames,
        fs,
        use_deep_model: useDeepModel,
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!apiResponse.ok) {
      const error = await apiResponse.json().catch(() => ({ detail: "Unknown error" }));
      return NextResponse.json(
        { error: error.detail || `HTTP ${apiResponse.status}` },
        { status: apiResponse.status }
      );
    }

    const result = await apiResponse.json();
    return NextResponse.json(result);
  } catch (error: any) {
    if (error?.name === "TimeoutError") {
      return NextResponse.json(
        { error: "SpO2 estimation timed out" },
        { status: 504 }
      );
    }

    // Fallback: compute STMap features locally and return for client-side inference
    console.warn("FastAPI backend unavailable, returning STMap features for client-side inference:", error);
    return NextResponse.json({
      error: "Backend unavailable",
      fallback: true,
    }, { status: 503 });
  }
}

export async function GET() {
  return NextResponse.json({
    models: ["rosenet", "empirical"],
    description: {
      rosenet: "ROSE-Net deep learning model (Chowdhury et al., BSPC 2025) - cross-domain trained on BIDMC",
      empirical: "AC/DC ratio-of-ratios (110 - 25*R), calibrated for 660nm/940nm",
    },
    default: "rosenet",
  });
}
