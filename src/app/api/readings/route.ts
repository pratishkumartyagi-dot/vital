import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    message: "Readings are stored client-side in localStorage. Use the client-side vital-storage.ts module.",
    maxReadings: 50,
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    return NextResponse.json({
      success: true,
      message: "Reading received (client-side storage handles persistence)",
      id: body.id || "local",
    });
  } catch {
    return NextResponse.json(
      { error: "Invalid request body" },
      { status: 400 }
    );
  }
}

export async function DELETE() {
  return NextResponse.json({
    success: true,
    message: "Cleared (client-side storage handles this via vital-storage.ts)",
  });
}