import { NextResponse } from "next/server";
import { getDb } from "@/db";
import { vitalReadings, validationMeasurements } from "@/db/schema";
import { desc, eq, and, gte, lte } from "drizzle-orm";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const db = getDb();
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "50");
    const offset = parseInt(searchParams.get("offset") || "0");
    const cameraMode = searchParams.get("cameraMode");
    const demoMode = searchParams.get("demoMode");
    const startDate = searchParams.get("startDate");
    const endDate = searchParams.get("endDate");
    const includeValidation = searchParams.get("includeValidation") === "true";

    const conditions = [];
    if (cameraMode) conditions.push(eq(vitalReadings.cameraMode, cameraMode as "front" | "rear"));
    if (demoMode) conditions.push(eq(vitalReadings.demoMode, demoMode === "true"));
    if (startDate) conditions.push(gte(vitalReadings.timestamp, new Date(startDate)));
    if (endDate) conditions.push(lte(vitalReadings.timestamp, new Date(endDate)));

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;
    
    const readings = await db.select()
      .from(vitalReadings)
      .where(whereClause)
      .orderBy(desc(vitalReadings.timestamp))
      .limit(limit)
      .offset(offset);

    // Optionally include validation measurements
    if (includeValidation) {
      const readingIds = readings.map((r) => r.id);
      const validation = await db.select()
        .from(validationMeasurements)
        .where(eq(validationMeasurements.sessionId, readingIds[0] || ""));
      
      return NextResponse.json({ readings, validation });
    }

    return NextResponse.json({ readings });
  } catch (error) {
    console.error("Error fetching readings:", error);
    return NextResponse.json({ error: "Failed to fetch readings" }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const db = getDb();
    const body = await request.json();
    
    const { patient, vitals, cameraMode, demoMode, spo2Offset, videoBlob, videoMimeType } = body;
    
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const timestamp = new Date();
    
    const reading = {
       id,
       timestamp,
       patientName: patient.name,
       patientAge: patient.age,
       patientSex: patient.sex,
       patientLmp: patient.lmp || null,
       patientPregnancy: patient.pregnancy || null,
       patientMenopause: patient.menopause || null,
       bpm: vitals.bpm || null,
      spo2: vitals.spo2 || null,
      hb: vitals.hb || null,
      bpSys: vitals.bpSys || null,
      bpDia: vitals.bpDia || null,
      sugar: vitals.sugar || null,
      signalQuality: vitals.signalQuality || null,
      respRate: vitals.respRate || null,
      sdnn: vitals.sdnn || null,
      rmssd: vitals.rmssd || null,
      pnn50: vitals.pnn50 || null,
      creatinine: vitals.creatinine || null,
      egfr: vitals.egfr || null,
      rhythm: vitals.rhythm || "unknown",
      cameraMode: cameraMode || "front",
      demoMode: demoMode || false,
      spo2Offset: spo2Offset || 0,
      videoData: videoBlob ? Buffer.from(videoBlob, "base64") : null,
      videoMimeType: videoMimeType || null,
      sessionDurationSec: 130,
      warmupSec: 10,
      measurementSec: 120,
    };

    await db.insert(vitalReadings).values(reading);
    
    return NextResponse.json({ success: true, id, reading });
  } catch (error) {
    console.error("Error saving reading:", error);
    return NextResponse.json({ error: "Failed to save reading" }, { status: 500 });
  }
}

export async function DELETE(request: Request) {
  try {
    const db = getDb();
    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");
    
    if (!id) {
      return NextResponse.json({ error: "ID required" }, { status: 400 });
    }
    
    await db.delete(vitalReadings).where(eq(vitalReadings.id, id));
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting reading:", error);
    return NextResponse.json({ error: "Failed to delete reading" }, { status: 500 });
  }
}