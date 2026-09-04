import { NextResponse } from "next/server";
import { db } from "@/db";
import { validationSessions, validationMeasurements } from "@/db/schema";
import { desc, eq, and, sql } from "drizzle-orm";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get("sessionId");
    const limit = parseInt(searchParams.get("limit") || "100");
    
    if (sessionId) {
      const measurements = await db.select()
        .from(validationMeasurements)
        .where(eq(validationMeasurements.sessionId, sessionId))
        .orderBy(desc(validationMeasurements.timestamp))
        .limit(limit);
      
      const session = await db.select()
        .from(validationSessions)
        .where(eq(validationSessions.id, sessionId))
        .limit(1);
      
      return NextResponse.json({ session: session[0], measurements });
    }
    
    const sessions = await db.select()
      .from(validationSessions)
      .orderBy(desc(validationSessions.timestamp))
      .limit(limit);
    
    return NextResponse.json({ sessions });
  } catch (error) {
    console.error("Error fetching validation data:", error);
    return NextResponse.json({ error: "Failed to fetch validation data" }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, ...data } = body;
    
    if (action === "createSession") {
      const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const session = {
        id,
        timestamp: new Date(),
        subjectId: data.subjectId || null,
        deviceInfo: data.deviceInfo || null,
        cameraMode: data.cameraMode || "front",
        referenceDevice: data.referenceDevice || null,
        protocolNotes: data.protocolNotes || null,
      };
      
      await db.insert(validationSessions).values(session);
      return NextResponse.json({ success: true, session });
    }
    
    if (action === "addMeasurement") {
      const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      // Calculate differences
      const diffBpm = data.appBpm !== null && data.gtBpm !== null ? data.appBpm - data.gtBpm : null;
      const diffSpo2 = data.appSpo2 !== null && data.gtSpo2 !== null ? data.appSpo2 - data.gtSpo2 : null;
      const diffHb = data.appHb !== null && data.gtHb !== null ? data.appHb - data.gtHb : null;
      const diffRespRate = data.appRespRate !== null && data.gtRespRate !== null ? data.appRespRate - data.gtRespRate : null;
      const diffBpSys = data.appBpSys !== null && data.gtBpSys !== null ? data.appBpSys - data.gtBpSys : null;
      const diffBpDia = data.appBpDia !== null && data.gtBpDia !== null ? data.appBpDia - data.gtBpDia : null;
      const diffSugar = data.appSugar !== null && data.gtSugar !== null ? data.appSugar - data.gtSugar : null;
      const diffSdnn = data.appSdnn !== null && data.gtSdnn !== null ? data.appSdnn - data.gtSdnn : null;
      const diffRmssd = data.appRmssd !== null && data.gtRmssd !== null ? data.appRmssd - data.gtRmssd : null;
      const diffPnn50 = data.appPnn50 !== null && data.gtPnn50 !== null ? data.appPnn50 - data.gtPnn50 : null;
      
      const measurement = {
        id,
        sessionId: data.sessionId,
        timestamp: new Date(),
        appBpm: data.appBpm || null,
        appSpo2: data.appSpo2 || null,
        appHb: data.appHb || null,
        appRespRate: data.appRespRate || null,
        appBpSys: data.appBpSys || null,
        appBpDia: data.appBpDia || null,
        appSugar: data.appSugar || null,
        appSignalQuality: data.appSignalQuality || null,
        appRhythm: data.appRhythm || null,
        appSdnn: data.appSdnn || null,
        appRmssd: data.appRmssd || null,
        appPnn50: data.appPnn50 || null,
        gtBpm: data.gtBpm || null,
        gtSpo2: data.gtSpo2 || null,
        gtHb: data.gtHb || null,
        gtRespRate: data.gtRespRate || null,
        gtBpSys: data.gtBpSys || null,
        gtBpDia: data.gtBpDia || null,
        gtSugar: data.gtSugar || null,
        gtRhythm: data.gtRhythm || null,
        gtSdnn: data.gtSdnn || null,
        gtRmssd: data.gtRmssd || null,
        gtPnn50: data.gtPnn50 || null,
        diffBpm,
        diffSpo2,
        diffHb,
        diffRespRate,
        diffBpSys,
        diffBpDia,
        diffSugar,
        diffSdnn,
        diffRmssd,
        diffPnn50,
        isValid: data.isValid !== false,
        exclusionReason: data.exclusionReason || null,
      };
      
      await db.insert(validationMeasurements).values(measurement);
      return NextResponse.json({ success: true, measurement });
    }
    
    if (action === "getStatistics") {
      const { sessionId, cameraMode } = data;
      
      let whereClause = undefined;
      if (sessionId) {
        whereClause = eq(validationMeasurements.sessionId, sessionId);
      } else if (cameraMode) {
        // Join with sessions to filter by camera mode
        const sessions = await db.select({ id: validationSessions.id })
          .from(validationSessions)
          .where(eq(validationSessions.cameraMode, cameraMode as "front" | "rear"));
        const sessionIds = sessions.map(s => s.id);
        if (sessionIds.length > 0) {
          whereClause = sql`${validationMeasurements.sessionId} IN (${sql.join(sessionIds.map(id => sql`${id}`), sql`, `)})`;
        }
      }
      
      const measurements = await db.select()
        .from(validationMeasurements)
        .where(whereClause || sql`1=1`);
      
      const valid = measurements.filter(m => m.isValid);
      
      const stats = {
        count: valid.length,
        bpm: calculateStats(valid, "bpm"),
        spo2: calculateStats(valid, "spo2"),
        hb: calculateStats(valid, "hb"),
        respRate: calculateStats(valid, "respRate"),
        bpSys: calculateStats(valid, "bpSys"),
        bpDia: calculateStats(valid, "bpDia"),
        sugar: calculateStats(valid, "sugar"),
        sdnn: calculateStats(valid, "sdnn"),
        rmssd: calculateStats(valid, "rmssd"),
        pnn50: calculateStats(valid, "pnn50"),
      };
      
      return NextResponse.json({ statistics: stats });
    }
    
    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Error with validation:", error);
    return NextResponse.json({ error: "Failed to process validation" }, { status: 500 });
  }
}

function calculateStats(measurements: any[], field: string) {
  const appField = `app${field.charAt(0).toUpperCase() + field.slice(1)}`;
  const gtField = `gt${field.charAt(0).toUpperCase() + field.slice(1)}`;
  const diffField = `diff${field.charAt(0).toUpperCase() + field.slice(1)}`;
  
  const diffs = measurements
    .filter(m => m[appField] !== null && m[gtField] !== null)
    .map(m => m[diffField]);
  
  if (diffs.length === 0) return { count: 0 };
  
  const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const mae = diffs.reduce((a, b) => a + Math.abs(b), 0) / diffs.length;
  const rmse = Math.sqrt(diffs.reduce((a, b) => a + b * b, 0) / diffs.length);
  const std = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length);
  
  const sorted = [...diffs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const loaLower = mean - 1.96 * std;
  const loaUpper = mean + 1.96 * std;
  
  // Percentage within thresholds
  const within5 = field === "bpm" ? diffs.filter(d => Math.abs(d) <= 5).length / diffs.length * 100 : 0;
  const within10 = field === "bpm" ? diffs.filter(d => Math.abs(d) <= 10).length / diffs.length * 100 : 0;
  const within2 = field === "spo2" ? diffs.filter(d => Math.abs(d) <= 2).length / diffs.length * 100 : 0;
  const within3 = field === "spo2" ? diffs.filter(d => Math.abs(d) <= 3).length / diffs.length * 100 : 0;
  const within1 = field === "hb" ? diffs.filter(d => Math.abs(d) <= 1).length / diffs.length * 100 : 0;
  const within2Hb = field === "hb" ? diffs.filter(d => Math.abs(d) <= 2).length / diffs.length * 100 : 0;
  
  return {
    count: diffs.length,
    meanBias: mean,
    mae,
    rmse,
    std,
    median,
    loaLower,
    loaUpper,
    within5: within5 || undefined,
    within10: within10 || undefined,
    within2: within2 || undefined,
    within3: within3 || undefined,
    within1: within1 || undefined,
    within2Hb: within2Hb || undefined,
  };
}