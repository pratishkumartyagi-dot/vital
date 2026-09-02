import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { readings, format } = body;
    
    if (!readings || !Array.isArray(readings) || readings.length === 0) {
      return NextResponse.json(
        { error: "No readings provided" },
        { status: 400 }
      );
    }
    
    const headers = [
      "ID",
      "Timestamp",
      "Patient Name",
      "Age",
      "Sex",
      "LMP",
      "Pregnancy",
      "Heart Rate (BPM)",
      "SpO2 (%)",
      "Hemoglobin (g/dL)",
      "BP Systolic (mmHg)",
      "BP Diastolic (mmHg)",
      "Glucose (mg/dL)",
      "Signal Quality (%)",
      "Respiratory Rate (/min)",
      "HRV SDNN (ms)",
      "HRV RMSSD (ms)",
      "HRV pNN50 (%)",
      "Rhythm",
      "Video Available",
    ];
    
    const rows = readings.map((r: any) => [
      r.id,
      r.timestamp,
      r.patient?.name || "",
      r.patient?.age || "",
      r.patient?.sex || "",
      r.patient?.lmp || "",
      r.patient?.pregnancy || "",
      r.vitals?.bpm?.toFixed(1) || "0",
      r.vitals?.spo2?.toString() || "0",
      r.vitals?.hb?.toFixed(1) || "0",
      r.vitals?.bpSys?.toString() || "0",
      r.vitals?.bpDia?.toString() || "0",
      r.vitals?.sugar?.toString() || "0",
      ((r.vitals?.signalQuality || 0) * 100).toFixed(1),
      r.vitals?.respRate?.toFixed(1) || "0",
      r.vitals?.sdnn?.toFixed(1) || "0",
      r.vitals?.rmssd?.toFixed(1) || "0",
      r.vitals?.pnn50?.toFixed(1) || "0",
      r.vitals?.rhythm || "unknown",
      r.videoData ? "Yes" : "No",
    ]);
    
    const csvContent = [headers, ...rows]
      .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    
    const filename = `vital-readings-${new Date().toISOString().split("T")[0]}.${format === "excel" ? "xls" : "csv"}`;
    const mimeType = format === "excel" 
      ? "application/vnd.ms-excel" 
      : "text/csv;charset=utf-8";
    
    return new NextResponse(csvContent, {
      headers: {
        "Content-Type": mimeType,
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Failed to export readings" },
      { status: 500 }
    );
  }
}