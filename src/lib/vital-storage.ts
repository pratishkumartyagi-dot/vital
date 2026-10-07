export interface VitalReading {
  id: string;
  timestamp: string;
  patient: {
    name: string;
    age: string;
    sex: "male" | "female";
    lmp?: string;
    pregnancy?: "pregnant" | "non-pregnant";
    menopause?: "yes" | "no";
  };
  vitals: {
    bpm: number;
    spo2: number;
    hb: number;
    bpSys: number;
    bpDia: number;
    sugar: number;
    signalQuality: number;
    respRate: number;
    sdnn: number;
    rmssd: number;
    pnn50: number;
    creatinine: number;
    egfr: number;
    rhythm: "regular" | "irregular" | "unknown";
  };
  videoData?: string;
  videoMimeType?: string;
  cameraMode?: "front" | "rear";
  demoMode?: boolean;
  spo2Offset?: number;
}

export interface ValidationStats {
  count: number;
  bpm: ValidationMetricStats;
  spo2: ValidationMetricStats;
  hb: ValidationMetricStats;
  respRate: ValidationMetricStats;
  bpSys: ValidationMetricStats;
  bpDia: ValidationMetricStats;
  sugar: ValidationMetricStats;
  sdnn: ValidationMetricStats;
  rmssd: ValidationMetricStats;
  pnn50: ValidationMetricStats;
  creatinine: ValidationMetricStats;
}

export interface ValidationMetricStats {
  count: number;
  meanBias: number;
  mae: number;
  rmse: number;
  std: number;
  median: number;
  loaLower: number;
  loaUpper: number;
  within5?: number;
  within10?: number;
  within2?: number;
  within3?: number;
  within1?: number;
  within2Hb?: number;
  withinCreatinine?: number;
}

const MAX_READINGS = 50;
const STORAGE_KEY = "vital_readings";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function getStoredReadings(): VitalReading[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveReadings(readings: VitalReading[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(readings));
}

export async function addVitalReading(
  reading: Omit<VitalReading, "id" | "timestamp">,
  videoBlob?: Blob
): Promise<VitalReading> {
  // Try to save to database first
  try {
    const videoBase64 = videoBlob 
      ? await blobToBase64(videoBlob)
      : undefined;
    
    const response = await fetch("/api/readings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient: reading.patient,
        vitals: reading.vitals,
        cameraMode: reading.cameraMode,
        demoMode: reading.demoMode,
        spo2Offset: reading.spo2Offset,
        videoBlob: videoBase64,
        videoMimeType: videoBlob?.type,
      }),
    });
    
    if (response.ok) {
      const result = await response.json();
      return { ...reading, id: result.id, timestamp: new Date().toISOString() } as VitalReading;
    }
  } catch (error) {
    console.warn("Database save failed, falling back to localStorage:", error);
  }
  
  // Fallback to localStorage
  const readings = getStoredReadings();
  
  const newReading: VitalReading = {
    ...reading,
    id: generateId(),
    timestamp: new Date().toISOString(),
    videoData: videoBlob ? URL.createObjectURL(videoBlob) : undefined,
    videoMimeType: videoBlob?.type,
  };
  
  readings.unshift(newReading);
  
  if (readings.length > MAX_READINGS) {
    const removed = readings.splice(MAX_READINGS);
    removed.forEach(r => {
      if (r.videoData) URL.revokeObjectURL(r.videoData);
    });
  }
  
  saveReadings(readings);
  return newReading;
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function getVitalReadings(params?: {
  limit?: number;
  offset?: number;
  cameraMode?: "front" | "rear";
  demoMode?: boolean;
  startDate?: string;
  endDate?: string;
}): Promise<VitalReading[]> {
  // Try database first
  try {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set("limit", params.limit.toString());
    if (params?.offset) searchParams.set("offset", params.offset.toString());
    if (params?.cameraMode) searchParams.set("cameraMode", params.cameraMode);
    if (params?.demoMode !== undefined) searchParams.set("demoMode", params.demoMode.toString());
    if (params?.startDate) searchParams.set("startDate", params.startDate);
    if (params?.endDate) searchParams.set("endDate", params.endDate);
    
    const response = await fetch(`/api/readings?${searchParams}`);
    if (response.ok) {
      const result = await response.json();
       return result.readings.map((r: any) => ({
         ...r,
         patient: {
           name: r.patientName,
           age: r.patientAge,
           sex: r.patientSex,
           lmp: r.patientLmp || undefined,
           pregnancy: r.patientPregnancy || undefined,
           menopause: r.patientMenopause || undefined,
         },
        vitals: {
          bpm: r.bpm,
          spo2: r.spo2,
          hb: r.hb,
          bpSys: r.bpSys,
          bpDia: r.bpDia,
          sugar: r.sugar,
          signalQuality: r.signalQuality,
          respRate: r.respRate,
          sdnn: r.sdnn,
          rmssd: r.rmssd,
          pnn50: r.pnn50,
          creatinine: r.creatinine ?? 0,
          egfr: r.egfr ?? 0,
          rhythm: r.rhythm,
        },
        cameraMode: r.cameraMode,
        demoMode: r.demoMode,
        spo2Offset: r.spo2Offset,
      }));
    }
  } catch (error) {
    console.warn("Database fetch failed, falling back to localStorage:", error);
  }
  
  // Fallback to localStorage
  return getStoredReadings();
}

export async function getVitalReading(id: string): Promise<VitalReading | undefined> {
  try {
    const response = await fetch(`/api/readings?id=${id}`);
    if (response.ok) {
      const result = await response.json();
      const r = result.readings[0];
      if (r) {
         return {
           ...r,
           patient: {
             name: r.patientName,
             age: r.patientAge,
             sex: r.patientSex,
             lmp: r.patientLmp || undefined,
             pregnancy: r.patientPregnancy || undefined,
             menopause: r.patientMenopause || undefined,
           },
          vitals: {
            bpm: r.bpm,
            spo2: r.spo2,
            hb: r.hb,
            bpSys: r.bpSys,
            bpDia: r.bpDia,
            sugar: r.sugar,
            signalQuality: r.signalQuality,
            respRate: r.respRate,
            sdnn: r.sdnn,
            rmssd: r.rmssd,
            pnn50: r.pnn50,
            creatinine: r.creatinine ?? 0,
            egfr: r.egfr ?? 0,
            rhythm: r.rhythm,
          },
        };
      }
    }
  } catch (error) {
    console.warn("Database fetch failed:", error);
  }
  
  return getStoredReadings().find(r => r.id === id);
}

export async function deleteVitalReading(id: string): Promise<boolean> {
  try {
    const response = await fetch(`/api/readings?id=${id}`, { method: "DELETE" });
    if (response.ok) return true;
  } catch (error) {
    console.warn("Database delete failed:", error);
  }
  
  // Fallback
  const readings = getStoredReadings();
  const index = readings.findIndex(r => r.id === id);
  if (index === -1) return false;
  
  const [removed] = readings.splice(index, 1);
  if (removed.videoData) URL.revokeObjectURL(removed.videoData);
  saveReadings(readings);
  return true;
}

export async function clearAllReadings(): Promise<void> {
  try {
    // Could add a bulk delete endpoint if needed
  } catch (error) {
    console.warn("Database clear failed:", error);
  }
  
  const readings = getStoredReadings();
  readings.forEach(r => {
    if (r.videoData) URL.revokeObjectURL(r.videoData);
  });
  saveReadings([]);
}

export function exportReadingsToCSV(readings: VitalReading[]): string {
  if (readings.length === 0) return "";
  
  const headers = [
    "ID",
    "Timestamp",
    "Patient Name",
    "Age",
    "Sex",
    "LMP",
    "Pregnancy",
    "Camera Mode",
    "Demo Mode",
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
    "Serum Creatinine (mg/dL)",
    "eGFR (mL/min/1.73m2)",
    "Rhythm",
    "Video Available",
  ];
  
  const rows = readings.map(r => [
    r.id,
    r.timestamp,
    r.patient.name,
    r.patient.age,
    r.patient.sex,
    r.patient.lmp || "",
    r.patient.pregnancy || "",
    r.cameraMode || "front",
    r.demoMode ? "Yes" : "No",
    r.vitals.bpm.toFixed(1),
    r.vitals.spo2.toString(),
    r.vitals.hb.toFixed(1),
    r.vitals.bpSys.toString(),
    r.vitals.bpDia.toString(),
    r.vitals.sugar.toString(),
    (r.vitals.signalQuality * 100).toFixed(1),
    r.vitals.respRate.toFixed(1),
    r.vitals.sdnn.toFixed(1),
    r.vitals.rmssd.toFixed(1),
    r.vitals.pnn50.toFixed(1),
    r.vitals.creatinine ? r.vitals.creatinine.toFixed(2) : "",
    r.vitals.egfr ? r.vitals.egfr.toFixed(0) : "",
    r.vitals.rhythm,
    r.videoData ? "Yes" : "No",
  ]);
  
  const csvContent = [headers, ...rows]
    .map(row => row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(","))
    .join("\n");
  
  return csvContent;
}

export function exportReadingsToExcel(readings: VitalReading[]): Blob {
  const csv = exportReadingsToCSV(readings);
  return new Blob([csv], { type: "application/vnd.ms-excel" });
}

export async function downloadReadings(readings: VitalReading[], format: "csv" | "excel" = "csv"): Promise<void> {
  if (readings.length === 0) return;
  
  let blob: Blob;
  let filename: string;
  
  if (format === "csv") {
    const csv = exportReadingsToCSV(readings);
    blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    filename = `vital-readings-${new Date().toISOString().split("T")[0]}.csv`;
  } else {
    blob = exportReadingsToExcel(readings);
    filename = `vital-readings-${new Date().toISOString().split("T")[0]}.xls`;
  }
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  
  await clearAllReadings();
}

export async function getReadingCount(): Promise<number> {
  try {
    const readings = await getVitalReadings({ limit: 1 });
    // This is a rough count - could add a count endpoint
    const response = await fetch("/api/readings?limit=1000");
    if (response.ok) {
      const result = await response.json();
      return result.readings.length;
    }
  } catch (error) {
    console.warn("Database count failed:", error);
  }
  return getStoredReadings().length;
}

export function getMaxReadings(): number {
  return MAX_READINGS;
}

// Validation API
export async function createValidationSession(data: {
  subjectId?: string;
  deviceInfo?: string;
  cameraMode: "front" | "rear";
  referenceDevice?: string;
  protocolNotes?: string;
  sex?: "male" | "female";
}) {
  const response = await fetch("/api/validation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "createSession", ...data }),
  });
  return response.json();
}

export async function addValidationMeasurement(data: {
  sessionId: string;
  appBpm?: number;
  appSpo2?: number;
  appHb?: number;
  appRespRate?: number;
  appBpSys?: number;
  appBpDia?: number;
  appSugar?: number;
  appSignalQuality?: number;
  appRhythm?: string;
  appSdnn?: number;
  appRmssd?: number;
  appPnn50?: number;
  appCreatinine?: number;
  gtBpm?: number;
  gtSpo2?: number;
  gtHb?: number;
  gtRespRate?: number;
  gtBpSys?: number;
  gtBpDia?: number;
  gtSugar?: number;
  gtRhythm?: string;
  gtSdnn?: number;
  gtRmssd?: number;
  gtPnn50?: number;
  gtCreatinine?: number;
  isValid?: boolean;
  exclusionReason?: string;
}) {
  const response = await fetch("/api/validation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "addMeasurement", ...data }),
  });
  return response.json();
}

export async function getValidationStatistics(params?: {
  sessionId?: string;
  cameraMode?: "front" | "rear";
}): Promise<ValidationStats | null> {
  try {
    const response = await fetch("/api/validation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "getStatistics", ...params }),
    });
    if (response.ok) {
      const result = await response.json();
      return result.statistics;
    }
  } catch (error) {
    console.warn("Validation statistics fetch failed:", error);
  }
  return null;
}

export async function getValidationSessions(limit = 50) {
  try {
    const response = await fetch(`/api/validation?limit=${limit}`);
    if (response.ok) {
      return (await response.json()).sessions;
    }
  } catch (error) {
    console.warn("Validation sessions fetch failed:", error);
  }
  return [];
}

export async function getValidationMeasurements(sessionId: string, limit = 100) {
  try {
    const response = await fetch(`/api/validation?sessionId=${sessionId}&limit=${limit}`);
    if (response.ok) {
      return (await response.json()).measurements;
    }
  } catch (error) {
    console.warn("Validation measurements fetch failed:", error);
  }
  return [];
}