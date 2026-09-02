export interface VitalReading {
  id: string;
  timestamp: string;
  patient: {
    name: string;
    age: string;
    sex: "male" | "female";
    lmp?: string;
    pregnancy?: "pregnant" | "non-pregnant";
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
    rhythm: "regular" | "irregular" | "unknown";
  };
  videoData?: string;
  videoMimeType?: string;
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

export function addVitalReading(
  reading: Omit<VitalReading, "id" | "timestamp">,
  videoBlob?: Blob
): VitalReading {
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

export function getVitalReadings(): VitalReading[] {
  return getStoredReadings();
}

export function getVitalReading(id: string): VitalReading | undefined {
  return getStoredReadings().find(r => r.id === id);
}

export function deleteVitalReading(id: string): boolean {
  const readings = getStoredReadings();
  const index = readings.findIndex(r => r.id === id);
  if (index === -1) return false;
  
  const [removed] = readings.splice(index, 1);
  if (removed.videoData) URL.revokeObjectURL(removed.videoData);
  saveReadings(readings);
  return true;
}

export function clearAllReadings(): void {
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
  
  const rows = readings.map(r => [
    r.id,
    r.timestamp,
    r.patient.name,
    r.patient.age,
    r.patient.sex,
    r.patient.lmp || "",
    r.patient.pregnancy || "",
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

export function downloadReadings(readings: VitalReading[], format: "csv" | "excel" = "csv"): void {
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
  
  clearAllReadings();
}

export function getReadingCount(): number {
  return getStoredReadings().length;
}

export function getMaxReadings(): number {
  return MAX_READINGS;
}