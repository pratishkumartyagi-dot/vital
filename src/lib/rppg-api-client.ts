"use client";

export interface FrameData {
  frame: string;
  landmarks?: number[][];
}

export interface RPPGRequest {
  frames: FrameData[];
  fs?: number;
  method?: "POS" | "CHROM" | "PBV" | "OMIT" | "GREEN";
  roi?: "face" | "forehead" | "left_cheek" | "right_cheek" | "nose";
}

export interface RPPGResponse {
  ppg: number[];
  hr: number | null;
  snr: number | null;
  method: string;
  quality: number | null;
}

export async function extractRPPG(request: RPPGRequest): Promise<RPPGResponse> {
  const response = await fetch("/api/rppg/rppg", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function extractRPPGBatch(
  frames: Blob[],
  fs = 30,
  method: "POS" | "CHROM" | "PBV" | "OMIT" | "GREEN" = "POS",
  roi: "face" | "forehead" | "left_cheek" | "right_cheek" | "nose" = "face"
): Promise<RPPGResponse> {
  const formData = new FormData();
  frames.forEach((frame, i) => {
    formData.append("frames", frame, `frame_${i}.jpg`);
  });
  formData.append("fs", fs.toString());
  formData.append("method", method);
  formData.append("roi", roi);

  const response = await fetch("/api/rppg/rppg/batch", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function listMethods(): Promise<{
  methods: string[];
  description: Record<string, string>;
}> {
  const response = await fetch("/api/rppg/methods");
  if (!response.ok) throw new Error("Failed to fetch methods");
  return response.json();
}

export async function healthCheck(): Promise<{ status: string; service: string }> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}