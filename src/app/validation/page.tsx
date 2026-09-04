'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import {
  createValidationSession,
  addValidationMeasurement,
  getValidationStatistics,
  getValidationSessions,
  getValidationMeasurements,
  type ValidationStats,
} from '@/lib/vital-storage';

interface ValidationSession {
  id: string;
  timestamp: string;
  subjectId: string | null;
  deviceInfo: string | null;
  cameraMode: 'front' | 'rear';
  referenceDevice: string | null;
  protocolNotes: string | null;
}

interface ValidationMeasurement {
  id: string;
  sessionId: string;
  timestamp: string;
  appBpm: number | null;
  appSpo2: number | null;
  appHb: number | null;
  appRespRate: number | null;
  appBpSys: number | null;
  appBpDia: number | null;
  appSugar: number | null;
  appSignalQuality: number | null;
  appRhythm: string | null;
  appSdnn: number | null;
  appRmssd: number | null;
  appPnn50: number | null;
  gtBpm: number | null;
  gtSpo2: number | null;
  gtHb: number | null;
  gtRespRate: number | null;
  gtBpSys: number | null;
  gtBpDia: number | null;
  gtSugar: number | null;
  gtRhythm: string | null;
  gtSdnn: number | null;
  gtRmssd: number | null;
  gtPnn50: number | null;
  diffBpm: number | null;
  diffSpo2: number | null;
  diffHb: number | null;
  diffRespRate: number | null;
  diffBpSys: number | null;
  diffBpDia: number | null;
  diffSugar: number | null;
  diffSdnn: number | null;
  diffRmssd: number | null;
  diffPnn50: number | null;
  isValid: boolean;
  exclusionReason: string | null;
}

type MetricKey = 'bpm' | 'spo2' | 'hb' | 'respRate';

const METRIC_LABELS: Record<MetricKey, { label: string; unit: string; appLabel: string; gtLabel: string }> = {
  bpm: { label: 'Heart Rate', unit: 'BPM', appLabel: 'appBpm', gtLabel: 'gtBpm' },
  spo2: { label: 'SpO2', unit: '%', appLabel: 'appSpo2', gtLabel: 'gtSpo2' },
  hb: { label: 'Hemoglobin', unit: 'g/dL', appLabel: 'appHb', gtLabel: 'gtHb' },
  respRate: { label: 'Respiratory Rate', unit: '/min', appLabel: 'appRespRate', gtLabel: 'gtRespRate' },
};

export default function ValidationPage() {
  const [sessions, setSessions] = useState<ValidationSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<ValidationSession | null>(null);
  const [measurements, setMeasurements] = useState<ValidationMeasurement[]>([]);
  const [stats, setStats] = useState<ValidationStats | null>(null);
  const [cameraModeFilter, setCameraModeFilter] = useState<'all' | 'front' | 'rear'>('all');
  const [activeMetric, setActiveMetric] = useState<MetricKey>('bpm');
  const [loading, setLoading] = useState(true);
  const [showNewSession, setShowNewSession] = useState(false);
  const [showNewMeasurement, setShowNewMeasurement] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importSuccess, setImportSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [newSession, setNewSession] = useState({
    subjectId: '',
    deviceInfo: '',
    cameraMode: 'rear' as 'front' | 'rear',
    referenceDevice: '',
    protocolNotes: '',
  });

  const [newMeasurement, setNewMeasurement] = useState({
    appBpm: '',
    appSpo2: '',
    appHb: '',
    appRespRate: '',
    gtBpm: '',
    gtSpo2: '',
    gtHb: '',
    gtRespRate: '',
    appSignalQuality: '',
    appRhythm: '',
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [sessionsData, statsData] = await Promise.all([
        getValidationSessions(100),
        getValidationStatistics(cameraModeFilter !== 'all' ? { cameraMode: cameraModeFilter } : undefined),
      ]);
      setSessions(sessionsData as ValidationSession[]);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load validation data:', error);
    } finally {
      setLoading(false);
    }
  }, [cameraModeFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (selectedSession) {
      getValidationMeasurements(selectedSession.id).then((data) => {
        setMeasurements(data as ValidationMeasurement[]);
      });
    }
  }, [selectedSession]);

  const drawBlandAltmanPlot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !stats) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const metricStats = stats[activeMetric];
    if (!metricStats || metricStats.count === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data available for this metric', canvas.width / 2, canvas.height / 2);
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    ctx.clearRect(0, 0, width, height);

    const appField = METRIC_LABELS[activeMetric].appLabel;
    const gtField = METRIC_LABELS[activeMetric].gtLabel;

    const points: { x: number; y: number }[] = [];
    let minAvg = Infinity;
    let maxAvg = -Infinity;
    let minDiff = Infinity;
    let maxDiff = -Infinity;

    measurements.forEach((m) => {
      const appVal = m[appField as keyof ValidationMeasurement] as number | null;
      const gtVal = m[gtField as keyof ValidationMeasurement] as number | null;
      if (appVal !== null && gtVal !== null && m.isValid) {
        const avg = (appVal + gtVal) / 2;
        const diff = appVal - gtVal;
        points.push({ x: avg, y: diff });
        minAvg = Math.min(minAvg, avg);
        maxAvg = Math.max(maxAvg, avg);
        minDiff = Math.min(minDiff, diff);
        maxDiff = Math.max(maxDiff, diff);
      }
    });

    if (points.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No valid measurement pairs', width / 2, height / 2);
      return;
    }

    const mean = metricStats.meanBias;
    const loaUpper = metricStats.loaUpper;
    const loaLower = metricStats.loaLower;

    const xRange = maxAvg - minAvg || 1;
    const yRange = Math.max(Math.abs(loaUpper), Math.abs(loaLower), 10);
    const yPadding = yRange * 0.2;

    const scaleX = (val: number) => padding.left + ((val - minAvg) / xRange) * plotWidth;
    const scaleY = (val: number) => padding.top + ((yRange + yPadding - val) / (2 * yRange + yPadding)) * plotHeight;

    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = -Math.ceil(yRange); i <= Math.ceil(yRange); i += Math.ceil(yRange / 4)) {
      const y = scaleY(i);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    ctx.strokeStyle = '#9ca3af';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(scaleX(minAvg), scaleY(mean));
    ctx.lineTo(scaleX(maxAvg), scaleY(mean));
    ctx.stroke();

    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(scaleX(minAvg), scaleY(loaUpper));
    ctx.lineTo(scaleX(maxAvg), scaleY(loaUpper));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(scaleX(minAvg), scaleY(loaLower));
    ctx.lineTo(scaleX(maxAvg), scaleY(loaLower));
    ctx.stroke();

    ctx.setLineDash([]);

    ctx.fillStyle = '#6b7280';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    for (let i = -Math.ceil(yRange); i <= Math.ceil(yRange); i += Math.ceil(yRange / 4)) {
      ctx.fillText(i.toFixed(1), padding.left - 10, scaleY(i) + 4);
    }

    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Mean: ${mean.toFixed(2)}`, width / 2, padding.top - 10);
    ctx.fillText(`+1.96 SD: ${loaUpper.toFixed(2)}`, width / 2, scaleY(loaUpper) - 5);
    ctx.fillText(`-1.96 SD: ${loaLower.toFixed(2)}`, width / 2, scaleY(loaLower) + 15);

    ctx.fillStyle = '#3b82f6';
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(scaleX(p.x), scaleY(p.y), 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`(App + Reference) / 2`, width - padding.right - 100, height - 15);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Difference (App - Reference)', 0, 0);
    ctx.restore();
  }, [stats, activeMetric, measurements]);

  useEffect(() => {
    if (stats && canvasRef.current) {
      drawBlandAltmanPlot();
    }
  }, [stats, activeMetric, drawBlandAltmanPlot]);

  const handleCreateSession = async () => {
    try {
      await createValidationSession(newSession);
      setShowNewSession(false);
      setNewSession({
        subjectId: '',
        deviceInfo: '',
        cameraMode: 'rear',
        referenceDevice: '',
        protocolNotes: '',
      });
      loadData();
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const handleAddMeasurement = async () => {
    if (!selectedSession) return;
    try {
      await addValidationMeasurement({
        sessionId: selectedSession.id,
        appBpm: newMeasurement.appBpm ? parseFloat(newMeasurement.appBpm) : undefined,
        appSpo2: newMeasurement.appSpo2 ? parseFloat(newMeasurement.appSpo2) : undefined,
        appHb: newMeasurement.appHb ? parseFloat(newMeasurement.appHb) : undefined,
        appRespRate: newMeasurement.appRespRate ? parseFloat(newMeasurement.appRespRate) : undefined,
        gtBpm: newMeasurement.gtBpm ? parseFloat(newMeasurement.gtBpm) : undefined,
        gtSpo2: newMeasurement.gtSpo2 ? parseFloat(newMeasurement.gtSpo2) : undefined,
        gtHb: newMeasurement.gtHb ? parseFloat(newMeasurement.gtHb) : undefined,
        gtRespRate: newMeasurement.gtRespRate ? parseFloat(newMeasurement.gtRespRate) : undefined,
        appSignalQuality: newMeasurement.appSignalQuality ? parseFloat(newMeasurement.appSignalQuality) : undefined,
        appRhythm: newMeasurement.appRhythm || undefined,
      });
      setShowNewMeasurement(false);
      setNewMeasurement({
        appBpm: '',
        appSpo2: '',
        appHb: '',
        appRespRate: '',
        gtBpm: '',
        gtSpo2: '',
        gtHb: '',
        gtRespRate: '',
        appSignalQuality: '',
        appRhythm: '',
      });
      loadData();
      if (selectedSession) {
        const data = await getValidationMeasurements(selectedSession.id);
        setMeasurements(data as ValidationMeasurement[]);
      }
    } catch (error) {
      console.error('Failed to add measurement:', error);
    }
  };

  const handleImportCsv = async (file: File) => {
    setImporting(true);
    setImportError(null);
    setImportSuccess(null);

    try {
      const text = await file.text();
      const lines = text.split(/\r?\n/).filter((line) => line.trim());
      if (lines.length < 2) {
        throw new Error('CSV file is empty or has no data rows');
      }

      const header = lines[0].split(',').map((h) => h.trim().toLowerCase());
      const expectedHeaders = ['timestamp', 'app_bpm', 'app_hb', 'app_spo2', 'gt_bpm', 'gt_spo2', 'gt_hb', 'gt_resp_rate'];
      const missing = expectedHeaders.filter((h) => !header.includes(h));
      if (missing.length > 0) {
        throw new Error(`Missing required CSV columns: ${missing.join(', ')}`);
      }

      const session = await createValidationSession({
        cameraMode: 'rear',
        protocolNotes: `Imported from CSV: ${file.name}`,
      });

      let imported = 0;
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map((v) => v.trim());
        if (values.length < header.length) continue;

        const row: Record<string, string> = {};
        header.forEach((key, idx) => {
          row[key] = values[idx] || '';
        });

        const appBpm = row.app_bpm ? parseFloat(row.app_bpm) : undefined;
        const appHb = row.app_hb ? parseFloat(row.app_hb) : undefined;
        const appSpo2 = row.app_spo2 ? parseFloat(row.app_spo2) : undefined;
        const gtBpm = row.gt_bpm ? parseFloat(row.gt_bpm) : undefined;
        const gtSpo2 = row.gt_spo2 ? parseFloat(row.gt_spo2) : undefined;
        const gtHb = row.gt_hb ? parseFloat(row.gt_hb) : undefined;
        const gtRespRate = row.gt_resp_rate ? parseFloat(row.gt_resp_rate) : undefined;

        if (appBpm == null && gtBpm == null) continue;

        await addValidationMeasurement({
          sessionId: session.id,
          appBpm,
          appHb,
          appSpo2,
          gtBpm,
          gtSpo2,
          gtHb,
          gtRespRate,
          isValid: true,
        });
        imported++;
      }

      setImportSuccess(`Imported ${imported} measurements into new session ${session.id}`);
      await loadData();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to import CSV';
      setImportError(message);
    } finally {
      setImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  const filteredSessions = sessions.filter((s) => {
    if (cameraModeFilter === 'all') return true;
    return s.cameraMode === cameraModeFilter;
  });

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Validation Dashboard</h1>
            <p className="text-neutral-400 mt-1">Bland-Altman analysis for clinical accuracy assessment</p>
          </div>
          <div className="flex gap-3">
            <Link
              href="/vital-monitoring"
              className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition"
            >
              Launch Monitor
            </Link>
            <Link
              href="/"
              className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition"
            >
              Home
            </Link>
          </div>
        </div>

        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setCameraModeFilter('all')}
            className={`px-4 py-2 text-sm rounded-lg transition ${
              cameraModeFilter === 'all'
                ? 'bg-red-600 text-white'
                : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
            }`}
          >
            All Sessions
          </button>
          <button
            onClick={() => setCameraModeFilter('front')}
            className={`px-4 py-2 text-sm rounded-lg transition ${
              cameraModeFilter === 'front'
                ? 'bg-red-600 text-white'
                : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
            }`}
          >
            Front (rPPG)
          </button>
          <button
            onClick={() => setCameraModeFilter('rear')}
            className={`px-4 py-2 text-sm rounded-lg transition ${
              cameraModeFilter === 'rear'
                ? 'bg-red-600 text-white'
                : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
            }`}
          >
            Rear (PPG)
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-neutral-900 rounded-2xl border border-neutral-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white">Sessions</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowNewSession(true)}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition"
                >
                  New Session
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white text-sm rounded-lg transition"
                >
                  {importing ? 'Importing...' : 'Import CSV'}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleImportCsv(file);
                  }}
                />
              </div>
            </div>
            {importError && (
              <div className="mb-4 p-2 bg-red-900/30 border border-red-700 rounded text-red-300 text-xs">
                {importError}
              </div>
            )}
            {importSuccess && (
              <div className="mb-4 p-2 bg-emerald-900/30 border border-emerald-700 rounded text-emerald-300 text-xs">
                {importSuccess}
              </div>
            )}

            {loading ? (
              <div className="text-center py-8 text-neutral-400">Loading...</div>
            ) : filteredSessions.length === 0 ? (
              <div className="text-center py-8 text-neutral-400">
                No validation sessions found. Create one to get started.
              </div>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {filteredSessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setSelectedSession(session)}
                    className={`w-full text-left p-4 rounded-lg transition ${
                      selectedSession?.id === session.id
                        ? 'bg-red-600/20 border border-red-600'
                        : 'bg-neutral-800 border border-neutral-700 hover:border-neutral-600'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-white">
                          {session.subjectId || 'Anonymous Subject'}
                        </p>
                        <p className="text-sm text-neutral-400">
                          {session.cameraMode === 'front' ? 'rPPG (Front)' : 'PPG (Rear)'} •{' '}
                          {session.referenceDevice || 'No reference device'}
                        </p>
                      </div>
                      <span className="text-xs text-neutral-500">{formatDate(session.timestamp)}</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Statistics</h2>
            {stats && stats.count > 0 ? (
              <div className="space-y-4">
                {(['bpm', 'spo2', 'hb', 'respRate'] as MetricKey[]).map((key) => {
                  const metric = stats[key];
                  if (!metric || metric.count === 0) return null;
                  return (
                    <div key={key} className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-neutral-300">
                          {METRIC_LABELS[key].label}
                        </span>
                        <span className="text-xs text-neutral-500">n={metric.count}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-neutral-500">Mean Bias:</span>
                          <span className="ml-1 text-white">{metric.meanBias.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-neutral-500">MAE:</span>
                          <span className="ml-1 text-white">{metric.mae.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-neutral-500">RMSE:</span>
                          <span className="ml-1 text-white">{metric.rmse.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-neutral-500">95% LOA:</span>
                          <span className="ml-1 text-white">
                            [{metric.loaLower.toFixed(1)}, {metric.loaUpper.toFixed(1)}]
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-neutral-400">
                No statistics available yet.
              </div>
            )}
          </div>
        </div>

        {selectedSession && (
          <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white">Bland-Altman Plot</h2>
              <div className="flex gap-2">
                {(['bpm', 'spo2', 'hb', 'respRate'] as MetricKey[]).map((key) => (
                  <button
                    key={key}
                    onClick={() => setActiveMetric(key)}
                    className={`px-3 py-1 text-xs rounded-lg transition ${
                      activeMetric === key
                        ? 'bg-red-600 text-white'
                        : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                    }`}
                  >
                    {METRIC_LABELS[key].label}
                  </button>
                ))}
              </div>
            </div>
            <canvas
              ref={canvasRef}
              width={800}
              height={400}
              className="w-full max-w-3xl mx-auto bg-neutral-800 rounded-lg"
            />
            <div className="mt-4 flex justify-center gap-6 text-xs text-neutral-400">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-neutral-600 border border-neutral-500" />
                <span>Mean Difference</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500/50" />
                <span>95% Limits of Agreement</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full" />
                <span>Measurement Pair</span>
              </div>
            </div>
          </div>
        )}

        {selectedSession && (
          <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white">
                Measurements for Session
                <span className="text-sm font-normal text-neutral-400 ml-2">
                  ({selectedSession.subjectId || 'Anonymous'})
                </span>
              </h2>
              <button
                onClick={() => setShowNewMeasurement(true)}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition"
              >
                Add Measurement
              </button>
            </div>

            {measurements.length === 0 ? (
              <div className="text-center py-8 text-neutral-400">
                No measurements in this session yet.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-700">
                      <th className="text-left py-3 px-2 text-neutral-400 font-medium">Time</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">App BPM</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Ref BPM</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">App SpO2</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Ref SpO2</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">App Hb</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Ref Hb</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Diff BPM</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Diff SpO2</th>
                      <th className="text-right py-3 px-2 text-neutral-400 font-medium">Valid</th>
                    </tr>
                  </thead>
                  <tbody>
                    {measurements.map((m) => (
                      <tr
                        key={m.id}
                        className={`border-b border-neutral-800 ${
                          !m.isValid ? 'opacity-50' : ''
                        }`}
                      >
                        <td className="py-2 px-2 text-neutral-300">{formatDate(m.timestamp)}</td>
                        <td className="py-2 px-2 text-right text-white">{m.appBpm?.toFixed(1) ?? '-'}</td>
                        <td className="py-2 px-2 text-right text-white">{m.gtBpm?.toFixed(1) ?? '-'}</td>
                        <td className="py-2 px-2 text-right text-white">{m.appSpo2?.toFixed(0) ?? '-'}</td>
                        <td className="py-2 px-2 text-right text-white">{m.gtSpo2?.toFixed(0) ?? '-'}</td>
                        <td className="py-2 px-2 text-right text-white">{m.appHb?.toFixed(1) ?? '-'}</td>
                        <td className="py-2 px-2 text-right text-white">{m.gtHb?.toFixed(1) ?? '-'}</td>
                        <td
                          className={`py-2 px-2 text-right ${
                            m.diffBpm !== null
                              ? m.diffBpm > 5 || m.diffBpm < -5
                                ? 'text-red-400'
                                : 'text-green-400'
                              : 'text-neutral-500'
                          }`}
                        >
                          {m.diffBpm?.toFixed(1) ?? '-'}
                        </td>
                        <td
                          className={`py-2 px-2 text-right ${
                            m.diffSpo2 !== null
                              ? m.diffSpo2 > 3 || m.diffSpo2 < -3
                                ? 'text-red-400'
                                : 'text-green-400'
                              : 'text-neutral-500'
                          }`}
                        >
                          {m.diffSpo2?.toFixed(1) ?? '-'}
                        </td>
                        <td className="py-2 px-2 text-right">
                          {m.isValid ? (
                            <span className="text-green-400">Yes</span>
                          ) : (
                            <span className="text-red-400" title={m.exclusionReason || 'Invalid'}>
                              No
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {showNewSession && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="bg-neutral-900 rounded-2xl border border-neutral-700 p-6 w-full max-w-md">
              <h3 className="text-xl font-bold text-white mb-4">New Validation Session</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Subject ID</label>
                  <input
                    type="text"
                    value={newSession.subjectId}
                    onChange={(e) => setNewSession({ ...newSession, subjectId: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="Optional"
                  />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Device Info</label>
                  <input
                    type="text"
                    value={newSession.deviceInfo}
                    onChange={(e) => setNewSession({ ...newSession, deviceInfo: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="e.g., Chrome on MacBook Pro"
                  />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Camera Mode</label>
                  <select
                    value={newSession.cameraMode}
                    onChange={(e) =>
                      setNewSession({ ...newSession, cameraMode: e.target.value as 'front' | 'rear' })
                    }
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                  >
                    <option value="rear">Rear (PPG - Fingertip)</option>
                    <option value="front">Front (rPPG - Face)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Reference Device</label>
                  <input
                    type="text"
                    value={newSession.referenceDevice}
                    onChange={(e) => setNewSession({ ...newSession, referenceDevice: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="e.g., FDA-cleared pulse oximeter"
                  />
                </div>
                <div>
                  <label className="block text-sm text-neutral-400 mb-1">Protocol Notes</label>
                  <textarea
                    value={newSession.protocolNotes}
                    onChange={(e) => setNewSession({ ...newSession, protocolNotes: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500 h-20"
                    placeholder="Optional notes about the validation protocol"
                  />
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowNewSession(false)}
                  className="flex-1 px-4 py-2 bg-neutral-700 hover:bg-neutral-600 text-white rounded-lg transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateSession}
                  className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition"
                >
                  Create Session
                </button>
              </div>
            </div>
          </div>
        )}

        {showNewMeasurement && selectedSession && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="bg-neutral-900 rounded-2xl border border-neutral-700 p-6 w-full max-w-lg">
              <h3 className="text-xl font-bold text-white mb-4">Add Measurement</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2 border-b border-neutral-700 pb-2 mb-1">
                  <span className="text-sm text-red-400 font-medium">App Measurements</span>
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Heart Rate (BPM)</label>
                  <input
                    type="number"
                    value={newMeasurement.appBpm}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, appBpm: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="72"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">SpO2 (%)</label>
                  <input
                    type="number"
                    value={newMeasurement.appSpo2}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, appSpo2: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="98"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Hemoglobin (g/dL)</label>
                  <input
                    type="number"
                    value={newMeasurement.appHb}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, appHb: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="14.5"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Resp Rate (/min)</label>
                  <input
                    type="number"
                    value={newMeasurement.appRespRate}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, appRespRate: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="16"
                  />
                </div>

                <div className="col-span-2 border-b border-neutral-700 pb-2 mt-2 mb-1">
                  <span className="text-sm text-green-400 font-medium">Reference (Ground Truth)</span>
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Heart Rate (BPM)</label>
                  <input
                    type="number"
                    value={newMeasurement.gtBpm}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, gtBpm: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-green-500"
                    placeholder="71"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">SpO2 (%)</label>
                  <input
                    type="number"
                    value={newMeasurement.gtSpo2}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, gtSpo2: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-green-500"
                    placeholder="99"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Hemoglobin (g/dL)</label>
                  <input
                    type="number"
                    value={newMeasurement.gtHb}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, gtHb: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-green-500"
                    placeholder="14.2"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Resp Rate (/min)</label>
                  <input
                    type="number"
                    value={newMeasurement.gtRespRate}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, gtRespRate: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-green-500"
                    placeholder="15"
                  />
                </div>

                <div className="col-span-2 border-b border-neutral-700 pb-2 mt-2 mb-1">
                  <span className="text-sm text-neutral-400 font-medium">Additional</span>
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Signal Quality</label>
                  <input
                    type="number"
                    value={newMeasurement.appSignalQuality}
                    onChange={(e) =>
                      setNewMeasurement({ ...newMeasurement, appSignalQuality: e.target.value })
                    }
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                    placeholder="0.85"
                    min="0"
                    max="1"
                    step="0.01"
                  />
                </div>
                <div>
                  <label className="block text-xs text-neutral-400 mb-1">Rhythm</label>
                  <select
                    value={newMeasurement.appRhythm}
                    onChange={(e) => setNewMeasurement({ ...newMeasurement, appRhythm: e.target.value })}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-lg text-white focus:outline-none focus:border-red-500"
                  >
                    <option value="">Unknown</option>
                    <option value="regular">Regular</option>
                    <option value="irregular">Irregular</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowNewMeasurement(false)}
                  className="flex-1 px-4 py-2 bg-neutral-700 hover:bg-neutral-600 text-white rounded-lg transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddMeasurement}
                  className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition"
                >
                  Add Measurement
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="mt-8 bg-neutral-900 rounded-2xl border border-neutral-800 p-6">
          <h2 className="text-xl font-bold text-white mb-4">About Bland-Altman Analysis</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-neutral-400">
            <div>
              <p className="mb-2">
                The Bland-Altman plot is the gold standard for comparing two measurement methods. It plots the
                difference between measurements against their mean.
              </p>
              <ul className="list-disc list-inside space-y-1 text-neutral-500">
                <li>
                  <span className="text-neutral-400">Mean Bias:</span> Average systematic error between methods
                </li>
                <li>
                  <span className="text-neutral-400">95% LOA:</span> Limits where 95% of differences fall
                </li>
                <li>
                  <span className="text-neutral-400">MAE:</span> Mean Absolute Error
                </li>
                <li>
                  <span className="text-neutral-400">RMSE:</span> Root Mean Square Error
                </li>
              </ul>
            </div>
            <div>
              <p className="mb-2">
                <span className="text-red-400 font-medium">Clinical Thresholds:</span>
              </p>
              <ul className="list-disc list-inside space-y-1 text-neutral-500">
                <li>Heart Rate: ±5 BPM (clinical), ±10 BPM (acceptable)</li>
                <li>SpO2: ±2% (clinical), ±3% (acceptable)</li>
                <li>Hemoglobin: ±1 g/dL (clinical), ±2 g/dL (acceptable)</li>
                <li>Respiratory Rate: ±2 /min (clinical)</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 p-3 bg-neutral-800 rounded-lg border border-neutral-700">
            <p className="text-xs text-neutral-500">
              <span className="text-yellow-400 font-medium">Note:</span> SpO2 and Hemoglobin measurements from
              smartphone cameras are estimates only. SpO2 lacks the near-infrared (940nm) wavelength required for
              clinical-grade pulse oximetry. Blood pressure and glucose cannot be measured from PPG signals.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
