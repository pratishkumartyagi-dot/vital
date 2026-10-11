import { sqliteTable, text, integer, real, blob } from "drizzle-orm/sqlite-core";

export const vitalReadings = sqliteTable("vital_readings", {
  id: text("id").primaryKey(),
  timestamp: integer("timestamp", { mode: "timestamp" }).notNull(),

  // Patient info
  patientName: text("patient_name").notNull(),
  patientAge: text("patient_age").notNull(),
  patientSex: text("patient_sex", { enum: ["male", "female"] }).notNull(),
  patientLmp: text("patient_lmp"),
  patientPregnancy: text("patient_pregnancy", { enum: ["pregnant", "non-pregnant"] }),
  patientMenopause: text("patient_menopause", { enum: ["yes", "no"] }),

  // Vitals
  bpm: real("bpm"),
  spo2: real("spo2"),
  hb: real("hb"),
  bpSys: real("bp_sys"),
  bpDia: real("bp_dia"),
  sugar: real("sugar"),
  signalQuality: real("signal_quality"),
  respRate: real("resp_rate"),
  sdnn: real("sdnn"),
  rmssd: real("rmssd"),
  pnn50: real("pnn50"),
  creatinine: real("creatinine"),
  egfr: real("egfr"),
  rhythm: text("rhythm", { enum: ["regular", "irregular", "unknown"] }).notNull(),

  // Camera mode and metadata
  cameraMode: text("camera_mode", { enum: ["front", "rear"] }).notNull(),
  demoMode: integer("demo_mode", { mode: "boolean" }).notNull().default(false),

  // Video data
  videoData: blob("video_data", { mode: "buffer" }),
  videoMimeType: text("video_mime_type"),

  // Ground truth validation
  groundTruthBpm: real("ground_truth_bpm"),
  groundTruthSpo2: real("ground_truth_spo2"),
  groundTruthHb: real("ground_truth_hb"),
  groundTruthRespRate: real("ground_truth_resp_rate"),
  groundTruthBpSys: real("ground_truth_bp_sys"),
  groundTruthBpDia: real("ground_truth_bp_dia"),
  groundTruthSugar: real("ground_truth_sugar"),
  groundTruthCreatinine: real("ground_truth_creatinine"),
  validationNotes: text("validation_notes"),

  // Session info
  sessionDurationSec: integer("session_duration_sec").notNull().default(130),
  warmupSec: integer("warmup_sec").notNull().default(10),
  measurementSec: integer("measurement_sec").notNull().default(120),
});

export const validationSessions = sqliteTable("validation_sessions", {
  id: text("id").primaryKey(),
  timestamp: integer("timestamp", { mode: "timestamp" }).notNull(),
  subjectId: text("subject_id"),
  deviceInfo: text("device_info"),
  cameraMode: text("camera_mode", { enum: ["front", "rear"] }).notNull(),
  referenceDevice: text("reference_device"),
  protocolNotes: text("protocol_notes"),
});

export const validationMeasurements = sqliteTable("validation_measurements", {
  id: text("id").primaryKey(),
  sessionId: text("session_id").notNull().references(() => validationSessions.id),
  timestamp: integer("timestamp", { mode: "timestamp" }).notNull(),

  // App measurements
  appBpm: real("app_bpm"),
  appSpo2: real("app_spo2"),
  appHb: real("app_hb"),
  appRespRate: real("app_resp_rate"),
  appBpSys: real("app_bp_sys"),
  appBpDia: real("app_bp_dia"),
  appSugar: real("app_sugar"),
  appSignalQuality: real("app_signal_quality"),
  appRhythm: text("app_rhythm"),
  appSdnn: real("app_sdnn"),
  appRmssd: real("app_rmssd"),
  appPnn50: real("app_pnn50"),
  appCreatinine: real("app_creatinine"),

  // Ground truth
  gtBpm: real("gt_bpm"),
  gtSpo2: real("gt_spo2"),
  gtHb: real("gt_hb"),
  gtRespRate: real("gt_resp_rate"),
  gtBpSys: real("gt_bp_sys"),
  gtBpDia: real("gt_bp_dia"),
  gtSugar: real("gt_sugar"),
  gtRhythm: text("gt_rhythm"),
  gtSdnn: real("gt_sdnn"),
  gtRmssd: real("gt_rmssd"),
  gtPnn50: real("gt_pnn50"),
  gtCreatinine: real("gt_creatinine"),

  // Differences
  diffBpm: real("diff_bpm"),
  diffSpo2: real("diff_spo2"),
  diffHb: real("diff_hb"),
  diffRespRate: real("diff_resp_rate"),
  diffBpSys: real("diff_bp_sys"),
  diffBpDia: real("diff_bp_dia"),
  diffSugar: real("diff_sugar"),
  diffSdnn: real("diff_sdnn"),
  diffRmssd: real("diff_rmssd"),
  diffPnn50: real("diff_pnn50"),
  diffCreatinine: real("diff_creatinine"),

  // Quality flags
  isValid: integer("is_valid", { mode: "boolean" }).notNull().default(true),
  exclusionReason: text("exclusion_reason"),
});

export const trainingJobs = sqliteTable("training_jobs", {
  id: text("id").primaryKey(),
  timestamp: integer("timestamp", { mode: "timestamp" }).notNull(),
  model: text("model").notNull(),
  dataset: text("dataset").notNull(),
  validFold: integer("valid_fold").notNull(),
  testFold: integer("test_fold").notNull(),
  batchSize: integer("batch_size").notNull(),
  device: text("device").notNull().default("cpu"),
  resultsFolder: text("results_folder"),
  testDatasets: text("test_datasets"),
  numFolds: integer("num_folds"),
  numWorkers: integer("num_workers"),

  status: text("status", { enum: ["pending", "running", "completed", "failed", "cancelled"] }).notNull().default("pending"),
  progress: real("progress").notNull().default(0),
  currentEpoch: integer("current_epoch").notNull().default(0),
  totalEpochs: integer("total_epochs").notNull().default(50),
  startTime: integer("start_time", { mode: "timestamp" }),
  endTime: integer("end_time", { mode: "timestamp" }),
  errorMessage: text("error_message"),

  trainLoss: real("train_loss"),
  validLoss: real("valid_loss"),
  ppgMae: real("ppg_mae"),
  hrMae: real("hr_mae"),
  results: text("results"),
});
