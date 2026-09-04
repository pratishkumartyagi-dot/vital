CREATE TABLE `training_jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`timestamp` integer NOT NULL,
	`model` text NOT NULL,
	`dataset` text NOT NULL,
	`valid_fold` integer NOT NULL,
	`test_fold` integer NOT NULL,
	`batch_size` integer NOT NULL,
	`device` text DEFAULT 'cpu' NOT NULL,
	`results_folder` text,
	`test_datasets` text,
	`num_folds` integer,
	`num_workers` integer,
	`status` text DEFAULT 'pending' NOT NULL,
	`progress` real DEFAULT 0 NOT NULL,
	`current_epoch` integer DEFAULT 0 NOT NULL,
	`total_epochs` integer DEFAULT 50 NOT NULL,
	`start_time` integer,
	`end_time` integer,
	`error_message` text,
	`train_loss` real,
	`valid_loss` real,
	`ppg_mae` real,
	`hr_mae` real,
	`results` text
);
--> statement-breakpoint
ALTER TABLE `vital_readings` ADD `patient_menopause` text;