ALTER TABLE `validation_measurements` ADD `app_creatinine` real;--> statement-breakpoint
ALTER TABLE `validation_measurements` ADD `gt_creatinine` real;--> statement-breakpoint
ALTER TABLE `validation_measurements` ADD `diff_creatinine` real;--> statement-breakpoint
ALTER TABLE `vital_readings` ADD `creatinine` real;--> statement-breakpoint
ALTER TABLE `vital_readings` ADD `egfr` real;--> statement-breakpoint
ALTER TABLE `vital_readings` ADD `ground_truth_creatinine` real;