import { runMigrations } from "@kilocode/app-builder-db";
import { getDb } from "./index";

await runMigrations(getDb(), {}, { migrationsFolder: "./src/db/migrations" });