import { createDatabase } from "@kilocode/app-builder-db";
import * as schema from "./schema";

export function getDb() {
  return createDatabase(schema);
}