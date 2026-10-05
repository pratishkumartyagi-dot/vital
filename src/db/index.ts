import { drizzle, LibSQLDatabase } from "drizzle-orm/libsql";
import { createClient } from "@libsql/client";
import * as schema from "./schema";

let dbInstance: LibSQLDatabase<typeof schema> | null = null;

export function getDb(): LibSQLDatabase<typeof schema> {
  if (dbInstance) {
    return dbInstance;
  }

  const client = createClient({
    url: "file:local.db",
  });
  dbInstance = drizzle(client, { schema });
  return dbInstance;
}