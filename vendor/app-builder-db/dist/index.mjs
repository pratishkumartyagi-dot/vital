// src/client.ts
import { drizzle } from "drizzle-orm/sqlite-proxy";
function createExecuteQuery(config = {}) {
  const url = config.url ?? process.env.DB_URL;
  const token = config.token ?? process.env.DB_TOKEN;
  if (!url || !token) {
    throw new Error(
      "Missing database configuration. Provide url and token in config or set DB_URL and DB_TOKEN environment variables."
    );
  }
  return async function executeQuery(sql, params, method) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ sql, params, method })
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || "Query failed");
    }
    return response.json();
  };
}
function createDatabase(schema, config = {}) {
  const executeQuery = createExecuteQuery(config);
  return drizzle(
    async (sql, params, method) => {
      const result = await executeQuery(sql, params, method);
      return { rows: result.rows };
    },
    { schema }
  );
}
function createDatabaseWithoutSchema(config = {}) {
  const executeQuery = createExecuteQuery(config);
  return drizzle(async (sql, params, method) => {
    const result = await executeQuery(sql, params, method);
    return { rows: result.rows };
  });
}

// src/migrate.ts
import { migrate as drizzleMigrate } from "drizzle-orm/sqlite-proxy/migrator";
async function runMigrations(db, config = {}, options = {}) {
  const executeQuery = createExecuteQuery(config);
  const migrationsFolder = options.migrationsFolder ?? "./src/db/migrations";
  await drizzleMigrate(
    db,
    async (queries) => {
      for (const query of queries) {
        await executeQuery(query, [], "run");
      }
    },
    {
      migrationsFolder
    }
  );
}
function createMigrationRunner(db, config = {}) {
  return async (migrationsFolder = "./src/db/migrations") => {
    await runMigrations(db, config, { migrationsFolder });
  };
}
export {
  createDatabase,
  createDatabaseWithoutSchema,
  createExecuteQuery,
  createMigrationRunner,
  runMigrations
};
