"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  createDatabase: () => createDatabase,
  createDatabaseWithoutSchema: () => createDatabaseWithoutSchema,
  createExecuteQuery: () => createExecuteQuery,
  createMigrationRunner: () => createMigrationRunner,
  runMigrations: () => runMigrations
});
module.exports = __toCommonJS(index_exports);

// src/client.ts
var import_sqlite_proxy = require("drizzle-orm/sqlite-proxy");
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
  return (0, import_sqlite_proxy.drizzle)(
    async (sql, params, method) => {
      const result = await executeQuery(sql, params, method);
      return { rows: result.rows };
    },
    { schema }
  );
}
function createDatabaseWithoutSchema(config = {}) {
  const executeQuery = createExecuteQuery(config);
  return (0, import_sqlite_proxy.drizzle)(async (sql, params, method) => {
    const result = await executeQuery(sql, params, method);
    return { rows: result.rows };
  });
}

// src/migrate.ts
var import_migrator = require("drizzle-orm/sqlite-proxy/migrator");
async function runMigrations(db, config = {}, options = {}) {
  const executeQuery = createExecuteQuery(config);
  const migrationsFolder = options.migrationsFolder ?? "./src/db/migrations";
  await (0, import_migrator.migrate)(
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  createDatabase,
  createDatabaseWithoutSchema,
  createExecuteQuery,
  createMigrationRunner,
  runMigrations
});
