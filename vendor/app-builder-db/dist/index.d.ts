import { SqliteRemoteDatabase } from 'drizzle-orm/sqlite-proxy';

/**
 * Query method types for SQLite operations
 */
type QueryMethod = "get" | "all" | "run" | "values";
/**
 * Result of a query execution
 */
interface QueryResult {
    rows: unknown[] | unknown[][];
}
/**
 * Configuration for the database client
 */
interface DatabaseConfig {
    /**
     * Full URL for the database API
     * Defaults to process.env.DB_URL
     */
    url?: string;
    /**
     * Authentication token for the database API
     * Defaults to process.env.DB_TOKEN
     */
    token?: string;
}
/**
 * Options for running migrations
 */
interface MigrationOptions {
    /**
     * Path to the migrations folder
     * @default "./src/db/migrations"
     */
    migrationsFolder?: string;
}

/**
 * Creates an execute query function for the given database configuration
 */
declare function createExecuteQuery(config?: DatabaseConfig): (sql: string, params: unknown[], method: QueryMethod) => Promise<QueryResult>;
/**
 * Creates a Drizzle database client
 *
 * @param schema - Your Drizzle schema object
 * @param config - Optional database configuration (defaults to environment variables)
 * @returns Drizzle database instance
 *
 * @example
 * ```typescript
 * import { createDatabase } from '@kilocode/app-builder-db';
 * import * as schema from './schema';
 *
 * export const db = createDatabase(schema);
 * ```
 */
declare function createDatabase<TSchema extends Record<string, unknown>>(schema: TSchema, config?: DatabaseConfig): SqliteRemoteDatabase<TSchema>;
/**
 * Creates a Drizzle database client without schema (for basic queries)
 *
 * @param config - Optional database configuration (defaults to environment variables)
 * @returns Drizzle database instance
 *
 * @example
 * ```typescript
 * import { createDatabaseWithoutSchema } from '@kilocode/app-builder-db';
 *
 * const db = createDatabaseWithoutSchema();
 * ```
 */
declare function createDatabaseWithoutSchema(config?: DatabaseConfig): SqliteRemoteDatabase;

/**
 * Runs database migrations using the Drizzle migrator
 *
 * @param db - Drizzle database instance
 * @param config - Database configuration for executing migration queries
 * @param options - Migration options
 *
 * @example
 * ```typescript
 * import { createDatabase, runMigrations } from '@kilocode/app-builder-db';
 * import * as schema from './schema';
 *
 * const db = createDatabase(schema);
 *
 * async function main() {
 *   await runMigrations(db, {}, {
 *     migrationsFolder: './src/db/migrations'
 *   });
 *   console.log('Migrations completed successfully');
 * }
 *
 * main().catch(console.error);
 * ```
 */
declare function runMigrations<TSchema extends Record<string, unknown>>(db: SqliteRemoteDatabase<TSchema>, config?: DatabaseConfig, options?: MigrationOptions): Promise<void>;
/**
 * Creates a migration runner function for use in migration scripts
 *
 * @param db - Drizzle database instance
 * @param config - Database configuration
 * @returns A function that runs migrations when called
 *
 * @example
 * ```typescript
 * // src/db/migrate.ts
 * import { createDatabase, createMigrationRunner } from '@kilocode/app-builder-db';
 * import * as schema from './schema';
 *
 * const db = createDatabase(schema);
 * const migrate = createMigrationRunner(db);
 *
 * migrate('./src/db/migrations')
 *   .then(() => console.log('Migrations completed'))
 *   .catch(console.error);
 * ```
 */
declare function createMigrationRunner<TSchema extends Record<string, unknown>>(db: SqliteRemoteDatabase<TSchema>, config?: DatabaseConfig): (migrationsFolder?: string) => Promise<void>;

export { type DatabaseConfig, type MigrationOptions, type QueryMethod, type QueryResult, createDatabase, createDatabaseWithoutSchema, createExecuteQuery, createMigrationRunner, runMigrations };
