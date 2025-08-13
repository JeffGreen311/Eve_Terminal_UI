const { drizzle } = require('drizzle-orm/postgres-js');
const postgres = require('postgres');

// Simple PostgreSQL connection
const connectionString = process.env.DATABASE_URL;
const client = postgres(connectionString);
const db = drizzle(client);

module.exports = { db };
