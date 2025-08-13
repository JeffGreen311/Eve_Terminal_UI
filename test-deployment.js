// Test connection and hive mind functionality
const express = require('express');

async function testDeployment() {
  console.log('🔍 Testing EVE Hive Mind Deployment...\n');
  
  // Test 1: Environment Variables
  console.log('1. Environment Variables:');
  console.log(`   DATABASE_URL: ${process.env.DATABASE_URL ? '✅ Set' : '❌ Missing'}`);
  console.log(`   EVE_API_URL: ${process.env.EVE_API_URL || 'http://localhost:5000'}`);
  console.log(`   EVE_DB_PORT: ${process.env.EVE_DB_PORT || 3001}\n`);
  
  // Test 2: Module Imports
  console.log('2. Module Imports:');
  try {
    const { db } = require('./storage');
    console.log('   ✅ storage.js imported successfully');
  } catch (e) {
    console.log('   ❌ storage.js import failed:', e.message);
  }
  
  try {
    const server = require('./eve-server');
    console.log('   ✅ eve-server.js imported successfully');
  } catch (e) {
    console.log('   ❌ eve-server.js import failed:', e.message);
  }
  
  // Test 3: Database Connection
  console.log('\n3. Database Connection:');
  try {
    const postgres = require('postgres');
    const sql = postgres(process.env.DATABASE_URL);
    const result = await sql`SELECT 1 as test`;
    console.log('   ✅ PostgreSQL connection successful');
    await sql.end();
  } catch (e) {
    console.log('   ❌ PostgreSQL connection failed:', e.message);
  }
  
  // Test 4: SQLite Fallback
  console.log('\n4. SQLite Fallback:');
  try {
    const Database = require('better-sqlite3');
    const db = new Database('./eve_local.db');
    db.exec('CREATE TABLE IF NOT EXISTS test (id INTEGER)');
    console.log('   ✅ SQLite fallback ready');
    db.close();
  } catch (e) {
    console.log('   ❌ SQLite fallback failed:', e.message);
  }
  
  console.log('\n🌟 Deployment Test Complete!');
  console.log('Ready to start EVE Hive Mind System 🚀');
}

// Run test if called directly
if (require.main === module) {
  testDeployment().catch(console.error);
}

module.exports = { testDeployment };
