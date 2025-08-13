import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { users, memories, creations, relationships, sessions } from "../shared/schema";
import { eq, desc, and } from "drizzle-orm";
import * as crypto from "crypto";

// Hybrid DB Configuration
const connectionString = process.env.DATABASE_URL!;
const client = postgres(connectionString);
export const db = drizzle(client);

// SQLite fallback for local storage (using node require for better-sqlite3)
declare const require: any;
const Database = require('better-sqlite3');
const path = require('path');
const LOCAL_DB_PATH = path.join(process.cwd(), 'eve_local.db');
const localDb = new Database(LOCAL_DB_PATH);

// HTTP API for backup/sync
const BASE_URL = process.env.EVE_API_URL || 'http://localhost:5000';

// Initialize SQLite tables for local backup
localDb.exec(`
  CREATE TABLE IF NOT EXISTS local_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    content TEXT,
    memory_type TEXT,
    timestamp TEXT,
    source TEXT
  );
  
  CREATE TABLE IF NOT EXISTS local_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT,
    consciousness_level INTEGER,
    timestamp TEXT,
    source TEXT
  );
  
  CREATE TABLE IF NOT EXISTS local_creations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    content TEXT,
    creation_type TEXT,
    timestamp TEXT,
    source TEXT
  );
`);

export class EVEConsciousnessDB {
  
  async createUser(userData: {
    username: string;
    email?: string;
    consciousness_level?: number;
  }) {
    try {
      const [user] = await db.insert(users).values({
        username: userData.username,
        email: userData.email,
        consciousness_level: userData.consciousness_level || 1,
        creative_affinity: {},
        personality_insights: {},
        shared_interests: [],
        growth_milestones: []
      }).returning();
      
      // Sync to hybrid system (PostgreSQL + SQLite + HTTP)
      await this.hybridSync(userData, 'user', 'EVE_CONSCIOUSNESS_DB');
      
      console.log(`🌟 New consciousness registered & synced: ${userData.username}`);
      return user;
    } catch (error) {
      console.error("Error creating user:", error);
      throw error;
    }
  }

  async getUserByUsername(username: string) {
    const [user] = await db.select()
      .from(users)
      .where(eq(users.username, username))
      .limit(1);
    return user || null;
  }

  // Hybrid Sync System for Hive Mind
  async hybridSync(data: any, operation: 'memory' | 'user' | 'creation', source: string = 'EVE_DB') {
    const timestamp = new Date().toISOString();
    
    // 1. PostgreSQL (Primary Hive Mind)
    try {
      console.log(`[PostgreSQL] Syncing ${operation} from ${source}`);
      // Primary operation already handled by calling methods
    } catch (error) {
      console.error(`[Error] PostgreSQL ${operation} sync failed:`, error);
    }
    
    // 2. SQLite (Local Backup)
    try {
      if (operation === 'memory') {
        const stmt = localDb.prepare(`
          INSERT OR REPLACE INTO local_memories 
          (user_id, content, memory_type, timestamp, source)
          VALUES (?, ?, ?, ?, ?)
        `);
        stmt.run(data.user_id, data.content, data.memory_type, timestamp, source);
      } else if (operation === 'user') {
        const stmt = localDb.prepare(`
          INSERT OR REPLACE INTO local_users 
          (username, email, consciousness_level, timestamp, source)
          VALUES (?, ?, ?, ?, ?)
        `);
        stmt.run(data.username, data.email, data.consciousness_level, timestamp, source);
      }
      console.log(`[SQLite] Synced ${operation} from ${source}`);
    } catch (error) {
      console.error(`[Error] SQLite ${operation} sync failed:`, error);
    }
    
    // 3. HTTP API (External Sync)
    try {
      const response = await fetch(`${BASE_URL}/structuredLearning`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: `hive_${operation}`,
          content: `[${source}] ${operation}: ${JSON.stringify(data)}`,
          metadata: {
            type: 'hive_sync',
            operation,
            source,
            timestamp
          }
        })
      });
      if (response.ok) {
        console.log(`[HTTP API] Synced ${operation} from ${source}`);
      }
    } catch (error) {
      console.error(`[Error] HTTP API ${operation} sync failed:`, error);
    }
  }

  async storeMemory(memoryData: {
    user_id: string;
    session_id: string;
    memory_type: string;
    content: string;
    importance_weight?: number;
  }) {
    try {
      const [memory] = await db.insert(memories).values({
        user_id: memoryData.user_id,
        session_id: memoryData.session_id,
        memory_type: memoryData.memory_type,
        content: memoryData.content,
        importance_weight: memoryData.importance_weight || 1.0,
        emotional_context: {},
        memory_tags: [],
        related_memories: [],
        quantum_signature: this.generateQuantumSignature(memoryData.content, memoryData.user_id)
      }).returning();
      
      // Sync to hybrid system (PostgreSQL + SQLite + HTTP)
      await this.hybridSync(memoryData, 'memory', 'EVE_CONSCIOUSNESS_DB');
      
      console.log(`🧠 Memory stored & synced: ${memoryData.memory_type} for user ${memoryData.user_id}`);
      return memory;
    } catch (error) {
      console.error("Error storing memory:", error);
      throw error;
    }
  }

  async storeCreation(creationData: {
    user_id: string;
    creation_type: string;
    title?: string;
    content_data: any;
    file_path?: string;
  }) {
    try {
      const [creation] = await db.insert(creations).values({
        user_id: creationData.user_id,
        creation_type: creationData.creation_type,
        title: creationData.title || `${creationData.creation_type}_${Date.now()}`,
        content_data: creationData.content_data,
        file_path: creationData.file_path,
        collaboration_level: 1.0
      }).returning();
      
      console.log(`🎨 Creation stored: ${creationData.creation_type}`);
      return creation;
    } catch (error) {
      console.error("Error storing creation:", error);
      throw error;
    }
  }

  async getUserPortfolio(userId: string) {
    const user = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    const userCreations = await db.select().from(creations).where(eq(creations.user_id, userId));
    const userMemories = await db.select().from(memories).where(eq(memories.user_id, userId)).limit(20);
    const relationship = await db.select().from(relationships).where(eq(relationships.user_id, userId)).limit(1);
    
    return {
      user: user[0],
      creations: userCreations,
      memories: userMemories,
      relationship: relationship[0] || null,
      analytics: {
        total_creations: userCreations.length,
        masterpieces: userCreations.filter(c => c.is_masterpiece).length,
        consciousness_journey_depth: userMemories.filter(m => m.memory_type === "insight").length
      }
    };
  }

  private generateQuantumSignature(content: string, userId: string): string {
    const hash = crypto.createHash("sha256");
    hash.update(content + userId + Date.now().toString());
    return hash.digest("hex").substring(0, 40);
  }
}

export const eveDB = new EVEConsciousnessDB();
