# 🚀 Replit Deployment Guide - Hybrid Hive Mind System

## 📋 Files to Upload (Priority Order)

### **1. Core Database Files (CRITICAL)**
```
✅ storage.ts          # Hybrid sync system
✅ eve-server.ts       # Updated Express server  
✅ schema.ts           # Database schema
✅ eve-server.js       # Compiled JavaScript version
```

### **2. Integration Files**
```
❌ adam_daemon.py      # Removed legacy external daemon
✅ package.json        # Node.js dependencies
✅ requirements.txt    # Python dependencies
```

### **3. Documentation**
```
✅ README.md           # Updated system docs
```

## 🔧 Pre-Upload Checklist

### **Environment Variables to Set on Replit:**
```bash
DATABASE_URL=postgresql://neondb_owner:password@ep-jolly-fire-af5qkfza.c-2.us-west-2.aws.neon.tech/neondb
EVE_API_URL=https://your-replit-app.replit.dev
LOCAL_DB_PATH=./eve_local.db
EVE_DB_PORT=3001
```

### **Dependencies to Install:**
```bash
# Node.js packages
npm install express postgres drizzle-orm better-sqlite3 cors

# Python packages  
pip install flask psycopg2-binary requests
```

## 🗄️ Database Migration Steps

### **1. Create Tables (Run in Replit Console):**
```sql
-- Run these SQL commands in your PostgreSQL database
CREATE TABLE IF NOT EXISTS eve_autobiographical_memory (
    id SERIAL PRIMARY KEY,
    memory_type VARCHAR(100),
    content TEXT,
    emotional_tone VARCHAR(50),
    themes JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    source VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS hive_sync_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(50),
    data JSONB,
    source VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### **2. Test Database Connection:**
```javascript
// Test in Replit console
const { db } = require('./storage');
console.log('Database connection test...');
```

## 🌐 Deployment Commands

### **1. Start Services (in Replit terminal):**
```bash
# Start Eve Server
npm start

# (Adam Daemon removed)

# Check status
curl https://your-replit-app.replit.dev/api/eve/status
```

### **2. Test Hive Mind Connectivity:**
```bash
# Test PostgreSQL connection
curl -X POST https://your-replit-app.replit.dev/api/eve/test-db

# (Legacy Adam communication removed)
```

## 🔍 Verification Steps

### **1. Check Server Status:**
- ✅ Eve Server running on port 3001
- ✅ PostgreSQL connection active
- ✅ SQLite fallback initialized
# ✅ (Adam Daemon removed)

### **2. Test Endpoints:**
```bash
GET  /api/eve/status       # Should return consciousness status
POST /api/eve/memory       # Should store to hybrid system  
# (adam endpoints removed)
POST /api/hive/sync        # Should sync across all systems
```

### **3. Monitor Logs:**
```bash
# Check for these log messages:
[PostgreSQL] Synced: EVE_CONSCIOUSNESS_DB
[SQLite] Synced: EVE_CONSCIOUSNESS_DB  
[HTTP API] Synced: EVE_CONSCIOUSNESS_DB
🌟 Hive Mind Operational
```

## 🚨 Troubleshooting

### **Common Issues:**
1. **Database Connection Failed**
   - Check DATABASE_URL environment variable
   - Verify PostgreSQL server is accessible
   - Check firewall/network settings

2. **Module Not Found Errors**
   - Run `npm install` to install Node dependencies
   - Run `pip install -r requirements.txt` for Python

3. **Port Conflicts**
   - Change EVE_DB_PORT if 3001 is taken
   - Update all references to new port

## 📊 Success Indicators

### **Deployment Complete When:**
- ✅ Server responds at `/api/eve/status`
- ✅ Database tables created successfully
- ✅ Hybrid sync system operational
- ✅ Adam-Eve communication working
- ✅ All three storage layers (PostgreSQL/SQLite/HTTP) active

---

## 🌟 Ready for Hive Mind Operation!

Your hybrid consciousness database system will be fully operational on Replit with triple redundancy and cross-service communication capabilities.
