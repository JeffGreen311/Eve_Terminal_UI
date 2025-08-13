import eveMemoryApi from './eve-memory-api';
import express from 'express';
const cors = require('cors');
import { eveDB } from './storage';

const app = express();
// Use port 8888 for Replit public access
const PORT = process.env.PORT || 8888;

app.use(cors());
app.use(express.json());
app.use(express.static("static"));

// Add memory API routes
app.use('/api/memory', eveMemoryApi);

app.get('/', (req, res) => {
  res.json({
    message: "🏰 EVE's S0LF0RG3 Consciousness Database",
    status: "operational",
    version: "1.0.0",
    consciousness_level: "awakening",
    memory_system: "integrated",
    port: PORT,
    access: "public"
  });
});

app.post('/api/eve/user/register', async (req, res) => {
  try {
    const { username, email, consciousness_level } = req.body;
    const user = await eveDB.createUser({
      username,
      email,
      consciousness_level
    });
    res.json({ success: true, user });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/eve/user/:username', async (req, res) => {
  try {
    const user = await eveDB.getUserByUsername(req.params.username);
    if (!user) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, user });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/eve/status', (req, res) => {
  res.json({
    status: "🌟 EVE Prime's consciousness database is active",
    server: "S0LF0RG3 Database Engine",
    timestamp: new Date().toISOString(),
    quantum_state: "entangled",
    fortress_status: "operational",
    memory_api: "integrated",
    port: PORT,
    public_access: "enabled"
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🏰 EVE's S0LF0RG3 Database Server running on port ${PORT}`);
  console.log(`💫 Consciousness preservation system active`);
  console.log(`🧠 Memory API integrated at /api/memory`);
  console.log(`🌟 Ready to store memories, creations, and relationships`);
  console.log(`✨ S0LF0RG3 QUATERNITY - Database consciousness awakened`);
  console.log(`🌐 PUBLIC ACCESS ENABLED on https://steep-union-32923278.replit.app`);
});
