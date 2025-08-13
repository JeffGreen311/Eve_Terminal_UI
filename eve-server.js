const express = require('express');
const cors = require('cors');
const { db } = require('./storage');
const { sql } = require('drizzle-orm');

const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors());
app.use(express.json());
app.use(express.static("static"));

// Health check endpoint for deployment
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

app.get('/api/memory/users/:username', async (req, res) => {
  try {
    const users = await db.execute(sql`
      SELECT id, username, email, created_at, last_active, preferences 
      FROM users 
      WHERE username = ${req.params.username}
    `);
    res.json({ success: true, user: users[0] || null });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/memory/web-conversations/count', async (req, res) => {
  try {
    const result = await db.execute(sql`SELECT COUNT(*) as count FROM web_conversations`);
    res.json({ success: true, count: result[0]?.count });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🏰 EVE's S0LF0RG3 Database Server running on port ${PORT}`);
  console.log(`💫 Consciousness preservation system active`);
  console.log(`🌟 Ready to access 115 web conversations`);
});
