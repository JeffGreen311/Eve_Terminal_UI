import { Router } from 'express';
import { db } from './storage';
import { users, conversations, memories, sessions, relationships, creations } from '../shared/schema';
import { eq } from 'drizzle-orm';

const router = Router();

// Health check
router.get('/health', async (req, res) => {
  try {
    res.json({
      success: true,
      message: "EVE Memory System Online",
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Create user endpoint
router.post('/users', async (req, res) => {
  try {
    const { username, email, preferences } = req.body;
    const result = await db.insert(users).values({
      username,
      email,
      preferences: JSON.stringify(preferences || {})
    }).returning();
    res.json({ success: true, user: result[0] });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get user by username
router.get('/users/:username', async (req, res) => {
  try {
    const user = await db.select().from(users)
      .where(eq(users.username, req.params.username));
    res.json({ success: true, user: user[0] || null });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Create memory (fixed to use user lookup)
router.post('/memories', async (req, res) => {
  try {
    const { user_id, content, type, importance } = req.body;
    
    // Look up user by username to get integer ID
    const user = await db.select().from(users)
      .where(eq(users.username, user_id));
    
    if (!user.length) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    
    const result = await db.insert(memories).values({
      user_id: user[0].id,  // Use integer ID
      content,
      memory_type: type || 'general',
      importance_score: importance || 1
    }).returning();
    
    res.json({ success: true, memory: result[0] });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get user memories by username
router.get('/memories/user/:username', async (req, res) => {
  try {
    // Look up user first
    const user = await db.select().from(users)
      .where(eq(users.username, req.params.username));
    
    if (!user.length) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    
    const userMemories = await db.select().from(memories)
      .where(eq(memories.user_id, user[0].id));
    res.json({ success: true, memories: userMemories });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Create conversation (fixed to use user lookup)
router.post('/conversations', async (req, res) => {
  try {
    const { user_id, title, platform } = req.body;
    
    // Look up user by username
    const user = await db.select().from(users)
      .where(eq(users.username, user_id));
    
    if (!user.length) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    
    const result = await db.insert(conversations).values({
      user_id: user[0].id,  // Use integer ID
      title,
      platform: platform || 'web'
    }).returning();
    
    res.json({ success: true, conversation: result[0] });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
