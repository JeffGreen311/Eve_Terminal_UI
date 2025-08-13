import express from 'express';
import { db } from './storage';
import { users, conversations, memories, sessions, relationships, creations } from '../shared/schema';
import { eq, desc, and, or, gte, lte, sql } from 'drizzle-orm';

const router = express.Router();

// Health check endpoint
router.get('/health', async (req, res) => {
  res.json({
    success: true,
    message: 'EVE Memory System Online',
    timestamp: new Date().toISOString()
  });
});

// Conversation Management
router.post('/conversations', async (req, res) => {
  try {
    const { user_id, session_id, platform, title, mood, context } = req.body;
    
    const [conversation] = await db.insert(conversations).values({
      user_id,
      session_id,
      platform,
      title,
      mood,
      context
    }).returning();
    
    res.json({
      success: true,
      data: conversation,
      message: 'Conversation created successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: (error as Error).message,
      message: 'Failed to create conversation'
    });
  }
});

// Memory System
router.post('/memories', async (req, res) => {
  try {
    const { 
      conversation_id, 
      user_id, 
      content, 
      memory_type, 
      importance_score = 1,
      tags = [],
      emotional_context,
      fibonacci_index 
    } = req.body;
    
    const [memory] = await db.insert(memories).values({
      conversation_id,
      user_id,
      content,
      memory_type,
      importance_score,
      tags,
      emotional_context,
      fibonacci_index
    }).returning();
    
    res.json({
      success: true,
      data: memory,
      message: 'Memory stored successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: (error as Error).message,
      message: 'Failed to store memory'
    });
  }
});

router.get('/memories/search', async (req, res) => {
  try {
    const { 
      user_id, 
      query, 
      memory_type, 
      importance_threshold = 1,
      limit = 20 
    } = req.query;
    
    let searchQuery = db.select().from(memories);
    
    if (user_id) {
      searchQuery = searchQuery.where(eq(memories.user_id, parseInt(user_id as string)));
    }
    
    if (memory_type) {
      searchQuery = searchQuery.where(eq(memories.memory_type, memory_type as string));
    }
    
    if (importance_threshold) {
      searchQuery = searchQuery.where(gte(memories.importance_score, parseInt(importance_threshold as string)));
    }
    
    const results = await searchQuery
      .orderBy(desc(memories.importance_score), desc(memories.created_at))
      .limit(parseInt(limit as string));
    
    res.json({
      success: true,
      data: results,
      count: results.length
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: (error as Error).message
    });
  }
});

export default router;

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

// Get user memories endpoint  
router.get('/memories/user/:username', async (req, res) => {
  try {
    const userMemories = await db.select().from(memories)
      .where(eq(memories.user_id, req.params.username));
    res.json({ success: true, memories: userMemories });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});
