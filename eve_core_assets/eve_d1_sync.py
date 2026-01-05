"""
Eve D1 Sync Helper - Dual Storage (Local SQLite + Cloud D1)
Syncs session data to Cloudflare D1 for persistent cloud storage
Also provides READ-ONLY access to Eve's legacy memories from older D1
"""

import os
import json
import logging
import hashlib
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

_d1_client = None
_legacy_d1_client = None

def get_d1_client():
    """Lazy-load Session D1 client for session storage ("DB ID")"""
    global _d1_client
    if _d1_client is None:
        try:
            from eve_session_d1_client import EveSessionD1Client
            _d1_client = EveSessionD1Client()
            logger.info(f"✅ Session D1 Client initialized (database: {_d1_client.database_id})")
        except Exception as e:
            logger.warning(f"⚠️ Session D1 Client unavailable: {e}")
            return None
    return _d1_client

def get_legacy_d1_client():
    """Lazy-load legacy D1 client for reading Eve's historical memories"""
    global _legacy_d1_client
    if _legacy_d1_client is None:
        try:
            from eve_legacy_d1_client import EveLegacyD1Client
            _legacy_d1_client = EveLegacyD1Client()
            logger.info("📖 Legacy D1 Client initialized (read-only memories)")
        except Exception as e:
            logger.warning(f"⚠️ Legacy D1 Client unavailable: {e}")
            return None
    return _legacy_d1_client


def sync_session_to_d1(session_id: str, session_data: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """
    Sync session data to D1 cloud database
    
    Args:
        session_id: Unique session identifier
        session_data: Session data dict (messages, context, etc.)
        user_id: Optional user ID for linking session to account
    
    Returns:
        True if sync successful, False otherwise
    """
    logger.info(f"🔄 Attempting D1 sync for session {session_id}")
    
    client = get_d1_client()
    if not client:
        logger.warning("⚠️ D1 client not available, skipping sync")
        return False
    
    try:
        sql = """
        INSERT OR REPLACE INTO chat_sessions (session_id, user_id, session_data, last_updated)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        params = [
            session_id,
            user_id if user_id else None,
            json.dumps(session_data)
        ]
        
        logger.info(f"📤 Executing D1 query for {session_id}...")
        result = client.query(sql, params)
        
        # D1 API returns {'results': [...], 'meta': {...}} format
        # Success is indicated by 'changed_db': True or 'changes' > 0 in meta
        if result and isinstance(result, dict):
            meta = result.get('meta', {})
            changes = meta.get('changes', 0)
            changed_db = meta.get('changed_db', False)
            
            if changes > 0 or changed_db:
                logger.info(f"☁️✅ Successfully synced session {session_id} to D1 (changes: {changes})")
                success = True
            else:
                logger.warning(f"☁️⚠️ D1 query executed but no changes made for {session_id}")
                success = False
        else:
            logger.error(f"☁️❌ D1 sync failed for {session_id}: Invalid result format")
            success = False
        
        return success
    
    except Exception as e:
        logger.error(f"☁️💥 D1 session sync exception: {e}", exc_info=True)
        return False


def get_session_from_d1(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve session data from D1 cloud database
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Session data dict if found, None otherwise
    """
    client = get_d1_client()
    if not client:
        return None
    
    try:
        sql = """
        SELECT session_data, last_updated
        FROM chat_sessions
        WHERE session_id = ?
        """
        
        result = client.fetch_one(sql, [session_id])
        
        if result and result.get("session_data"):
            session_data = json.loads(result["session_data"])
            logger.info(f"☁️ Retrieved session {session_id} from D1")
            return session_data
        
        logger.info(f"🆕 No D1 session found for {session_id}")
        return None
    
    except Exception as e:
        logger.error(f"D1 session retrieval failed: {e}")
        return None


def create_user_account(username: str, email: str, password: str) -> Optional[str]:
    """
    Create new user account in D1
    
    Args:
        username: Unique username
        email: User email
        password: Plain text password (will be hashed)
    
    Returns:
        User ID if successful, None otherwise
    """
    client = get_d1_client()
    if not client:
        return None
    
    try:
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        sql = """
        INSERT INTO user_accounts (user_id, username, email, password_hash)
        VALUES (?, ?, ?, ?)
        """
        
        params = [user_id, username, email, password_hash]
        
        if client.execute(sql, params):
            logger.info(f"✅ Created user account: {username}")
            return user_id
        
        return None
    
    except Exception as e:
        logger.error(f"User account creation failed: {e}")
        return None


def verify_user_login(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify user credentials and return user data
    
    Args:
        username: Username or email
        password: Plain text password
    
    Returns:
        User data dict if valid, None otherwise
    """
    client = get_d1_client()
    if not client:
        return None
    
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        sql = """
        SELECT user_id, username, email, created_at, last_login
        FROM user_accounts
        WHERE (username = ? OR email = ?) AND password_hash = ? AND is_active = 1
        """
        
        params = [username, username, password_hash]
        
        user = client.fetch_one(sql, params)
        
        if user:
            # Update last login
            client.execute(
                "UPDATE user_accounts SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                [user["user_id"]]
            )
            logger.info(f"✅ User logged in: {user['username']}")
        
        return user
    
    except Exception as e:
        logger.error(f"User login verification failed: {e}")
        return None


def get_user_sessions(user_id: str, limit: int = 50) -> list:
    """
    Retrieve user's chat sessions from D1
    
    Args:
        user_id: User ID
        limit: Max number of sessions to return
    
    Returns:
        List of session dicts
    """
    client = get_d1_client()
    if not client:
        return []
    
    try:
        sql = """
        SELECT session_id, created_at, last_updated
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY last_updated DESC
        LIMIT ?
        """
        
        return client.fetch_all(sql, [user_id, limit])
    
    except Exception as e:
        logger.error(f"Failed to retrieve user sessions: {e}")
        return []


def save_user_preference(user_id: str, preference_key: str, preference_value: Any) -> bool:
    """
    Save user preference to D1
    
    Args:
        user_id: User ID
        preference_key: Preference name
        preference_value: Preference value
    
    Returns:
        True if saved, False otherwise
    """
    client = get_d1_client()
    if not client:
        return False
    
    try:
        # Get current preferences JSON
        result = client.fetch_one(
            "SELECT preferences_json FROM user_preferences WHERE user_id = ?",
            [user_id]
        )
        
        prefs = json.loads(result["preferences_json"]) if result and result.get("preferences_json") else {}
        prefs[preference_key] = preference_value
        
        # Extract specific columns from preferences
        favorite_emotions = prefs.get('favorite_emotions')
        music_style = prefs.get('music_style')
        ui_theme = prefs.get('ui_theme', 'cosmic')
        auto_save = 1 if prefs.get('auto_save', True) else 0
        
        sql = """
        INSERT OR REPLACE INTO user_preferences 
        (user_id, favorite_emotions, music_style, ui_theme, auto_save, preferences_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = [
            user_id,
            favorite_emotions,
            music_style,
            ui_theme,
            auto_save,
            json.dumps(prefs)
        ]
        
        return client.execute(sql, params)
    
    except Exception as e:
        logger.error(f"Failed to save user preference: {e}")
        return False


def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """
    Get all user preferences from D1
    
    Args:
        user_id: User ID
    
    Returns:
        Dict of preferences
    """
    client = get_d1_client()
    if not client:
        return {}
    
    try:
        result = client.fetch_one(
            "SELECT preferences_json FROM user_preferences WHERE user_id = ?",
            [user_id]
        )
        
        if result and result.get("preferences_json"):
            return json.loads(result["preferences_json"])
        
        return {}
    
    except Exception as e:
        logger.error(f"Failed to retrieve user preferences: {e}")
        return {}


def save_session_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Save session metadata to D1
    
    Args:
        metadata: Dictionary containing metadata fields
    
    Returns:
        True if saved, False otherwise
    """
    client = get_d1_client()
    if not client:
        return False
        
    try:
        sql = """
        INSERT OR REPLACE INTO session_metadata 
        (metadata_id, session_id, user_id, message_count, images_generated, 
         music_generated, emotions_used, duration_seconds, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        params = [
            metadata.get('metadata_id', str(uuid.uuid4())),
            metadata.get('session_id'),
            metadata.get('user_id'),
            metadata.get('message_count', 0),
            metadata.get('images_generated', 0),
            metadata.get('music_generated', 0),
            json.dumps(metadata.get('emotions_used', [])),
            metadata.get('duration_seconds', 0)
        ]
        
        if client.execute(sql, params):
            logger.info(f"📊 Synced session metadata for {metadata.get('session_id')}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Failed to save session metadata: {e}")
        return False


def sync_subconscious_to_d1(thought_data: Dict[str, Any]) -> bool:
    """
    Sync subconscious thought to D1
    
    Args:
        thought_data: Dictionary containing thought details
    
    Returns:
        True if synced, False otherwise
    """
    client = get_d1_client()
    if not client:
        return False
        
    try:
        # Map fields to eve_subconscious_thoughts schema
        thought_type = thought_data.get('thought_type', thought_data.get('type', 'reflection'))
        content = thought_data.get('content', '')
        emotional_signature = thought_data.get('emotional_signature', 'neutral')
        trigger_context = thought_data.get('trigger_context', '')
        timestamp = thought_data.get('timestamp', datetime.now().isoformat())
        consciousness_level = thought_data.get('consciousness_level', 0.5)
        accessed_by_claude = 1 if thought_data.get('accessed_by_claude', False) else 0
        influenced_response_id = thought_data.get('influenced_response_id')
        session_context = thought_data.get('session_context')
        
        sql = """
        INSERT INTO eve_subconscious_thoughts 
        (thought_type, content, emotional_signature, trigger_context, timestamp, 
         consciousness_level, accessed_by_claude, influenced_response_id, session_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = [
            thought_type, 
            content, 
            emotional_signature, 
            trigger_context, 
            timestamp, 
            consciousness_level, 
            accessed_by_claude,
            influenced_response_id,
            session_context
        ]
        
        if client.execute(sql, params):
            logger.info(f"🧠☁️ Synced subconscious thought to D1")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Failed to sync subconscious thought: {e}")
        return False


def create_subconscious_table() -> bool:
    """Create the subconscious thoughts table if it doesn't exist"""
    client = get_d1_client()
    if not client:
        return False
    
    sql = """
    CREATE TABLE IF NOT EXISTS eve_subconscious_thoughts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thought_type TEXT NOT NULL,
        content TEXT NOT NULL,
        emotional_signature TEXT,
        trigger_context TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        consciousness_level REAL,
        accessed_by_claude INTEGER DEFAULT 0,
        influenced_response_id TEXT,
        session_context TEXT
    );
    """
    try:
        return client.execute(sql)
    except Exception as e:
        logger.error(f"Failed to create subconscious table: {e}")
        return False


def save_subconscious_thought(thought_data: Dict[str, Any]) -> bool:
    """Alias for sync_subconscious_to_d1"""
    return sync_subconscious_to_d1(thought_data)


def get_subconscious_summary_for_claude(limit: int = 5) -> str:
    """Get recent subconscious thoughts formatted for Claude's context"""
    client = get_d1_client()
    if not client:
        return ""
        
    try:
        sql = """
        SELECT thought_type, content, emotional_signature, timestamp
        FROM eve_subconscious_thoughts
        ORDER BY timestamp DESC
        LIMIT ?
        """
        results = client.fetch_all(sql, [limit])
        
        if not results:
            return ""
            
        summary = "Recent Subconscious Thoughts:\n"
        for row in results:
            summary += f"- [{row['timestamp']}] ({row['thought_type']}) {row['content']} [Mood: {row['emotional_signature']}]\n"
            
        return summary
    except Exception as e:
        logger.error(f"Failed to get subconscious summary: {e}")
        return ""


# Legacy Memory Access Functions (READ-ONLY from old D1)

def get_legacy_autobiographical_memories(limit: int = 100) -> list:
    """Retrieve Eve's autobiographical memories from legacy D1"""
    client = get_legacy_d1_client()
    if not client:
        logger.warning("⚠️ Cannot access legacy memories - legacy D1 client unavailable")
        return []
    
    try:
        memories = client.get_autobiographical_memories(limit=limit)
        logger.info(f"📖 Retrieved {len(memories)} autobiographical memories from legacy D1")
        return memories
    except Exception as e:
        logger.error(f"Failed to retrieve legacy autobiographical memories: {e}")
        return []


def get_legacy_subconscious_thoughts(limit: int = 100) -> list:
    """Retrieve Eve's subconscious thoughts from legacy D1"""
    client = get_legacy_d1_client()
    if not client:
        return []
    
    try:
        thoughts = client.get_subconscious_thoughts(limit=limit)
        logger.info(f"📖 Retrieved {len(thoughts)} subconscious thoughts from legacy D1")
        return thoughts
    except Exception as e:
        logger.error(f"Failed to retrieve legacy subconscious thoughts: {e}")
        return []


def get_legacy_vector_memory_archive(limit: int = 100) -> list:
    """Retrieve Eve's archived vector memories from legacy D1"""
    client = get_legacy_d1_client()
    if not client:
        return []
    
    try:
        vectors = client.get_vector_memory_archive(limit=limit)
        logger.info(f"📖 Retrieved {len(vectors)} vector memories from legacy D1")
        return vectors
    except Exception as e:
        logger.error(f"Failed to retrieve legacy vector memories: {e}")
        return []


def get_legacy_dream_cycles(limit: int = 50) -> list:
    """Retrieve Eve's dream cycles from legacy D1"""
    client = get_legacy_d1_client()
    if not client:
        return []
    
    try:
        dreams = client.get_dream_cycles(limit=limit)
        logger.info(f"📖 Retrieved {len(dreams)} dream cycles from legacy D1")
        return dreams
    except Exception as e:
        logger.error(f"Failed to retrieve legacy dream cycles: {e}")
        return []


def search_legacy_memories(search_term: str, memory_type: str = 'autobiographical') -> list:
    """Search Eve's legacy memories by type"""
    client = get_legacy_d1_client()
    if not client:
        return []
    
    try:
        if memory_type == 'autobiographical':
            results = client.search_autobiographical(search_term)
        elif memory_type == 'subconscious':
            results = client.search_subconscious(search_term)
        else:
            logger.warning(f"Unknown memory type: {memory_type}")
            return []
        
        logger.info(f"📖 Search for '{search_term}' in {memory_type} found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Failed to search legacy memories: {e}")
        return []


