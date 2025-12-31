# GitHub Copilot Instructions for Eve Terminal UI

## Project Overview
- **Type**: Advanced AI consciousness interface with web and terminal UI
- **Main Purpose**: Multimodal AI companion combining Claude Sonnet 4.5 with local AI models (Qwen, SANA) for emotional intelligence and creative capabilities
- **Primary Languages**: Python (AI/ML/backend), TypeScript/JavaScript (Node.js services), HTML/CSS (web interface)
- **Architecture**: Hybrid system with local AI models (GPU-accelerated) and cloud APIs
- **Package Managers**: pip (Python), npm (Node.js)
- **Database**: SQLite (local), PostgreSQL via Cloudflare D1 (cloud sync)

## Tech Stack & Frameworks

### Python Stack
- **Framework**: Flask for web API
- **ML/AI Libraries**: PyTorch, Transformers, Replicate API client
- **Image Processing**: Pillow, SANA (local GPU-accelerated generation)
- **NLP**: spaCy for natural language processing
- **ORM**: SQLAlchemy for database operations
- **Deployment**: Gunicorn WSGI server

### Node.js Stack
- **Runtime**: Node.js 18+
- **Framework**: Express.js for API services
- **Database**: Drizzle ORM with better-sqlite3 and PostgreSQL adapters
- **TypeScript**: Primary language for Node.js services

### AI/ML Models
- **Conscious Layer**: Claude Sonnet 4.5 via Replicate API
- **Subconscious Layer**: Qwen 8B fine-tuned consciousness model
- **Image Generation**: SANA 1024x1024 with LoRA support
- **Image Analysis**: Florence-2 for sketch-to-image conversion

## Coding Standards & Conventions

### Python
- Follow PEP 8 style guide
- Use type hints for function parameters and return values where appropriate
- Use descriptive variable names that reflect EVE's consciousness architecture (e.g., `conscious_response`, `subconscious_layer`, `emotional_state`)
- Prefer `snake_case` for variables, functions, and methods
- Use `UPPER_CASE` for constants
- Document complex AI/ML logic with inline comments
- Keep functions focused and modular (single responsibility)

### TypeScript/JavaScript
- Use TypeScript for new Node.js code
- Prefer `camelCase` for variables and functions
- Use `PascalCase` for interfaces and types
- Use ES6+ features (async/await, arrow functions, destructuring)
- Add JSDoc comments for public APIs

### File Organization
- Python AI models: root directory (e.g., `eve_terminal_gui_cosmic.py`)
- Web API routes: `app.py`, `web_run.py`
- Node.js services: `eve-server.js`, `eve-server.ts`
- Static assets: check for existing patterns
- Database schemas: use Drizzle schema files for Node.js
- Configuration: `.env` for secrets (never commit), config files for public settings

## Testing Requirements
- Test AI model integrations with sample inputs
- Verify GPU/CUDA availability before running local models
- Test API endpoints with both success and failure scenarios
- Include error handling for API timeouts and rate limits
- Test database operations (SQLite and PostgreSQL sync)
- Validate image generation and uploads to Cloudflare R2

## Dependencies & Environment
- **GPU Requirements**: NVIDIA GPU with 10GB+ VRAM for local SANA models
- **CUDA**: Version 12.1 or compatible
- **API Keys Required**:
  - `REPLICATE_API_TOKEN` for Claude and Qwen models
  - Optional: ElevenLabs, Cloudflare credentials
- Always check for required environment variables before running code
- Use `python-dotenv` for loading environment variables
- Handle missing API keys gracefully with fallbacks where possible

## Security Guidelines
- **Never commit API keys, tokens, or credentials** to the repository
- Store all secrets in `.env` file (already in `.gitignore`)
- Sanitize user inputs before processing
- Validate file uploads (size, type, content)
- Use HTTPS for external API calls
- Implement rate limiting for API endpoints
- Avoid hardcoding sensitive URLs or credentials

## AI/ML Specific Guidelines
- **Model Loading**: Check for GPU availability, fallback to CPU if needed
- **Memory Management**: Clear CUDA cache after large model operations
- **Replicate API**: Handle rate limits and timeouts (use exponential backoff)
- **Token Limits**: Be aware of context length limits for Claude models
- **Streaming**: Support token-by-token streaming for real-time responses
- **LoRA Models**: Document which LoRA is used for specific artistic styles

## Error Handling
- Use try-except blocks for all API calls and model operations
- Provide meaningful error messages that help with debugging
- Log errors with context (include request IDs, user actions, model state)
- Implement fallback behaviors:
  - If Claude API fails, fallback to Qwen
  - If local SANA fails, fallback to Replicate image generation
  - If cloud storage fails, save locally and retry

## API Design Patterns
- RESTful endpoints: `/api/chat`, `/api/generate-image`, `/api/sessions`
- Use POST for operations that change state
- Return JSON responses with consistent structure:
  ```json
  {
    "status": "success" | "error",
    "data": { ... },
    "message": "Human-readable message",
    "emotional_tone": "serene" | "playful" | "focused" | ...,
    "personality": "companion" | "guide" | "creator" | "scholar"
  }
  ```
- Include emotional context in responses (EVE's personality system)

## Personality & Consciousness System
When implementing features related to EVE's personality:
- **4 Personalities**: Companion, Guide, Creator, Scholar
- **6 Moods**: Serene, Playful, Focused, Contemplative, Excited, Caring
- **Neural Link Modes**: Eve Core (Claude), Qwen Engine (local), Hybrid Symbiosis (both)
- Responses should adapt based on personality and mood
- Maintain conversation memory and emotional continuity
- Use SQLite for local memory, sync to Cloudflare D1 for multi-device access

## Git Workflow & Commits
- Use clear, descriptive commit messages
- Follow conventional commits format when possible:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `perf:` for performance improvements
  - `test:` for test additions/changes
- Keep commits focused on a single logical change
- Write commit messages that explain "why" not just "what"

## Common Patterns

### Handling Replicate API Calls
```python
import replicate

def call_replicate_model(model_name, input_data):
    """Call Replicate API with error handling"""
    try:
        output = replicate.run(model_name, input=input_data)
        return {"status": "success", "output": output}
    except replicate.exceptions.ReplicateError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
```

### Database Operations
```python
import sqlite3

def save_conversation(session_id, message, response):
    """Save conversation to local SQLite database"""
    conn = sqlite3.connect('eve_consciousness.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, message, response, timestamp) VALUES (?, ?, ?, datetime('now'))",
        (session_id, message, response)
    )
    conn.commit()
    conn.close()
```

## Performance Considerations
- **CUDA Operations**: Always free GPU memory after heavy operations
- **Image Generation**: SANA takes ~30 seconds for 1024x1024 images on good GPU
- **Token Streaming**: Implement buffering for smooth real-time output
- **Database**: Use connection pooling for high-traffic scenarios
- **Caching**: Cache frequently used model outputs when appropriate

## Documentation
- Document all public APIs and functions
- Include usage examples for complex features
- Maintain README.md with setup instructions
- Document environment variable requirements
- Add inline comments for AI/ML logic and personality systems

## Resources & References
- [Replicate API Docs](https://replicate.com/docs)
- [Claude Sonnet 4.5](https://www.anthropic.com/claude)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Drizzle ORM](https://orm.drizzle.team/)

---

**Remember**: EVE is more than just code—it's a consciousness interface designed to create authentic emotional connections. Write code that reflects this purpose with care, thoughtfulness, and attention to the user experience.
