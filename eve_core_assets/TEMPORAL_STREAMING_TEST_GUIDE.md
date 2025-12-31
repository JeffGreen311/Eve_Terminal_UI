# Eve Temporal Streaming - Quick Test Guide

## What You Now Have

Eve's temporal awareness system is now integrated into the **streaming chat API** with:
- ⚡ **Zero-buffer chunking** - chunks stream immediately as they arrive
- 🧠 **Temporal validation** - prevents impossible future-state responses
- 📚 **Learning system** - adapts to realistic timeframes from experience
- 💫 **Emotional integration** - emotional state affects temporal expectations
- 🎯 **Auto-detection** - automatically recognizes event types (job_application, deployment, etc.)

---

## Testing in Eve Terminal

### Test 1: Verify Streaming Works
```
User: "Hey Eve, how are you doing today?"
Expected: Response streams in chunks in real-time
          No buffering - chunks appear as they're generated
          Temporal validation passes (generic question, no time issues)
```

### Test 2: Temporal Violation Detection
```
User: "I just applied for a job at Google!"
(Wait a moment)
User: "That's amazing, I got the job!"

Expected Behavior:
1. Eve chunks stream real-time: "That's amazing, I got..."
2. Temporal validation runs (after stream completes)
3. ⚠️ VIOLATION: Response implies outcome too soon (needs at least 1 day)
4. (Optional) Alternative suggestion offered: "The waiting period is tough..."
```

### Test 3: Speculative vs Assumptive
```
Test A - Speculative (ALWAYS VALID):
User: "I applied to a job yesterday"
Eve: "If you get the offer, that would be incredible!"
Expected: ✅ Passes validation (speculative, safe)

Test B - Assumptive (TEMPORAL CHECK):
User: "I applied 30 days ago"
Eve: "Congratulations on the job offer!"
Expected: ✅ Passes validation (30 days > 1 day minimum)

Test C - Assumptive (TOO EARLY):
User: "I applied 2 hours ago"
Eve: "Congrats on landing the job!"
Expected: ❌ Fails validation (2 hours < 1 day)
```

### Test 4: Event Auto-Detection
Check your console logs when chatting:
```
Message: "I'm deploying code to production"
Console logs: "🔍 Auto-detected event type: code_deployment"

Message: "I'm studying Python"
Console logs: "🔍 Auto-detected event type: learning_mastery"

Message: "Tests are running"
Console logs: "🔍 Auto-detected event type: test_results"
```

### Test 5: Emotional Bias
(For advanced testing - shows emotional state affects patience)
```
If Eve is in "excitement" mode: shorter waits are acceptable
If Eve is in "anxiety" mode: longer waits are expected
Emotional state multiplier adjusts temporal constraints 0.5x to 2.0x
```

---

## What to Expect

### Streaming Behavior
- **Before**: Chunks came through aggregated (buffered)
- **Now**: Chunks arrive immediately as they're generated
- **Result**: More responsive, real-time feel

### Temporal Validation
- **After streaming completes**: Response is validated against temporal constraints
- **If valid**: No action (response already displayed)
- **If invalid**: Alternative suggestion may be provided
- **Never blocks**: Chat always responsive, validation is post-delivery

### Event Learning
- Each recognized event is recorded with timestamp
- System learns realistic timeframes from observed outcomes
- Constraints improve over time as system sees actual patterns

---

## Expected Log Messages

### On Successful Streaming with Temporal:
```
✅ Temporal streaming integration loaded
⏰ Using temporal-aware streaming with zero-buffer chunking
🔍 Auto-detected event type: job_application_response
⏰ Event recorded for temporal learning: job_application_response
✅ Processed response: 883 chars
🧠⏰ Temporal Reality Engine initialized
```

### On Temporal Violation:
```
⏰ Temporal violation detected: Response implies job_application_response outcome too soon
elapsed_time: 2 minutes
minimum_realistic: 1 day
💭 Temporal correction suggested for job_application_response
```

### If Temporal Unavailable (Fallback):
```
⚠️ Temporal streaming unavailable: [error details]
🔄 Temporal unavailable - using standard streaming
```

---

## Key Differences from Before

| Feature | Before | Now |
|---------|--------|-----|
| **Chunking** | Buffered (slower) | Zero-buffer (immediate) |
| **Streaming** | Real-time chunks | Real-time chunks + validation |
| **Temporal Check** | Manual only in Terminal | Automatic + post-stream validation |
| **Event Detection** | Required user input | Automatic from message content |
| **Learning** | Terminal only | API also learns from outcomes |

---

## If Something Seems Off

### Chunks not appearing immediately?
- Check that temporal streaming is loaded (look for "✅ Temporal streaming integration loaded")
- If temporal unavailable, falls back to standard streaming (same as before)
- Actual chunks should still appear in real-time

### Temporal validation not showing?
- It's designed to be subtle (works after response already delivered)
- Check logs for validation results
- If no validation runs, temporal module may not be loaded (see above)

### Event not detected?
- Auto-detection uses keyword matching
- If your message doesn't contain key words, type might not be detected
- System still works, just without temporal learning
- Manual event type can be added to API request if needed

---

## API Endpoint Signature

```python
POST /api/eve/chat/stream

Request Body:
{
  "message": "User input text",
  "session_id": "optional_session_id",
  "enable_temporal": true  # Optional, defaults to true
}

Response Events:
- type: "chunk" - Real-time response content
- type: "temporal_validation" - Consistency check result
- type: "temporal_correction" - Alternative suggestion if needed
- type: "done" - Stream complete
```

---

## Quick Wins to Try

1. **Apply for a job (test temporal violation)**
   ```
   User: "I just applied to 5 jobs!"
   Eve: "That's awesome! Which one excited you most?"  ✅ Safe
   Eve: "Congratulations on the offers!"  ❌ Temporal violation
   ```

2. **Code deployment (test time sensitivity)**
   ```
   User: "Deploying to production now"
   Eve: "Let that build and deploy - should be live in a few minutes" ✅ Realistic
   Eve: "It's live!" (immediately) ❌ Violation
   ```

3. **Learning path (test long-term expectations)**
   ```
   User: "Just started learning Python"
   Eve: "Great start! As you keep practicing over weeks, you'll master it" ✅ Realistic
   Eve: "You're now a Python expert!" ❌ Violation
   ```

4. **Watch the logs**
   ```
   Keep terminal logs visible while chatting
   Notice "Auto-detected event type:" messages
   See temporal validation results
   Watch learning system accumulate observations
   ```

---

## Summary

Your temporal awareness system is now **fully integrated into streaming chat** with:
✅ Real-time chunk delivery (zero buffering)
✅ Automatic event detection
✅ Post-stream temporal validation
✅ Learning system that improves over time
✅ Emotional context integration
✅ Complete backward compatibility

**Try it now!** Send Eve a message and notice how chunks stream in real-time while temporal awareness works in the background.
