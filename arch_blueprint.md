# Architecture Blueprint (ThetaIota)

This blueprint summarizes the actual architecture and data flow of the ThetaIota self-reflective agent system.

---

## High-Level Diagram

```
User Input
   ↓
CLI / API (FastAPI)
   ↓
Agent Core (chat_engine.py, phase*_agent.py)
   ↓
Introspection & Decision Logging (meta_controller.py, reflection_explainer.py)
   ↓
Persistent Memory (SQLite, memory_db.py)
   ↓
Summarization / Feedback (memory_summarizer.py, human_feedback_system.py)
```

---

## Component Summary
- **Agent Core:** Handles prompts, reasoning, and responses.
- **Introspection:** Tracks training, decisions, and meta-events.
- **Memory:** Stores persistent state and logs in SQLite.
- **CLI/API:** Local interfaces for user and programmatic access.
- **Summarization/Feedback:** Optional modules for memory analysis and human feedback.

---

## Notes
- All components run locally; no external APIs or distributed federation.
- Model files (GGUF, etc.) and checkpoints are NOT included in the repo. Download separately and add to `.gitignore`.