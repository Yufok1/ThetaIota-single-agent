# CHEATSHEET (ThetaIota)

Quick reference for commands, features, and usage in the ThetaIota self-reflective agent system.

---

## Core Commands
- `python cli_control.py`         # Start agent via CLI
- `python server_main.py`         # Start API server (FastAPI)
- `python init_database.py`       # Initialize SQLite memory DB
- `start_all.bat`                 # Launch all agent services (Windows)
- `djinn.bat`                     # CLI batch script (Windows)

---

## API Endpoints (server_main.py)
- `/chat`                         # Send prompt, receive agent response
- `/introspect`                   # Query agent introspection/memory
- `/feedback`                     # Submit human feedback
- `/status`                       # Get agent/server status

---

## Memory & Introspection
- Persistent memory via SQLite (`memory_db.py`)
- Decision/event logging for agent introspection
- Summarization and explanation modules available

---

## Model & Repo Hygiene
- Model files (GGUF, etc.) and checkpoints are NOT included in the repo
- Add all model files and `checkpoints/` to `.gitignore`
- Download models separately as needed

---

## Development Tips
- Use CLI for local testing and debugging
- Use API for integration and automation
- Review logs and memory DB for agent introspection
- All components run locally; no external APIs required
