# ThetaIota

Self-reflective AI agent with introspection, decision logging, persistent memory, and local operation.

---

## Features
- Self-reflective agent with introspection and decision/event logging
- Persistent memory via SQLite
- Human feedback integration
- CLI and API (FastAPI) interfaces
- All components run locally; no external APIs required

---

## Quick Start
1. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Download GGUF model file and place in the correct directory (see model provider docs)
3. Initialize database:
   ```sh
   python init_database.py
   ```
4. Start agent via CLI:
   ```sh
   python cli_control.py
   ```
   or (Windows):
   ```sh
   djinn.bat
   ```
5. (Optional) Start API server:
   ```sh
   python server_main.py
   ```

---

## Usage
- Interact with the agent via CLI or API
- Review introspection and decision logs in the memory database
- Submit human feedback via API or CLI

---

## Repo Hygiene
- Model files (GGUF, etc.) and checkpoints are NOT included in the repo
- Add all model files and `checkpoints/` to `.gitignore`
- Download models separately as needed

---

## Documentation
- See `PROJECT_STRUCTURE.md` for file map and setup
- See `design_doc.md` for architecture and introspection details
- See `CHEATSHEET.md` for command reference
- See `arch_blueprint.md` for architecture diagram
