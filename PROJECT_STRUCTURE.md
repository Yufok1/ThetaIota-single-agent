# Project Structure (ThetaIota)

This document describes the organization, key files, and setup for the ThetaIota self-reflective agent system.

---

## Directory & File Map

```
/ (root)
├── README.md                # Project overview & setup
├── design_doc.md            # System design & architecture
├── arch_blueprint.md        # Architecture diagrams & notes
├── CHEATSHEET.md            # Quick reference for commands/features
├── PROJECT_STRUCTURE.md     # This file
├── requirements.txt         # Python dependencies
├── setup.py                 # Packaging config (optional)
├── LICENSE                  # License info
├── .gitignore               # Exclude model files, checkpoints, etc.
│
├── chat_engine.py           # Core chat agent logic
├── memory_db.py             # Persistent memory (SQLite)
├── meta_controller.py       # Meta-controller for introspection & logging
├── cli_control.py           # CLI interface
├── server_main.py           # API server (FastAPI)
│
├── phase1_agent.py          # Phase 1: base agent
├── phase2_agent.py          # Phase 2: meta-controller
├── phase3_agent.py          # Phase 3: self-awareness
├── phase3_shell.py          # Shell interface for phase 3
├── phase4_agent_service.py  # Phase 4: production wrapper
├── phase4_api_server.py     # Phase 4: API server
├── phase5_communication.py  # Communication logic
├── phase5_consensus.py      # Consensus logic
├── phase5_registry.py       # Registry logic
├── phase5_shared_memory.py  # Shared memory logic
│
├── human_feedback_system.py # Human feedback integration
├── memory_summarizer.py     # Memory summarization
├── reflection_explainer.py  # Decision explanation
├── task_spawner.py          # Task orchestration
├── init_database.py         # DB initialization
├── toy_dataset.py           # Example dataset
│
├── train_conversational_lm_windows.py      # LM training (Windows)
├── train_minimal_transformer_windows.py    # Minimal transformer training (Windows)
├── train_tiny_lm.py         # Tiny LM training
├── transformer_model.py     # Transformer model implementation
│
├── djinn.bat                # CLI batch script
├── djinn-train.bat          # Training batch script
├── start_all.bat            # Launch all agents
│
├── test_phase3_complete.py  # Phase 3 agent test
├── test_phase4_api.py       # Phase 4 API test
│
├── eval/                    # Evaluation data/results
│   └── canary_prompts.jsonl # Canary prompt set
├── checkpoints/             # Model checkpoints (excluded from repo)
├── guardian/                # Guardian agent & validator
│   └── validator.py         # Validator logic
```

---

## Repo Hygiene & Model Files
- **Model files (GGUF, etc.) and checkpoints are NOT included in the repo.**
- Add all model files and `checkpoints/` to `.gitignore`.
- Download models separately as needed (see README.md).

---

## Quick Start
1. Install Python dependencies: `pip install -r requirements.txt`
2. Download GGUF model file and place in the correct directory
3. Initialize database: `python init_database.py`
4. Start agent via CLI: `python cli_control.py` or `djinn.bat`
5. (Optional) Start API server: `python server_main.py`

---

## Development Workflow
- Clone repo
- Create virtual environment
- Install dependencies
- Download model files
- Run tests: `pytest`
- Launch agent(s) as needed
- Use CLI or API as needed

---

## Notes
- All agents, controllers, and memory modules run locally. No external APIs required.
- See other docs for architecture, design, and command reference.
