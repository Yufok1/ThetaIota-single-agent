# Design Document (ThetaIota)

This document outlines the architecture, introspection, and operational principles of the ThetaIota self-reflective agent system.

---

## System Overview
- **Agent:** Self-reflective agent with introspection, decision logging, and memory.
- **Introspection:** Tracks training dynamics, performance, and decisions; logs meta-events for analysis.
- **Decision Logging:** Explains self-updates, task spawning, and learning adjustments in a transparent, queryable log.
- **Local-Only:** All components run locally; no external APIs or cloud dependencies.

---

## Self-Awareness & Introspection
- Monitors: training/validation loss, gradient norms, parameter statistics, token confidence, resource usage.
- Triggers: self-update, sub-task generation, memory summarization, human feedback integration.

---

## Task & Meta-Learning
- Agent can spawn tasks based on confidence, plateau, error patterns, or resource limits.
- Meta-controller adjusts learning rate, training mode, attention focus, and memory management.
- Success metrics: validation loss, training stability, human feedback.

---

## Interaction Modalities
- **CLI:** Local command-line interface for development and debugging.
- **API:** FastAPI server for programmatic access and monitoring.
- **Batch Scripts:** Launch agent and training via Windows batch files.

---

## Repo Hygiene
- Model files (GGUF, etc.) and checkpoints are NOT included in the repo. Download separately and add to `.gitignore`.
- See PROJECT_STRUCTURE.md and README.md for setup and file handling details.

---

## Phase 0 Success Criteria
- Design document complete
- Architecture and resource constraints documented
- All deliverables verified before coding begins