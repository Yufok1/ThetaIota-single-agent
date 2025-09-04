# ThetaIota 🤖

**Self-reflective AI agent system with advanced introspection, persistent memory, and seamless multi-line input support.**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Yufok1/ThetaIota-single-agent)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌟 Key Features

### Core Capabilities
- **🧠 Self-Reflective Agent**: Advanced introspection with decision logging and meta-learning
- **💾 Persistent Memory**: SQLite-based memory system with summarization and querying
- **🔄 Multi-Phase Architecture**: Progressive agent development from basic to self-aware
- **🚀 Smart CLI Input**: Seamless multi-line paste support - no special commands needed!
- **🌐 REST API**: FastAPI-based server for programmatic access
- **👥 Human Feedback Integration**: Real-time feedback loops for agent improvement
- **🔧 Local-First**: Everything runs locally - no external API dependencies

### Advanced Features
- **Decision Tracking**: Transparent logging of agent reasoning and self-updates
- **Task Orchestration**: Dynamic task spawning based on confidence and performance
- **Memory Summarization**: Intelligent compression of conversation history
- **Model Training**: Built-in transformer and language model training scripts
- **Cross-Platform**: Windows batch scripts and Unix compatibility
- **Comprehensive Testing**: Full test suite with evaluation frameworks
- **Enhanced CLI**: Improved input handling for better user experience

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Yufok1/ThetaIota-single-agent.git
cd ThetaIota-single-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
python init_database.py
```

### 4. Download Models (Optional)
```bash
# Download GGUF model files and place in appropriate directory
# See model provider documentation for specific instructions
```

### 5. Launch Agent
```bash
# CLI Mode (Recommended)
python cli_control.py

# Or use Windows batch file
djinn.bat

# API Server Mode
python server_main.py
```

---

## 💬 Enhanced CLI Interface

**Improved input handling for a better user experience.**

### ✨ Key Features
- **Flexible Input**: Support for both single-line and multi-line text input
- **Cross-Platform**: Works seamlessly on Windows, macOS, and Linux
- **Mode Switching**: Easy switching between different agent modes during conversation
- **Session Management**: Persistent conversation contexts

### 🎯 Usage Examples
```bash
# Single line input
> Hello, how are you?

# Multi-line input (paste directly)
> You are an expert software architect with 20 years of experience.
> Please analyze this codebase and provide recommendations for:
> 1. Code organization
> 2. Performance optimization
> 3. Security improvements

# Mode switching during conversation
> !lm (switches to language model mode)
> !reflect (switches to reflection mode)
```

---

## 🏗️ Architecture Overview

### Phase-Based Development
- **Phase 1**: Basic agent with core functionality
- **Phase 2**: Meta-controller for decision making
- **Phase 3**: Self-awareness and introspection
- **Phase 4**: Production-ready service architecture
- **Phase 5**: Advanced communication and consensus

### Core Components
```
├── 🤖 Agent Core
│   ├── chat_engine.py          # Main conversation logic
│   ├── meta_controller.py      # Introspection & meta-learning
│   └── reflection_explainer.py # Decision explanations
│
├── 💾 Memory System
│   ├── memory_db.py            # SQLite persistence
│   ├── memory_summarizer.py    # History compression
│   └── human_feedback_system.py # Feedback integration
│
├── 🚀 Interfaces
│   ├── cli_control.py          # Smart CLI with paste support
│   ├── server_main.py          # FastAPI server
│   └── phase3_shell.py         # Advanced shell interface
│
└── 🧪 Training & Testing
    ├── transformer_model.py    # Custom transformer
    ├── train_*.py             # Training scripts
    └── test_*.py              # Comprehensive tests
```

---

## 📚 API Reference

### REST Endpoints
```http
POST /chat              # Send message, receive response
GET  /introspect        # Query agent memory/introspection
POST /feedback          # Submit human feedback
GET  /status            # Agent/server health status
POST /task              # Spawn new tasks
GET  /memory/query      # Query persistent memory
```

### CLI Commands
```bash
# Interactive mode
python cli_control.py

# Direct commands
python cli_control.py chat --text "Hello" --mode reflect
python cli_control.py query --text "What was my last decision?"
python cli_control.py status --agent A
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Custom API authentication
export RA_AUTH_SECRET="your-secret-key"

# Optional: Custom model paths
export MODEL_PATH="/path/to/your/model.gguf"
```

### Model Setup
- Download GGUF format models from your preferred provider
- Place in project directory (excluded from git)
- Models are automatically detected and loaded

---

## 🧪 Testing & Evaluation

### Run Tests
```bash
# Phase 3 complete test
python test_phase3_complete.py

# API functionality test
python test_phase4_api.py

# Canary evaluation
python canary_eval.py
```

### Evaluation Framework
- **Canary Prompts**: Adversarial test cases in `eval/canary_prompts.jsonl`
- **Performance Metrics**: Validation loss, response quality, introspection accuracy
- **Human Feedback**: Real-time quality assessment integration

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Complete file map and organization |
| [`design_doc.md`](design_doc.md) | Architecture and design principles |
| [`arch_blueprint.md`](arch_blueprint.md) | System architecture diagrams |
| [`CHEATSHEET.md`](CHEATSHEET.md) | Quick command reference |
| [`phase*_verification.md`](phase0_verification.md) | Phase-specific verification |

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run linting
black . && flake8 .

# Run type checking
mypy .

# Run tests
pytest
```

---

## 📋 Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for models and checkpoints
- **OS**: Windows, macOS, Linux

### Dependencies
- **FastAPI**: High-performance web framework
- **PyTorch**: Deep learning framework
- **SQLite**: Local database (built-in)
- **NumPy**: Numerical computing
- **Uvicorn**: ASGI server

---

## 🛡️ Security & Privacy

- **Local-First**: All processing happens on your machine
- **No Data Collection**: No telemetry or external data sharing
- **Model Isolation**: Downloaded models stay in your local environment
- **Memory Encryption**: Optional encryption for sensitive conversation data

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with modern Python and PyTorch
- Inspired by advanced AI safety and alignment research
- Designed for transparent, introspective AI development

---

**Ready to explore self-reflective AI?** Start with `python cli_control.py` and paste your first multi-line prompt! 🚀
