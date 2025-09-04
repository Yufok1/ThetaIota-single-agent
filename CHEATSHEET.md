# CHEATSHEET (ThetaIota) ⚡

**Quick reference for commands, features, and usage in the ThetaIota self-reflective AI agent system.**

---

## 🚀 Quick Launch Commands

### Primary Interfaces
```bash
# 🎯 Smart CLI (Recommended - with multi-line paste support!)
python cli_control.py

# 🌐 API Server Mode
python server_main.py

# 🪟 Windows Batch Files
djinn.bat              # CLI launcher
start_all.bat          # Launch all services
djinn-train.bat        # Training launcher
```

### Database & Setup
```bash
# 🗄️ Initialize memory database
python init_database.py

# 📦 Install dependencies
pip install -r requirements.txt
```

---

## 💬 Enhanced CLI Features

**Improved input handling for better user experience.**

### ✨ Key Features
- **Flexible Input**: Support for both single-line and multi-line text
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Mode Switching**: Easy switching between agent modes
- **Session Management**: Persistent conversation contexts

### 🎯 Usage Examples
```bash
# Start enhanced CLI
python cli_control.py

# Single line input
> Hello, how are you?

# Multi-line input (paste directly)
> You are an expert software architect.
> Please analyze this codebase...

# Mode switching
> !lm (language model mode)
> !reflect (reflection mode)
```

### ⌨️ Interactive Commands
```bash
# Start interactive mode
python cli_control.py

# Available during interaction
exit, quit              # Exit the CLI
!lm                    # Switch to language model mode
!reflect               # Switch to reflection mode
```

---

## 🌐 API Endpoints

### Core Endpoints
```http
POST /chat              # Send message, get response
GET  /status            # Server and agent health
POST /feedback          # Submit human feedback
GET  /introspect        # Query agent introspection
POST /task              # Spawn new tasks
```

### Memory & Query
```http
GET  /memory/query      # Query persistent memory
POST /memory/summarize  # Summarize conversation history
GET  /memory/stats      # Memory usage statistics
```

### Advanced Endpoints
```http
POST /quorum/test       # Test agent consensus
POST /checkpoint        # Manage model checkpoints
GET  /agent/status      # Individual agent status
```

---

## 🎯 CLI Command Reference

### Interactive Mode
```bash
# Start natural language interface
python cli_control.py

# With specific agent
python cli_control.py --agent A

# With specific mode
python cli_control.py --mode reflect
```

### Direct Commands
```bash
# Chat with specific parameters
python cli_control.py chat --text "Hello world" --mode lm --session mychat

# Query memory
python cli_control.py query --text "What was my last decision?"

# Check agent status
python cli_control.py status --agent A

# Spawn training task
python cli_control.py task --task_type fine_tune --objective "Improve response quality"

# Manage checkpoints
python cli_control.py checkpoint --op promote
python cli_control.py checkpoint-pull-if-newer
```

### Batch Operations
```bash
# Target all agents
python cli_control.py status --agent ALL
python cli_control.py start --agent ALL
python cli_control.py stop --agent ALL
```

---

## 🧠 Agent Modes & Features

### Interaction Modes
- **🔍 `reflect`**: Memory-based reasoning (default)
- **🤖 `lm`**: Direct language model generation
- **🔄 Auto-switching**: Agent can switch modes based on context

### Self-Reflection Features
- **Decision Logging**: All major decisions are recorded
- **Performance Tracking**: Monitors response quality and patterns
- **Memory Summarization**: Compresses conversation history
- **Task Spawning**: Creates subtasks for complex queries

### Human Feedback Integration
- **Real-time Feedback**: Submit feedback during conversations
- **Quality Assessment**: Rate response helpfulness
- **Learning Loop**: Agent improves based on feedback

---

## 🧪 Testing & Evaluation

### Test Suites
```bash
# Phase 3 comprehensive test
python test_phase3_complete.py

# API functionality test
python test_phase4_api.py

# Dialogue system test
python test_dialogue.py
```

### Evaluation Tools
```bash
# Canary prompt evaluation
python canary_eval.py

# Custom evaluation
python eval/custom_eval.py
```

### Performance Metrics
- **Response Quality**: Coherence and relevance scores
- **Memory Efficiency**: Storage and retrieval performance
- **Self-Reflection**: Introspection accuracy
- **Task Completion**: Success rates for spawned tasks

---

## 🎓 Training & Model Development

### Training Scripts
```bash
# Tiny language model
python train_tiny_lm.py

# Conversational LM (Windows)
python train_conversational_lm_windows.py

# Minimal transformer (Windows)
python train_minimal_transformer_windows.py
```

### Model Management
```bash
# Download models (GGUF format recommended)
# Place in project directory (auto-detected)

# Checkpoints are automatically managed
# Use .gitignore to exclude large model files
```

### Curriculum Learning
```bash
# Use curriculum dataset
python curriculum_dataset.py

# Custom training data
python toy_dataset.py
```

---

## 🔧 Configuration & Environment

### Environment Variables
```bash
# API Authentication (optional)
export RA_AUTH_SECRET="your-secret-key"

# Custom model paths (optional)
export MODEL_PATH="/path/to/models"

# Database path (optional)
export DB_PATH="./agent_A.db"
```

### Configuration Files
- **`.env`**: Environment variables (copy from `.env.example`)
- **`.gitignore`**: Excludes models, checkpoints, cache
- **`requirements.txt`**: Python dependencies with versions

---

## 🚨 Troubleshooting

### Common Issues
```bash
# Database not initialized
python init_database.py

# Dependencies missing
pip install -r requirements.txt

# Model files not found
# Download GGUF models and place in directory

# Port conflicts
python server_main.py --port 8081
```

### Debug Mode
```bash
# Verbose logging
python cli_control.py --debug

# API with debug
python server_main.py --debug
```

---

## 📊 System Monitoring

### Health Checks
```bash
# Agent status
curl http://localhost:8081/agent/status

# Memory usage
curl http://localhost:8081/memory/stats

# API health
curl http://localhost:8081/status
```

### Performance Metrics
- **Memory Usage**: SQLite database size and query performance
- **Response Times**: API endpoint latency
- **Model Performance**: Inference speed and quality metrics
- **Resource Usage**: CPU, memory, and disk utilization

---

## 🎨 Advanced Features

### Multi-Agent Coordination
- **Phase 5 Architecture**: Communication between agent instances
- **Consensus Mechanisms**: Decision validation across agents
- **Shared Memory**: Distributed memory systems
- **Service Registry**: Dynamic service discovery

### Guardian System
- **Input Validation**: `guardian/validator.py`
- **Safety Checks**: Adversarial input detection
- **Output Filtering**: Response quality assurance

### Custom Extensions
- **Plugin Architecture**: Extensible component system
- **Custom Tasks**: Define new task types
- **Model Integration**: Support for additional model formats

---

## 📚 Documentation Map

| Document | Description |
|----------|-------------|
| [`README.md`](../README.md) | Project overview and setup |
| [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) | Complete file organization |
| [`design_doc.md`](../design_doc.md) | Architecture and principles |
| [`arch_blueprint.md`](../arch_blueprint.md) | Technical diagrams |
| [`phase*_verification.md`](../phase0_verification.md) | Phase verification |

---

## 🚀 Pro Tips

- **Start Simple**: Use `python cli_control.py` for first interactions
- **Multi-Line Magic**: Paste system prompts directly - no special syntax!
- **Mode Switching**: Use `!lm` and `!reflect` during conversations
- **Memory Queries**: Ask "What was my last decision?" for introspection
- **Feedback Loop**: Rate responses to improve agent performance
- **Batch Operations**: Use `--agent ALL` for multi-agent commands

---

**Ready to explore self-reflective AI?** Start with `python cli_control.py` and paste your first prompt! 🤖✨
