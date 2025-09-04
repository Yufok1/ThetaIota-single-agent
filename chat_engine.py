#!/usr/bin/env python3
"""
Enhanced chat engine for the Phase 4/5 agent with MinimalTransformer integration.

Design goals:
- Integrate trained MinimalTransformer for reasoning guidance
- Use LLM for text generation with transformer-guided prompts
- Maintain ThetaIota's self-reflective architecture
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Deque, Tuple
from collections import deque

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None  # type: ignore

# Import our custom transformer
try:
    from transformer_model import MinimalTransformer
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False

import math
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Deque, Tuple
from collections import deque

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None  # type: ignore

# Import our custom transformer
try:
    from transformer_model import MinimalTransformer
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False


# -------------------- Tokenizer --------------------

class ByteTokenizer:
    """Simple byte-level tokenizer (UTF-8) with BOS/EOS specials.
    Vocab: 0..255 (bytes) + 256=BOS + 257=EOS
    """

    def __init__(self):
        self.BOS = 256
        self.EOS = 257
        self.vocab_size = 258

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        b = text.encode("utf-8", errors="ignore")
        ids = list(b)
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        # strip specials
        filtered = [i for i in ids if i < 256]
        return bytes(filtered).decode("utf-8", errors="ignore")


# -------------------- Tiny Causal LM --------------------

if TORCH_AVAILABLE:
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1):
            super().__init__()
            self.scale = math.sqrt(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D]
            q = self.q(x); k = self.k(x); v = self.v(x)
            att = (q @ k.transpose(-2, -1)) / self.scale
            # causal mask: allow j <= i
            L = x.size(1)
            mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0)
            att = att.masked_fill(mask == 0, float('-inf'))
            w = torch.softmax(att, dim=-1)
            w = self.dropout(w)
            out = w @ v
            return self.o(out)

    class TinyCausalLM(nn.Module):
        def __init__(self, vocab_size: int = 258, d_model: int = 1024, d_ff: int = 4096, max_len: int = 512, dropout: float = 0.1, n_layers: int = 12):
            super().__init__()
            self.max_len = max_len
            self.n_layers = n_layers
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            
            # Multiple transformer layers for better capacity
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'ln1': nn.LayerNorm(d_model),
                    'attn': CausalSelfAttention(d_model, dropout),
                    'ln2': nn.LayerNorm(d_model),
                    'ff': nn.Sequential(
                        nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
                    )
                }) for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
            self.drop = nn.Dropout(dropout)
            self._init()

        def _init(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            # idx: [B, L]
            B, L = idx.shape
            if L > self.max_len:
                idx = idx[:, -self.max_len:]
                L = self.max_len
            pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
            x = self.drop(self.tok(idx) + self.pos(pos))
            
            # Pass through multiple transformer layers
            for layer in self.layers:
                x = layer['ln1'](x + layer['attn'](x))
                x = layer['ln2'](x + layer['ff'](x))
            
            x = self.ln_final(x)
            logits = self.head(x)
            return logits

        @torch.no_grad()
        def generate(self, idx: torch.Tensor, max_new_tokens: int = 1024, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
            self.eval()
            for _ in range(max_new_tokens):
                logits = self(idx)  # [B,L,V]
                logits = logits[:, -1, :]
                if temperature and temperature > 0.0 and temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                if top_k and top_k > 0:
                    k = min(top_k, probs.size(-1))
                    vals, inds = torch.topk(probs, k=k, dim=-1)
                    vals = vals / vals.sum(dim=-1, keepdim=True)
                    cat = torch.distributions.Categorical(vals)
                    choice = cat.sample().unsqueeze(-1)
                    nxt = inds.gather(-1, choice)
                else:
                    nxt = probs.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
                # stop early if EOS
                if int(nxt[0, 0].item()) == 257:
                    break
                if idx.size(1) > self.max_len:
                    idx = idx[:, -self.max_len:]
            return idx


@dataclass
class ChatConfig:
    mode: str = "lm"  # Only LLM mode
    max_new_tokens: int = 1024  # Increased from 64 to allow full responses
    device: str = "cpu"
    history_max_turns: int = 10
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    d_model: int = 1024
    d_ff: int = 4096
    lm_max_len: int = 2048  # Increased from 512 for longer context
    n_layers: int = 12


class ChatEngine:
    def chat_reasoning_loop(self, text: str, mode: Optional[str] = None, session: Optional[str] = None, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Enhanced multi-round reasoning loop with MinimalTransformer guidance:
        1. Use trained transformer to analyze conversation context
        2. Build enhanced prompts with transformer insights
        3. Get LLM responses guided by transformer reasoning
        4. Meta-controller evaluates response completeness
        5. Iterate until optimal response achieved
        """
        session_id = session or "default"
        self._append_history(session_id, "user", text)
        previous_reasoning = []
        meta_feedback = None

        # Get conversation history for transformer analysis
        conversation_context = self._get_conversation_context(session_id, text)

        # Use transformer for initial reasoning strategy analysis
        transformer_guidance = self._get_transformer_reasoning_guidance(conversation_context)

        print(f"🎯 Transformer Reasoning Strategy: {transformer_guidance['strategy']} "
              f"(confidence: {transformer_guidance['confidence']:.2f})")

        for round_num in range(max_rounds):
            # Build enhanced prompt with transformer guidance
            prompt_lines = []
            prompt_lines.append("SYSTEM: You are ThetaIota, a self-reflective AI agent with advanced cognitive capabilities.")

            # Add transformer reasoning insights
            if transformer_guidance['insights']:
                prompt_lines.append(f"Reasoning Strategy: {transformer_guidance['strategy'].upper()}")
                prompt_lines.append("Key Insights:")
                for insight in transformer_guidance['insights'][:3]:  # Limit to top 3
                    prompt_lines.append(f"- {insight}")
                prompt_lines.append("")

            prompt_lines.append("Stay focused on the current conversation context. Avoid repeating previous reasoning.")

            if self.agent_ref is not None:
                try:
                    result = self.agent_ref.explainer.query_agent_memory(text)
                    human_summary = (result.get("summary") or "").strip() or None
                    if human_summary:
                        prompt_lines.append(f"Memory summary: {human_summary}")
                except Exception:
                    pass
                try:
                    mem = self.agent_ref.memory
                    cursor = mem.conn.cursor()
                    cursor.execute("""
                        SELECT * FROM meta_events 
                        WHERE event_type = 'meta_decision' 
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if row:
                        info = json.loads(row[3]) if isinstance(row[3], str) else {}
                        last_decision = str(info.get('action', None)) or None
                        if last_decision:
                            prompt_lines.append(f"Last decision: {last_decision}.")
                except Exception:
                    pass
            if meta_feedback:
                prompt_lines.append(f"Meta feedback: {meta_feedback}")

            # Strategy-specific prompt guidance
            if transformer_guidance['strategy'] == 'analytical':
                if previous_reasoning:
                    prompt_lines.append("Previous response:")
                    prompt_lines.append(previous_reasoning[-1] if previous_reasoning else "")
                    prompt_lines.append("---")
                    prompt_lines.append("ANALYZE further: Break down the problem into logical components. Use evidence-based reasoning. Identify cause-effect relationships.")
                else:
                    prompt_lines.append("Provide a comprehensive answer using step-by-step analytical reasoning and evidence from your memory.")
            elif transformer_guidance['strategy'] == 'creative':
                if previous_reasoning:
                    prompt_lines.append("Previous response:")
                    prompt_lines.append(previous_reasoning[-1] if previous_reasoning else "")
                    prompt_lines.append("---")
                    prompt_lines.append("EXPLORE creatively: Consider unconventional approaches. Find novel connections. Think beyond traditional boundaries.")
                else:
                    prompt_lines.append("Provide a comprehensive answer using creative thinking and innovative approaches from your memory.")
            else:
                if previous_reasoning:
                    prompt_lines.append("Previous response:")
                    prompt_lines.append(previous_reasoning[-1] if previous_reasoning else "")
                    prompt_lines.append("---")
                    prompt_lines.append("Expand your previous answer with NEW information. Do NOT repeat what you already said. Add concrete details, examples, or next steps.")
                else:
                    prompt_lines.append("Provide a comprehensive answer using your memory, database, and recent decisions.")

            prompt_lines.append(f"User: {text}")
            prompt_lines.append("Assistant:")
            prompt = "\n".join(prompt_lines)
            print(f"DEBUG: Enhanced reasoning loop round {round_num+1} prompt:\n{prompt}")

            # Generate response with transformer-guided LLM
            if self.llama_model is not None:
                # Adjust temperature based on transformer confidence
                temperature = 0.7 if transformer_guidance['confidence'] > 0.7 else 0.5

                output = self.llama_model(
                    prompt,
                    max_tokens=1024,
                    temperature=temperature,
                    top_p=0.92,
                    stop=["User:", "Assistant:", "==="]
                )
                resp = output["choices"][0]["text"].strip()
            else:
                resp = "(no GGUF model available)"
            self._append_history(session_id, "assistant", resp)

            # Log round in memory with transformer insights
            try:
                if self.agent_ref is not None and hasattr(self.agent_ref, 'memory'):
                    self.agent_ref.memory.log_chat_message(
                        session_id=session_id,
                        role="assistant",
                        text=resp,
                        mode="enhanced_reasoning_loop",
                        metadata={
                            "transformer_strategy": transformer_guidance['strategy'],
                            "transformer_confidence": transformer_guidance['confidence'],
                            "round": round_num + 1
                        }
                    )
            except Exception:
                pass

            previous_reasoning.append(resp)

            # Meta-controller evaluates response with transformer context
            complete = False
            if self.agent_ref is not None and hasattr(self.agent_ref, 'meta_controller'):
                try:
                    # Pass transformer insights to meta-controller for better evaluation
                    complete, meta_feedback = self.agent_ref.meta_controller.evaluate_response_with_context(
                        resp, transformer_guidance
                    )
                except Exception:
                    # Fallback to regular evaluation
                    complete, meta_feedback = self.agent_ref.meta_controller.evaluate_response(resp)
            else:
                # Simple heuristic: stop if response contains completion indicators
                if "Final answer:" in resp or len(resp.strip()) > 300:
                    complete = True
            if complete:
                break
        return {"reply": resp, "mode": "enhanced_reasoning_loop", "rounds": round_num+1, "transformer_strategy": transformer_guidance['strategy']}

    def _get_conversation_context(self, session_id: str, current_text: str) -> str:
        """Build conversation context for transformer analysis."""
        context_parts = []

        # Add recent conversation history
        history = self.sessions.get(session_id, [])
        recent_messages = list(history)[-5:]  # Last 5 exchanges

        for role, msg in recent_messages:
            context_parts.append(f"{role}: {msg}")

        # Add current user input
        context_parts.append(f"user: {current_text}")

        # Add agent memory context if available
        if self.agent_ref is not None:
            try:
                result = self.agent_ref.explainer.query_agent_memory(current_text)
                human_summary = (result.get("summary") or "").strip() or None
                if human_summary:
                    context_parts.append(f"memory: {human_summary}")
            except Exception:
                pass

        return " | ".join(context_parts)
    def __init__(self, agent_ref=None, config: Optional[ChatConfig] = None):
        self.agent_ref = agent_ref
        self.config = config or ChatConfig()
        self.tokenizer = ByteTokenizer()
        self.lm = None
        self.lm_tokenizer = None
        # session_id -> deque of (role, text)
        self.sessions = {}

        # Load trained MinimalTransformer for reasoning guidance
        self.reasoning_transformer = None
        self.transformer_tokenizer = None
        if TRANSFORMER_AVAILABLE and TORCH_AVAILABLE:
            self._load_reasoning_transformer()

        # Use llama-cpp-python for GGUF models only
        try:
            from llama_cpp import Llama
            gguf_model_path = r"D:\end-GAME\HF-models\llama3.2\Llama-3.2-3B-Instruct-Q4_0.gguf"
            self.llama_model = Llama(model_path=gguf_model_path, n_ctx=4096)  # Increased context window
        except Exception as e:
            print(f"Failed to load GGUF model: {e}")
            self.llama_model = None

    def _load_reasoning_transformer(self):
        """Load the trained MinimalTransformer for reasoning guidance."""
        try:
            # Try to load the final trained model
            checkpoint_path = os.path.join('checkpoints', 'minimal_transformer_final.pt')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join('checkpoints', 'minimal_transformer.pt')

            if os.path.exists(checkpoint_path):
                print(f"Loading reasoning transformer from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract model state
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    model_state = checkpoint

                # Get model dimensions from state
                if 'token_embedding.weight' in model_state:
                    vocab_size, d_model = model_state['token_embedding.weight'].shape
                    max_seq_len = model_state['position_embedding.weight'].shape[0]

                    # Create model with correct dimensions
                    self.reasoning_transformer = MinimalTransformer(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        d_ff=d_model * 4,  # Assume 4x expansion
                        max_seq_len=max_seq_len,
                        num_classes=2,
                        n_layers=6  # Assume 6 layers based on training
                    )

                    # Load the trained weights
                    self.reasoning_transformer.load_state_dict(model_state)
                    self.reasoning_transformer.eval()

                    # Create a simple tokenizer for the transformer
                    self.transformer_tokenizer = self._create_transformer_tokenizer(vocab_size)

                    print(f"✅ Reasoning transformer loaded: {vocab_size:,} vocab, {d_model}d model")
                else:
                    print("❌ Could not determine model dimensions from checkpoint")
            else:
                print("❌ No trained transformer checkpoint found")

        except Exception as e:
            print(f"❌ Failed to load reasoning transformer: {e}")
            self.reasoning_transformer = None

    def _create_transformer_tokenizer(self, vocab_size: int):
        """Create a simple tokenizer for the reasoning transformer."""
        class SimpleTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def encode(self, text: str) -> List[int]:
                # Simple character-level encoding
                tokens = []
                for char in text[:512]:  # Limit length
                    # Map characters to token IDs (0-255 for ASCII, then hash the rest)
                    if ord(char) < 256:
                        tokens.append(ord(char))
                    else:
                        tokens.append(hash(char) % (self.vocab_size - 256) + 256)
                return tokens[:512]  # Ensure max length

            def decode(self, tokens: List[int]) -> str:
                # Simple decoding
                chars = []
                for token in tokens:
                    if token < 256:
                        chars.append(chr(token))
                    else:
                        chars.append('?')  # Unknown character
                return ''.join(chars)

        return SimpleTokenizer(vocab_size)

    def _get_transformer_reasoning_guidance(self, conversation_context: str) -> Dict[str, Any]:
        """Use the trained transformer to analyze conversation and provide reasoning guidance."""
        if self.reasoning_transformer is None or self.transformer_tokenizer is None:
            return {"strategy": "default", "confidence": 0.5, "insights": []}

        try:
            # Tokenize the conversation context
            tokens = self.transformer_tokenizer.encode(conversation_context)
            if len(tokens) == 0:
                return {"strategy": "default", "confidence": 0.5, "insights": []}

            # Convert to tensor
            input_tensor = torch.tensor([tokens], dtype=torch.long)

            # Get transformer predictions
            with torch.no_grad():
                logits = self.reasoning_transformer(input_tensor)
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # Use last token predictions
                confidence = probs.max().item()
                prediction = probs.argmax().item()

            # Interpret prediction as reasoning strategy
            strategies = {
                0: "analytical",  # Break down problem step-by-step
                1: "creative",    # Think outside the box
            }

            strategy = strategies.get(prediction, "balanced")
            insights = []

            # Add strategy-specific insights
            if strategy == "analytical":
                insights = [
                    "Break down the problem into smaller components",
                    "Use logical reasoning and evidence",
                    "Consider cause and effect relationships"
                ]
            elif strategy == "creative":
                insights = [
                    "Explore unconventional approaches",
                    "Consider multiple perspectives",
                    "Look for patterns and connections"
                ]
            else:
                insights = [
                    "Balance analytical and creative thinking",
                    "Draw from both logic and intuition",
                    "Consider practical implications"
                ]

            return {
                "strategy": strategy,
                "confidence": confidence,
                "insights": insights
            }

        except Exception as e:
            print(f"Transformer reasoning guidance failed: {e}")
            return {"strategy": "default", "confidence": 0.5, "insights": []}

    def attach_agent(self, agent_obj):
        self.agent_ref = agent_obj

    # No-op: LM weights are loaded from Hugging Face

    def chat(self, text: str, mode: Optional[str] = None, session: Optional[str] = None) -> Dict[str, Any]:
        """Main chat method with mode selection: 'reflect' (default) or 'lm'."""
        session_id = session or "default"
        self._append_history(session_id, "user", text)

        # Mode selection: default to 'reflect' if not specified
        if mode is None:
            mode = "reflect"

        if mode == "lm":
            # Direct LLM mode - use PyTorch transformer
            if self.lm is None or not TORCH_AVAILABLE:
                return {"reply": "LM mode not available - no PyTorch model loaded", "mode": "lm"}
            return self._chat_lm(text, session_id)
        else:
            # Reflective mode (default) - use reasoning loop with ThetaIota context
            return self.chat_reasoning_loop(text, mode=mode, session=session)

    def _chat_lm(self, text: str, session_id: str) -> Dict[str, Any]:
        """Direct LLM mode using GGUF model via llama-cpp-python."""
        if self.llama_model is None:
            return {"reply": "LM mode not available - GGUF model not loaded", "mode": "lm"}

        # Build rolling prompt from session history
        history = self.sessions.get(session_id)
        prompt_lines: List[str] = []
        if history:
            for role, msg in list(history)[-self.config.history_max_turns:]:
                if role == "user":
                    prompt_lines.append(f"User: {msg}")
                else:
                    prompt_lines.append(f"Assistant: {msg}")
        prompt_lines.append(f"User: {text}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)

        # Generate response using GGUF model
        try:
            output = self.llama_model(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=0.92,
                stop=["User:", "Assistant:", "==="]
            )
            resp = output["choices"][0]["text"].strip()
        except Exception as e:
            resp = f"Error generating response: {e}"

        if not resp:
            resp = "(thinking…)"

        self._append_history(session_id, "assistant", resp)
        return {"reply": resp, "mode": "lm"}

    def _append_history(self, session_id: str, role: str, text: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.config.history_max_turns * 2)
        self.sessions[session_id].append((role, text))
        try:
            if self.agent_ref is not None and hasattr(self.agent_ref, 'memory'):
                self.agent_ref.memory.log_chat_message(session_id=session_id, role=role, text=text, mode=None)
        except Exception:
            pass


