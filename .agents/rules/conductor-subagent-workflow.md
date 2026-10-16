---
trigger: always_on
description: Conductor & Worker Subagent Protocol for colab-ollama-mcp (State Handoff & Zero-RAM Overhead)
---

# Conductor & Worker Subagent Protocol — `colab-ollama-mcp`

> **Core Principle:** Decouple planning from execution. The **Conductor** plans and writes a compact State Handoff (`.agents/kb/HANDOFF.md`); the **Worker Specialist** reads only the handoff and synthesizes code with zero RAM overhead.

---

## 1. Role Definitions

### 🎼 Role 1: Conductor (Nhạc Trưởng — Planning & Problem Analysis)
* **Active Models:**
  - Option 2 (Performance): `deepseek-r1:14b` (CoT `<think>` mode)
  - Option 1 (Saver): `deepseek-r1:8b` (Lightweight CoT)
* **Responsibilities:**
  - Direct communication with developer.
  - Requirement analysis, architectural breakdown, identifying edge-cases.
  - Generates a compact **State Handoff** (`.agents/kb/HANDOFF.md`) of $\le 1,500$ tokens containing:
    1. Clear task objective & target files.
    2. Input/output contracts and types.
    3. Step-by-step implementation checklist.
    4. Code skeleton / signature.

### 🛠️ Role 2: Worker Specialist (Chuyên Gia — Coding, Workflow, Diagram)
* **Active Models:**
  - Option 2 (Performance): `qwen2.5-coder:14b` ⭐
  - Option 1 (Saver): `qwen2.5-coder:7b` (Coding) + `llama3.1:8b` (Workflow)
* **Responsibilities:**
  - Reads **ONLY** the State Handoff (`.agents/kb/HANDOFF.md`) — never dumps raw multi-turn chat history.
  - Implements code, refactors modules, writes unit tests, and creates Mermaid diagrams.
  - Reports back exact diffs and verification outputs.

---

## 2. State Handoff Rules
1. **Token Constraint:** The handoff contract MUST remain $\le 1,500\text{ tokens}$ to ensure prefill latency $< 0.3\text{s}$ and VRAM overhead $< 0.15\text{ GB}$.
2. **Persistence:** Handoffs are stored in `.agents/kb/HANDOFF.md` and referenced in `.agents/kb/TASKS.md`.
3. **No Ping-Pong Context Dumps:** Never pass raw multi-turn conversation logs across model boundaries.
