# Toy Function Discovery - Implementation Plan

**Goal:** Generate a synthetic dataset and run the full auto-scientist pipeline in auto-discovery mode (no pre-loaded domain) to validate the methodology end-to-end.

**Architecture:** Generate a CSV with known ground truth, then let the framework discover everything: success criteria, domain knowledge, initial approach, and iterative improvements. No domain config files needed.

**Tech Stack:** numpy (data generation), existing auto-scientist framework

---

### Task 1: Generate Synthetic Dataset

**Files:**
- Create: `domains/toy_function/seed/generate_data.py`
- Create: `domains/toy_function/seed/data/toy_function.csv`

**What:** Standalone script that generates a reproducible CSV. `y = 2.5*sin(1.5x) + 0.3x^2 + noise(σ=0.5)`, 200 points, x in [-5, 5], seed 42.

- [ ] **Step 1: Write generate_data.py, run it, verify CSV shape and values**
- [ ] **Step 2: Commit** - "feat: add toy function synthetic dataset"

---

### Task 2: Verify Model Selection

**What:** Confirm `claude_agent_sdk` can be configured to use Sonnet instead of defaulting to Opus. Check for `ANTHROPIC_MODEL` env var or `model` param in `ClaudeAgentOptions`.

- [ ] **Step 1: Check SDK source/docs for model configuration**
- [ ] **Step 2: If not supported via env var, add `--model` CLI flag and thread through**
- [ ] **Step 3: Commit if changes needed**

---

### Task 3: Run the Pipeline

**What:** Run the full auto-discovery pipeline. No `--domain` flag, just data + goal. The framework should generate its own config, criteria, and scripts.

- [ ] **Step 1: Smoke test with 1 iteration, no critics**
  ```bash
  ANTHROPIC_MODEL=claude-sonnet-4-6 uv run auto-scientist run \
      --data domains/toy_function/seed/data/toy_function.csv \
      --goal "Discover the mathematical function that generated this data. Find the best f(x) that minimizes prediction error." \
      --max-iterations 1 \
      --output-dir experiments/toy_function
  ```
  - Verify: Ingestor canonicalizes CSV, Discovery generates config + v00, one iteration runs

- [ ] **Step 2: Full run with 3 iterations + critic**
  ```bash
  ANTHROPIC_MODEL=claude-sonnet-4-6 uv run auto-scientist run \
      --data domains/toy_function/seed/data/toy_function.csv \
      --goal "Discover the mathematical function that generated this data. Find the best f(x) that minimizes prediction error." \
      --max-iterations 3 \
      --critics "openai:gpt-5.4-mini" \
      --output-dir experiments/toy_function_full
  ```
  - Verify: debate transcripts, RMSE improvement across iterations, state persistence

- [ ] **Step 3: Review results** - notebook progression, state.json, generated scripts
