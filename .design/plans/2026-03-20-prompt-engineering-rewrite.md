# Prompt Engineering Rewrite Implementation Plan

**Goal:** Clean-rewrite all agent prompts to follow XML-delimited prompt engineering standards with few-shot examples, positive framing, and explicit output format specs.

**Architecture:** 7 prompt files (6 rewrite + 1 new), 1 agent file modified, 1 test file updated. Each prompt follows the XML skeleton: `<role>` → `<instructions>` → `<examples>` → `<output_format>` → `<recap>`. System prompts hold static content; user prompts hold dynamic content. Critic prompts extracted from `agents/critic.py` to `prompts/critic.py`, refinement prompt removed (stateless round 2+).

**Tech Stack:** Python, pytest, XML-delimited prompt strings

---

### Task 1: Ingestor Prompt

**Files:**
- Rewrite: `src/auto_scientist/prompts/ingestor.py`

**What:** Rewrite with XML tags, positive framing, and explicit scope boundary (data plumbing only, no scientific analysis or hypotheses in notebook entries). This is the simplest prompt (no examples, no recap) so it's a good first file to establish the pattern.

- [ ] **Step 1: Rewrite `ingestor.py`**
  - XML skeleton: `<role>`, `<instructions>`, `<output_format>`, `<task>`
  - Add `<scope_boundary>` section with concrete examples of what to include vs. exclude in notebook entries (per spec section 6)
  - Positive framing: replace all "NEVER modify" with "preserve original files" etc.
  - Keep same template variables: `{raw_data_path}`, `{goal}`, `{mode}`, `{data_dir}`, `{notebook_path}`

- [ ] **Step 2: Run tests, verify no import breakage**
  - `uv run pytest tests/ -x`

- [ ] **Step 3: Commit** — "refactor: rewrite Ingestor prompt with XML structure and scope boundary"

---

### Task 2: Discovery Prompt

**Files:**
- Rewrite: `src/auto_scientist/prompts/discovery.py`

**What:** XML rewrite. Discovery is also tool-using with no examples. Keep same template variables. Ensure role clearly distinguishes from Ingestor (Discovery does the scientific exploration).

- [ ] **Step 1: Rewrite `discovery.py`**
  - XML skeleton, positive framing
  - Keep same template variables: `{data_path}`, `{goal}`, `{domain_knowledge}`, `{notebook_path}`, `{config_path}`
  - Domain config JSON schema stays in the prompt (it's the output format spec)

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit** — "refactor: rewrite Discovery prompt with XML structure"

---

### Task 3: Coder Prompt

**Files:**
- Rewrite: `src/auto_scientist/prompts/coder.py`

**What:** XML rewrite. Add motivation for key rules (self-contained scripts = reproducibility, SUCCESS CRITERIA in code = honest evaluation). Keep CODER_NO_PREVIOUS and CODER_HAS_PREVIOUS helper strings.

- [ ] **Step 1: Rewrite `coder.py`**
  - XML skeleton with `<role>`, `<instructions>`, `<output_format>`
  - Add `<motivation>` for non-obvious rules
  - SUCCESS CRITERIA stdout format stays as concrete template in `<output_format>`
  - Keep same template variables: `{experiment_dependencies}`, `{data_path}`, `{domain_knowledge}`, `{plan_json}`, `{previous_script_section}`, `{new_script_path}`, `{version}`

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit** — "refactor: rewrite Coder prompt with XML structure and rule motivations"

---

### Task 4: Report Prompt

**Files:**
- Rewrite: `src/auto_scientist/prompts/report.py`

**What:** XML rewrite. Tighten vague constraints ("be honest" → specific instructions). Keep the 10-section report structure.

- [ ] **Step 1: Rewrite `report.py`**
  - XML skeleton, positive framing, concrete constraints
  - Keep same template variables: `{domain}`, `{goal}`, `{total_iterations}`, `{best_version}`, `{best_score}`, `{notebook_content}`, `{report_path}`

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit** — "refactor: rewrite Report prompt with XML structure"

---

### Task 5: Analyst Prompt (with examples)

**Files:**
- Rewrite: `src/auto_scientist/prompts/analyst.py`

**What:** First prompt with few-shot examples. 3 examples using water quality monitoring domain. Full JSON schema + example output + fallback rules.

- [ ] **Step 1: Rewrite `analyst.py`**
  - XML skeleton with all sections including `<examples>` and `<recap>`
  - 3 examples per spec: normal case, null/empty case (first iteration, crash), most typical (last)
  - `<output_format>`: verbal description, JSON schema with types, example JSON, fallback rules
  - `<recap>`: "Report only what you observe. Every claim references a specific number. Valid JSON with all required keys."
  - Keep same template variables: `{domain_knowledge}`, `{success_criteria}`, `{results_content}`, `{notebook_content}`, `{plot_list}`

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Commit** — "refactor: rewrite Analyst prompt with XML structure, examples, and output schema"

---

### Task 6: Scientist Prompt (with examples)

**Files:**
- Rewrite: `src/auto_scientist/prompts/scientist.py`

**What:** Largest rewrite. 5 examples covering all strategy types + stop + v00. Full JSON schema. Also rewrite SCIENTIST_REVISION_SYSTEM/USER (2 examples). Use different domains per example for breadth.

- [ ] **Step 1: Rewrite SCIENTIST_SYSTEM and SCIENTIST_USER**
  - XML skeleton with `<examples>` (5), `<output_format>`, `<recap>`
  - Examples: incremental (crop yield), structural (traffic flow), exploratory (weather), v00 (first iteration), should_stop (most typical, last)
  - JSON schema with enums: strategy in {incremental, structural, exploratory}, priority in {1, 2, 3}
  - Fallback rules for first iteration, missing domain knowledge, script crash
  - Keep same template variables: `{domain_knowledge}`, `{analysis_json}`, `{notebook_content}`, `{version}`

- [ ] **Step 2: Rewrite SCIENTIST_REVISION_SYSTEM and SCIENTIST_REVISION_USER**
  - 2 examples: accepting critique, mostly rejecting critique
  - Same JSON schema as Scientist
  - Fallback rules for empty debate, criteria-only debate
  - Keep same template variables: `{domain_knowledge}`, `{analysis_json}`, `{notebook_content}`, `{original_plan}`, `{debate_transcript}`, `{version}`

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit** — "refactor: rewrite Scientist and Revision prompts with XML structure, examples, and output schema"

---

### Task 7: Critic Prompts (new file + agent refactor)

**Files:**
- Create: `src/auto_scientist/prompts/critic.py`
- Modify: `src/auto_scientist/agents/critic.py`

**What:** Extract critic and scientist-debate prompts to `prompts/critic.py`. Remove `_build_critic_refinement_prompt`. CRITIC_USER has an optional `{scientist_defense}` section for round 2+ (stateless design). Update `agents/critic.py` to import templates and format them.

- [ ] **Step 1: Write failing tests for new prompt structure**
  - Test: CRITIC_USER.format() with and without defense produces valid prompts
  - Test: plan content appears in formatted critic prompt
  - Test: notebook content appears in both critic and scientist debate prompts

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Create `prompts/critic.py`**
  - CRITIC_SYSTEM, CRITIC_USER (with optional `{scientist_defense}` section)
  - SCIENTIST_DEBATE_SYSTEM, SCIENTIST_DEBATE_USER
  - XML skeleton, positive framing

- [ ] **Step 4: Refactor `agents/critic.py`**
  - Import templates from `prompts/critic.py`
  - Replace `_build_critic_prompt` → format CRITIC templates (round 1: empty defense, round 2+: defense included)
  - Replace `_build_scientist_response_prompt` → format SCIENTIST_DEBATE templates
  - Remove `_build_critic_refinement_prompt` entirely
  - Update `run_debate` loop to use new function signatures

- [ ] **Step 5: Run tests, verify they pass**

- [ ] **Step 6: Commit** — "refactor: extract critic prompts to prompts/critic.py, remove refinement prompt"

---

### Task 8: Update Critic Tests

**Files:**
- Modify: `tests/test_critic.py`

**What:** Update string assertions that check for markdown headings (e.g., `"Scientist's Plan"`, `"Your Plan"`, `"Lab Notebook"`) to match new XML tag names. Preserve the information boundary checks.

- [ ] **Step 1: Update heading assertions**
  - `"Scientist's Plan"` → whatever XML tag wraps the plan in CRITIC_USER
  - `"Your Plan"` → whatever XML tag wraps the plan in SCIENTIST_DEBATE_USER
  - `"Lab Notebook"` → whatever XML tag wraps notebook in both
  - Keep `"success_criteria"` (JSON content, not a heading)
  - Keep all "not in" assertions (no analysis, no script, no compressed history)

- [ ] **Step 2: Run full test suite**
  - `uv run pytest tests/ -v`

- [ ] **Step 3: Commit** — "test: update critic test assertions for XML prompt structure"

---

### Task 9: Final Validation

- [ ] **Step 1: Run full test suite** — `uv run pytest tests/ -v`
- [ ] **Step 2: Run linter** — `uv run ruff check src/auto_scientist/prompts/ tests/test_critic.py`
- [ ] **Step 3: Spot-check each prompt file** — verify XML structure, no markdown delimiters mixed with XML, examples present where specified
- [ ] **Step 4: Commit any final fixes**
