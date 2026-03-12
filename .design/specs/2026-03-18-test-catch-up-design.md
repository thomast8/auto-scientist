# Test Catch-Up: Closing Coverage Gaps

## Problem

5 of 22 source modules have tests. The tested modules (runner, scheduler, state, critic, ingestor) are pure-logic or well-isolated. The untested modules cover the agent layer, orchestrator, model clients, config, and CLI, which is the majority of the codebase's runtime surface.

## Goal

Add practical test coverage for all untested modules, prioritizing testable logic (JSON parsing, prompt assembly, error handling, permission callbacks) over mocking LLM round-trips that just return mocked results.

## Approach

Bottom-up, four tiers. Each tier builds confidence for the next. Commit after each tier.

## Tier 1: Easy Wins

### `test_config.py` - Pydantic model validation

`config.py` defines `SuccessCriterion` and `DomainConfig` as Pydantic models. Tests should verify:

- **Required fields**: constructing with missing required fields raises `ValidationError`
- **Defaults**: `target_min`, `target_max` default to `None`; `required` defaults to `True`; `DomainConfig.run_command` has a default template; `success_criteria` defaults to empty list
- **Optional fields**: `domain_knowledge` defaults to empty string, `protected_paths` and `experiment_dependencies` default to empty lists
- **Type coercion**: Pydantic's behavior on wrong types (string where int expected, etc.)

### `test_models.py` - LLM client wrappers

All three clients (`query_openai`, `query_google`, `query_anthropic`) follow the same pattern: create a client, optionally configure web search, call the API, extract text. Tests should mock the SDK client and verify:

- **Standard call**: correct model and prompt passed through, text extracted from response
- **Web search branch**: `web_search=True` adds the right tool configuration per provider
- **Empty response**: returns empty string, not `None`

Each provider has slightly different SDK shapes, so each needs its own test class.

### Why these first

These are the simplest modules with zero dependencies on other project code. Getting them green builds the mocking patterns we'll reuse in Tiers 2-3.

## Tier 2: Agent Logic

### Common pattern across agents

All agents (analyst, scientist, coder, discovery, report) follow a similar structure:
1. Read input files (notebook, results, etc.)
2. Format a prompt from templates
3. Call `query()` from `claude_agent_sdk`
4. Parse the response (usually JSON)
5. Validate output artifacts exist

The interesting testable logic is steps 1-2, 4-5. Step 3 is mocked.

### `test_analyst.py`

- **`_format_success_criteria()`**: pure function, test directly
  - Empty list returns placeholder text
  - Single criterion with `target_min` only
  - Single criterion with `target_max` only
  - Criterion with both bounds
  - Required vs optional labels
- **`run_analyst()` prompt assembly**: mock `query()`, verify the prompt includes results content, notebook content, plot paths, success criteria
- **`run_analyst()` JSON parsing**: mock `query()` to return JSON (plain and markdown-fenced), verify correct dict returned
- **`run_analyst()` error on empty output**: mock `query()` returning nothing, verify `RuntimeError`

### `test_scientist.py`

- **`_parse_json_response()`**: pure function, test directly
  - Clean JSON string
  - Markdown-fenced JSON (` ```json ... ``` `)
  - Invalid JSON raises `json.JSONDecodeError`
- **`run_scientist()` prompt assembly**: verify prompt includes analysis JSON, notebook content, version, domain knowledge
- **`run_scientist()` with no notebook**: verify fallback text used
- **`run_scientist()` with no analysis**: verify fallback text for first iteration
- **`run_scientist()` empty output**: verify `RuntimeError`
- **`run_scientist_revision()` transcript formatting**: verify debate transcript is formatted with role headers
- **`run_scientist_revision()` empty output**: verify `RuntimeError`

### `test_coder.py`

- **`_make_permission_callback()`**: this has real logic worth testing
  - Write inside output dir: allowed
  - Write outside output dir: denied with message
  - Edit inside output dir: allowed
  - Edit outside output dir: denied
  - Bash with blocked pattern (`rm -rf`, `git push`, etc.): denied
  - Bash with safe command: allowed
  - Read/Glob/Grep: always allowed
- **`run_coder()` prompt assembly**: verify plan JSON, previous script path, version, dependencies, data path are in the prompt
- **`run_coder()` no previous script**: verify `CODER_NO_PREVIOUS` template used
- **`run_coder()` with previous script**: verify `CODER_HAS_PREVIOUS` template used
- **`run_coder()` missing output**: verify `FileNotFoundError` when script not created

### `test_discovery.py`

- **`run_discovery()` tool selection**: interactive mode adds `AskUserQuestion`
- **`run_discovery()` missing config**: verify `FileNotFoundError` when config not created
- **`run_discovery()` missing script**: verify `FileNotFoundError` when script not created
- **`run_discovery()` config validation**: verify `DomainConfig.model_validate()` is called on output

### `test_report.py`

- **`run_report()` prompt assembly**: verify state fields (domain, goal, iterations, best version/score) and notebook content are in the prompt
- **`run_report()` missing output**: verify `FileNotFoundError` when report not created

## Tier 3: Orchestrator

The orchestrator is a state machine with many methods, each delegating to an agent. Tests mock all agent calls and verify the state machine logic.

### `test_orchestrator.py`

**Constructor and setup:**
- Defaults (max_iterations=20, empty critic list, debate_rounds=2)
- Custom values passed through

**Phase transitions (`run()`):**
- Fresh start from ingestion: phases transition ingestion -> discovery -> iteration
- Resume from discovery: skips ingestion
- Resume from iteration: goes straight to iteration loop
- Max iterations triggers report phase
- Consecutive failures trigger report phase
- Scientist `should_stop=True` triggers report phase

**`_run_ingestion()`:**
- Uses `raw_data_path` on resume, `data_path` on fresh run
- Raises `ValueError` when no data path available
- Updates `state.data_path` with canonical path
- Adds canonical data dir to `config.protected_paths`

**`_run_iteration()` flow:**
- Increments iteration counter
- Calls analyst -> scientist -> debate -> revision -> coder -> validate -> run -> evaluate in order
- Skips debate when no critics configured
- Skips revision when no debate result
- Records failure on syntax error (skips run)
- Records failure on coder error

**`_evaluate()`:**
- `None` result: status=failed, record_failure
- Timed out: status=failed, record_failure
- Non-zero exit: status=failed, record_failure
- Success: status=completed, record_success, sets results_path

**`_notebook_content()`:**
- Returns file content when notebook exists
- Returns empty string when missing

## Tier 4: CLI

### `test_cli.py`

Using Click's `CliRunner`:

- **`load_domain_config()`**: mock `importlib.import_module`, verify config and domain knowledge loading, verify fallback when prompts module missing
- **`run` command**: verify required options (`--data`, `--goal`), verify `Orchestrator` constructed with correct args, verify `asyncio.run()` called
- **`resume` command**: verify state loaded from path, verify `Orchestrator` constructed for resumption
- **`status` command**: create a state file, invoke status, verify output contains expected fields

## Testing conventions

Following established patterns from existing tests:
- `pytest` with `@pytest.mark.asyncio` for async tests
- `unittest.mock.AsyncMock` and `patch()` for mocking
- `tmp_path` fixture for file I/O
- Class-based grouping by function/feature
- One test file per source module

## Out of scope

- Prompt template modules (pure string constants, no logic)
- Integration tests that call real LLM APIs
- Coverage percentage targets (focus on testing logic that can break)
