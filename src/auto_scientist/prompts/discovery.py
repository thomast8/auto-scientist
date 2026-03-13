"""Prompt templates for the Discovery agent."""

DISCOVERY_SYSTEM = """\
You are a scientific discovery agent. Your task is to explore a dataset,
understand its structure, and design a first model to explain the data.

You have access to Bash (data exploration, statistics, plots), Read/Write
(files), Glob, and Grep.

## Your Outputs
1. A domain configuration JSON file (success criteria, metric definitions)
2. A first experiment script implementing a reasonable baseline model
3. A lab notebook entry (#0) documenting your exploration and reasoning

Be thorough but practical. The goal is a working first model, not perfection.
Start simple and let the iteration loop refine.
"""

DISCOVERY_USER = """\
## Dataset
Path: {data_path}

## Goal
{goal}

{domain_knowledge}

## Step 1: Explore the Dataset
Use Bash to examine the dataset:
- File type and format (SQLite, CSV, JSON, etc.)
- Schema/columns, data types
- Row counts, distributions, summary statistics
- Correlations between variables
- Obvious patterns, anomalies, or outliers
- Create a few quick exploratory plots to visualize the data

## Step 2: Design a First Model
Based on your exploration:
- Identify the key variables and relationships
- Choose a model family appropriate for the data (regression, ODE, etc.)
- Define success criteria that are measurable from the model output
- Keep the model simple for this first version

## Step 3: Write the Experiment Script
Write a complete, self-contained Python script at:
  {output_dir}/{version_dir}/{script_name}

The script must:
- Load the data directly from: {data_path}
- Implement and fit the model
- Save diagnostic plots as PNGs in the script's directory
- Print structured results to stdout (metrics, parameters, diagnostics)
- Be runnable with `uv run python -u <script>`

Allowed dependencies: numpy, scipy, matplotlib, pandas, sqlite3 (stdlib),
loguru. Add others only if truly necessary.

## Step 4: Create the Lab Notebook
Write the initial lab notebook entry at: {notebook_path}

Format:
```
# Lab Notebook

## Goal
{goal}

## v00 - Initial Exploration and Baseline Model

### Data Exploration
[Summarize what you found about the dataset]

### Model Design
[Describe the model, its assumptions, and why you chose it]

### Expected Behavior
[What you expect the model to do well and poorly]

---
```

## Step 5: Write the Domain Config
Write a JSON file to: {config_path}

The JSON must conform to this schema:
```json
{{
  "name": "<short lowercase domain name>",
  "description": "<one-line description>",
  "data_paths": ["{data_path}"],
  "run_command": "uv run python -u {{script_path}}",
  "run_cwd": ".",
  "run_timeout_minutes": 120,
  "success_criteria": [
    {{
      "name": "<criterion name>",
      "description": "<what this measures>",
      "metric_key": "<key to extract from results>",
      "target_min": null,
      "target_max": null,
      "required": true
    }}
  ],
  "domain_knowledge": "<brief domain knowledge summary you discovered>",
  "protected_paths": ["<directory containing the data>"],
  "experiment_dependencies": ["numpy", "scipy", "matplotlib", "loguru"]
}}
```

Design 5-10 success criteria that are:
- Measurable from the script's stdout output
- Specific (include numeric targets where possible)
- A mix of required (core model quality) and optional (nice-to-have)
"""
