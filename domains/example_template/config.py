"""Example domain configuration template.

Copy this file and modify for your domain. See domains/spo2/config.py for a
complete real-world example.
"""

from auto_scientist.config import DomainConfig, SuccessCriterion

EXAMPLE_CONFIG = DomainConfig(
    name="example",
    description="Description of what this domain models",
    data_paths=["path/to/your/data.csv"],
    run_command="uv run python -u {script_path}",
    run_cwd=".",
    run_timeout_minutes=30,
    version_prefix="v",
    success_criteria=[
        SuccessCriterion(
            name="example criterion",
            description="What this criterion measures",
            metric_key="your_metric",
            target_min=0.0,
            target_max=1.0,
        ),
    ],
    domain_knowledge="Domain-specific context injected into agent prompts.",
    protected_paths=["path/to/data/"],
)
