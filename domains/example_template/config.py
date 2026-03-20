"""Example domain configuration template.

Copy this file and modify for your domain. See domains/spo2/config.py for a
complete real-world example.

Note: success_criteria and domain_knowledge are defined adaptively by the
Scientist during the iteration loop and stored in ExperimentState.
"""

from auto_scientist.config import DomainConfig

EXAMPLE_CONFIG = DomainConfig(
    name="example",
    description="Description of what this domain models",
    data_paths=["path/to/your/data.csv"],
    run_command="uv run {script_path}",
    run_cwd=".",
    run_timeout_minutes=30,
    version_prefix="v",
    protected_paths=["path/to/data/"],
)
