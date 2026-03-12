"""SpO2 domain configuration.

Note: success_criteria and domain_knowledge have moved to ExperimentState.
They are now defined adaptively by the Scientist during the iteration loop.
"""

from auto_scientist.config import DomainConfig

SPO2_CONFIG = DomainConfig(
    name="spo2",
    description=(
        "Model SpO2 dynamics during voluntary breath-holds using a two-stage approach: "
        "sensor calibration (latent SaO2 + gamma kernel) then physiology (Severinghaus ODC). "
        "Single subject, 5 breath-holds at different lung volumes."
    ),
    data_paths=["domains/spo2/seed/data/spo2.db"],
    run_command="uv run {script_path}",
    run_cwd=".",
    run_timeout_minutes=120,
    version_prefix="v",
    protected_paths=["domains/spo2/seed/data/"],
)
