"""SpO2 domain configuration."""

from auto_scientist.config import DomainConfig, SuccessCriterion

SPO2_CONFIG = DomainConfig(
    name="spo2",
    description=(
        "Model SpO2 dynamics during voluntary breath-holds using a two-stage approach: "
        "sensor calibration (latent SaO2 + gamma kernel) then physiology (Severinghaus ODC). "
        "Single subject, 5 breath-holds at different lung volumes."
    ),
    data_paths=["domains/spo2/seed/data/spo2.db"],
    run_command="uv run python -u {script_path}",
    run_cwd=".",
    run_timeout_minutes=120,
    version_prefix="v",
    success_criteria=[
        SuccessCriterion(
            name="b_s near unity",
            description="Sensor gain b_s should be close to 1.0 (no amplification needed)",
            metric_key="b_s",
            target_min=0.8,
            target_max=1.2,
        ),
        SuccessCriterion(
            name="tau_0 physiological",
            description="Base sensor delay should reflect real circulation time",
            metric_key="tau_0",
            target_min=10.0,
            target_max=30.0,
        ),
        SuccessCriterion(
            name="low saturation",
            description="Predictions exceeding 100% SpO2 should be rare",
            metric_key="saturation_pct",
            target_max=5.0,
        ),
        SuccessCriterion(
            name="delta range reasonable",
            description="Per-hold timing adjustments should not span too wide",
            metric_key="delta_range_s",
            target_max=15.0,
        ),
        SuccessCriterion(
            name="LOHO timing",
            description="Leave-one-hold-out timing error should be small",
            metric_key="loho_timing_s",
            target_max=5.0,
        ),
        SuccessCriterion(
            name="tau_0 identifiable",
            description="Profile likelihood should be non-monotone (clear minimum)",
            metric_key="profile_nonmonotone",
            target_min=1.0,
            target_max=1.0,
        ),
        SuccessCriterion(
            name="gamma interior",
            description="ODC steepness should not be at parameter bounds",
            metric_key="gamma_at_bound",
            target_max=0.0,
        ),
        SuccessCriterion(
            name="weak-lag stability",
            description="Physiology estimates stable under timing perturbation",
            metric_key="weaklag_divergence_pct",
            target_max=20.0,
        ),
    ],
    domain_knowledge="See domains/spo2/prompts.py for full domain knowledge.",
    protected_paths=["domains/spo2/seed/data/"],
    experiment_dependencies=[
        "scipy>=1.14.0",
        "numpy>=2.1.0",
        "matplotlib>=3.10.0",
        "loguru>=0.7.3",
    ],
)
