"""Tests for the read_predictions MCP tool."""

import json
from pathlib import Path

import pytest

from auto_scientist.agents.prediction_tool import (
    _handle_read_predictions,
    build_prediction_mcp_server,
)
from auto_scientist.state import PredictionRecord


@pytest.fixture
def sample_history() -> list[PredictionRecord]:
    return [
        PredictionRecord(
            pred_id="0.1",
            iteration_prescribed=0,
            iteration_evaluated=1,
            prediction="Negative hardness clusters in high-alloying",
            diagnostic="check composition overlap",
            if_confirmed="exclude safely",
            if_refuted="keep all data",
            outcome="confirmed",
            evidence="11/13 (85%) meet criterion",
            summary="85% negative-hardness rows meet high-alloy criterion",
        ),
        PredictionRecord(
            pred_id="0.2",
            iteration_prescribed=0,
            iteration_evaluated=1,
            prediction="Cr correlation is strongest for corrosion",
            diagnostic="compute CLR correlations",
            if_confirmed="use Cr model",
            if_refuted="look elsewhere",
            outcome="refuted",
            evidence="Ni dominates at r_s=0.613; Cr near zero",
            summary="Cr r_s near zero; Ni dominates at 0.613",
        ),
        PredictionRecord(
            pred_id="1.1",
            iteration_prescribed=1,
            iteration_evaluated=2,
            prediction="Corrosion is synthetic score",
            diagnostic="regress on raw elements",
            if_confirmed="focus on hardness",
            if_refuted="corrosion is real",
            follows_from="0.2",
            outcome="confirmed",
            evidence="OLS R²=0.84 > 0.80 threshold",
            summary="OLS R²=0.84; corrosion is synthetic",
        ),
        PredictionRecord(
            pred_id="2.1",
            iteration_prescribed=2,
            prediction="K-means reveals alloy families",
            diagnostic="cluster and compare variance",
            if_confirmed="per-cluster models",
            if_refuted="noise is intrinsic",
            follows_from="1.1",
        ),
        PredictionRecord(
            pred_id="2.2",
            iteration_prescribed=2,
            iteration_evaluated=3,
            prediction="RF beats Elastic Net",
            diagnostic="nested CV comparison",
            if_confirmed="use RF",
            if_refuted="linear is sufficient",
            outcome="confirmed",
            evidence="RF R²=0.80 vs EN R²=0.47",
            summary="RF R²=0.80 vs EN R²=0.47",
        ),
        PredictionRecord(
            pred_id="3.1",
            iteration_prescribed=3,
            iteration_evaluated=3,
            prediction="Mo effect is nonlinear",
            diagnostic="ALE plots",
            if_confirmed="report Mo shape",
            if_refuted="Mo is linear",
            outcome="inconclusive",
            evidence="Positive but sparse at high-Mo",
            summary="Positive trend but sparse at >12% Mo",
        ),
    ]


class TestHandleReadPredictions:
    @pytest.mark.asyncio
    async def test_query_by_pred_id(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"pred_ids": ["1.1"]})
        text = result["content"][0]["text"]
        assert "[1.1]" in text
        assert "Corrosion is synthetic score" in text
        assert "OLS R²=0.84" in text

    @pytest.mark.asyncio
    async def test_query_multiple_pred_ids(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"pred_ids": ["0.1", "0.2"]})
        text = result["content"][0]["text"]
        assert "[0.1]" in text
        assert "[0.2]" in text

    @pytest.mark.asyncio
    async def test_query_unknown_pred_id(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"pred_ids": ["99.99"]})
        text = result["content"][0]["text"]
        assert "not found" in text.lower() or "no predictions" in text.lower()

    @pytest.mark.asyncio
    async def test_filter_pending(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"filter": "pending"})
        text = result["content"][0]["text"]
        assert "[2.1]" in text
        assert "[0.1]" not in text  # confirmed, not pending

    @pytest.mark.asyncio
    async def test_filter_refuted(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"filter": "refuted"})
        text = result["content"][0]["text"]
        assert "[0.2]" in text
        assert "[0.1]" not in text

    @pytest.mark.asyncio
    async def test_filter_active_chains(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"filter": "active_chains"})
        text = result["content"][0]["text"]
        # 2.1 is pending
        assert "[2.1]" in text
        # 1.1 is ancestor of 2.1 (via follows_from)
        assert "[1.1]" in text
        # 0.2 is ancestor of 1.1
        assert "[0.2]" in text
        # 0.1 is NOT in the active chain (different branch)
        assert "[0.1]" not in text

    @pytest.mark.asyncio
    async def test_filter_by_iteration(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"iteration": 2})
        text = result["content"][0]["text"]
        assert "[2.1]" in text
        assert "[2.2]" in text
        assert "[0.1]" not in text

    @pytest.mark.asyncio
    async def test_filter_confirmed_returns_subset(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"filter": "confirmed"})
        text = result["content"][0]["text"]
        assert "[0.1] CONFIRMED" in text
        assert "[1.1] CONFIRMED" in text
        # Refuted prediction 0.2 should not appear as a primary entry
        assert "[0.2] REFUTED" not in text

    @pytest.mark.asyncio
    async def test_empty_history(self):
        result = await _handle_read_predictions([], {})
        text = result["content"][0]["text"]
        assert "no prediction" in text.lower()

    @pytest.mark.asyncio
    async def test_no_args_returns_error(self, sample_history):
        """No arguments should return an error, not dump all predictions."""
        result = await _handle_read_predictions(sample_history, {})
        text = result["content"][0]["text"]
        assert "please specify" in text.lower()
        # Should list available IDs to help the model
        assert "0.1" in text

    @pytest.mark.asyncio
    async def test_chain_returns_ancestors_and_descendants(self, sample_history):
        """chain='1.1' should return 0.2 (ancestor), 1.1 (self), 2.1 (descendant)."""
        result = await _handle_read_predictions(sample_history, {"chain": "1.1"})
        text = result["content"][0]["text"]
        assert "[0.2]" in text  # ancestor (1.1 follows_from 0.2)
        assert "[1.1]" in text  # self
        assert "[2.1]" in text  # descendant (2.1 follows_from 1.1)
        # Should NOT include unrelated predictions
        assert "[0.1]" not in text
        assert "[2.2]" not in text

    @pytest.mark.asyncio
    async def test_chain_unknown_id(self, sample_history):
        result = await _handle_read_predictions(sample_history, {"chain": "99.99"})
        text = result["content"][0]["text"]
        assert "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_chain_root_prediction(self, sample_history):
        """chain on a root (no ancestors) returns self + descendants."""
        result = await _handle_read_predictions(sample_history, {"chain": "0.2"})
        text = result["content"][0]["text"]
        assert "[0.2]" in text
        assert "[1.1]" in text  # child
        assert "[2.1]" in text  # grandchild

    @pytest.mark.asyncio
    async def test_chain_leaf_prediction(self, sample_history):
        """chain on a leaf (no descendants) returns ancestors + self."""
        result = await _handle_read_predictions(sample_history, {"chain": "2.1"})
        text = result["content"][0]["text"]
        assert "[0.2]" in text  # grandparent
        assert "[1.1]" in text  # parent
        assert "[2.1]" in text  # self

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, sample_history):
        """stats=true returns counts by status and iteration, not full detail."""
        result = await _handle_read_predictions(sample_history, {"stats": True})
        text = result["content"][0]["text"]
        assert "Total: 6 predictions" in text
        assert "confirmed:" in text
        assert "refuted:" in text
        assert "inconclusive:" in text
        assert "iter 0:" in text
        assert "iter 1:" in text
        # Should NOT contain full detail fields
        assert "Diagnostic:" not in text
        assert "If confirmed:" not in text

    @pytest.mark.asyncio
    async def test_output_includes_full_detail(self, sample_history):
        """Full detail should include prediction, diagnostic, evidence, etc."""
        result = await _handle_read_predictions(sample_history, {"pred_ids": ["0.2"]})
        text = result["content"][0]["text"]
        assert "Prediction:" in text
        assert "Diagnostic:" in text
        assert "Evidence:" in text


class TestBuildPredictionMcpServer:
    def test_creates_stdio_server(self, sample_history):
        server = build_prediction_mcp_server(sample_history)
        assert server is not None
        assert server["type"] == "stdio"
        assert server["command"] == "python3"
        assert "_prediction_mcp_server.py" in server["args"][0]
        # Verify the temp file was written with correct data
        predictions_path = server["args"][1]
        data = json.loads(Path(predictions_path).read_text())
        assert len(data) == len(sample_history)
        assert data[0]["pred_id"] == "0.1"

    def test_writes_to_output_dir(self, sample_history, tmp_path):
        server = build_prediction_mcp_server(sample_history, output_dir=tmp_path)
        predictions_path = server["args"][1]
        assert predictions_path == str(tmp_path / "predictions.json")
        data = json.loads(Path(predictions_path).read_text())
        assert len(data) == len(sample_history)
        assert data[0]["pred_id"] == "0.1"
