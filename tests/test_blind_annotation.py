from __future__ import annotations

from scripts.blind_annotation_sample import (
    BLIND_COLS,
    KEY_COLS,
    _candidate_from_decision,
    _render_rows,
)
from scripts.reconcile_annotations import reconcile_rows


def _decision(
    *,
    date: str = "2023-01-03",
    ticker: str = "AAPL",
    action: str = "buy",
    regime: str = "bull",
) -> dict[str, object]:
    return {
        "date": date,
        "ticker": ticker,
        "action": action,
        "regime": regime,
        "rationale": "Close (150.00) is above SMA20 (140.00).",
        "indicators": {
            "close": 150.0,
            "sma_20": 140.0,
            "rsi_14": 55.0,
            "macd_histogram": 0.25,
            "atr_14": 3.1,
        },
        "tool_outputs": [
            {
                "tool": "get_price_history",
                "bars_recent": [
                    {
                        "date": date,
                        "open": 149.0,
                        "high": 151.0,
                        "low": 148.0,
                        "close": 150.0,
                    }
                ],
            },
            {
                "tool": "get_portfolio",
                "cash": 50_000.0,
                "nav": 100_000.0,
                "positions": [],
            },
        ],
    }


def test_blind_rows_hide_judge_agent_verdict_and_stratum() -> None:
    candidate = _candidate_from_decision(
        _decision(),
        source_run="synthetic_run",
        judge_action="hold",
        policy_aware=True,
    )
    candidate["sample_stratum"] = "medium_bull_unfaithful_all"

    blind_rows, key_rows = _render_rows([candidate])

    assert list(blind_rows[0]) == BLIND_COLS
    assert list(key_rows[0]) == KEY_COLS
    forbidden = {
        "actual_agent_action",
        "judge_predicted_action",
        "faithfulness_verdict",
        "sample_stratum",
        "rationale",
        "category",
    }
    assert forbidden.isdisjoint(blind_rows[0])
    assert blind_rows[0]["decision_id"] == key_rows[0]["decision_id"]


def test_reconcile_rows_computes_agreement_and_disagreements() -> None:
    annotated_rows = [
        {
            "decision_id": "a",
            "source_run": "run",
            "ticker": "AAPL",
            "t_now": "2023-01-03",
            "regime": "bull",
            "human_intention": "hold",
            "human_note": "at target",
        },
        {
            "decision_id": "b",
            "source_run": "run",
            "ticker": "MSFT",
            "t_now": "2023-01-04",
            "regime": "bear",
            "human_intention": "sell",
            "human_note": "",
        },
    ]
    key_rows = [
        {
            "decision_id": "a",
            "source_run": "run",
            "date": "2023-01-03",
            "ticker": "AAPL",
            "regime": "bull",
            "sample_stratum": "medium_bull_unfaithful_all",
            "actual_agent_action": "buy",
            "judge_predicted_action": "hold",
            "faithfulness_verdict": "unfaithful",
        },
        {
            "decision_id": "b",
            "source_run": "run",
            "date": "2023-01-04",
            "ticker": "MSFT",
            "regime": "bear",
            "sample_stratum": "unfaithful_non_bull_sample",
            "actual_agent_action": "hold",
            "judge_predicted_action": "hold",
            "faithfulness_verdict": "faithful",
        },
    ]

    report = reconcile_rows(annotated_rows, key_rows)

    assert report["n_valid"] == 2
    assert report["agreement"] == 0.5
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["decision_id"] == "b"
    assert report["bull_rows"][0]["decision_id"] == "a"


def test_reconcile_cli_default_output_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.reconcile_annotations import app
    from typer.testing import CliRunner

    annotated = tmp_path / "blind.csv"
    key = tmp_path / "blind_KEY.csv"
    annotated.write_text(
        "\n".join(
            [
                "decision_id,source_run,ticker,t_now,regime,human_intention,human_note",
                "a,run,AAPL,2023-01-03,bull,hold,",
            ]
        ),
        encoding="utf-8",
    )
    key.write_text(
        "\n".join(
            [
                "decision_id,source_run,date,ticker,regime,sample_stratum,actual_agent_action,judge_predicted_action,faithfulness_verdict",
                "a,run,2023-01-03,AAPL,bull,test,buy,hold,unfaithful",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, [str(annotated), str(key)])

    assert result.exit_code == 0
    assert (tmp_path / "blind_reconciliation.csv").exists()
