"""FinBERT sentiment analysis — adapted from Salvi reference code.

Source: ``gen-ai-imperial/code/week3_agent/sentiment.py`` (read-only reference).
Adaptations:
- Type annotations.
- Lazy import of transformers (only when called, so the server starts quickly
  even without torch installed).
- Docstrings.

Requires the ``sentiment`` optional extra:
    uv pip install 'mcp-quant-agent[sentiment]' torch
"""

from __future__ import annotations

from typing import Any

_PIPELINE: Any | None = None


def _get_pipeline() -> Any:
    """Lazily load the FinBERT sentiment pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for sentiment analysis. "
                "Install: uv pip install 'mcp-quant-agent[sentiment]' torch"
            ) from exc

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _PIPELINE = pipeline(  # type: ignore[call-overload]
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
        )
    return _PIPELINE


def analyze_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    """Score each text with FinBERT as positive / negative / neutral.

    Parameters
    ----------
    texts:
        List of financial headline strings.

    Returns
    -------
    list[dict[str, Any]]
        One dict per input text: ``{"text": ..., "label": ..., "score": ...}``.
        ``label`` is one of ``"positive"``, ``"negative"``, ``"neutral"``.
        ``score`` is the model confidence in [0, 1].

    Examples
    --------
    >>> results = analyze_sentiment(["Apple reports record earnings"])
    >>> results[0]["label"] in {"positive", "negative", "neutral"}
    True
    """
    if not texts:
        return []
    pipe = _get_pipeline()
    raw: list[dict[str, Any]] = pipe(texts, batch_size=16)
    return [
        {
            "text": t[:120],
            "label": r["label"].lower(),
            "score": round(float(r["score"]), 4),
        }
        for t, r in zip(texts, raw, strict=False)
    ]
