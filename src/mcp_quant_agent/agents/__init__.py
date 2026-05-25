"""Agent modules — LangGraph orchestration of MCP tool calls.

Implementation order (see CLAUDE.md and docs/PLAN.md):
1. Single-agent orchestrator (this sprint) — minimal perceive → reason → act → observe
   LangGraph graph.  Gets evaluation pipeline running before adding complexity.
2. Multi-agent roles (week 2) — Bull/Bear debate, risk node, synthesising trader.

Status: stubs in place; implementation in the next sprint (after eval pipeline).
"""
