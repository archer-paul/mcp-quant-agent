"""mcp-quant-agent — Autonomous Trading Agents via Model Context Protocol.

MSc Mathematics & Finance thesis · Imperial College London
Supervisor: Cristopher Salvi

Entry points
------------
- ``mcp_quant_agent.clock``    — simulation clock (t_now), anti-look-ahead guards
- ``mcp_quant_agent.config``   — project-wide settings (pydantic-settings)
- ``mcp_quant_agent.mcp_servers`` — MCP tool servers (data, analytics, execution)
- ``mcp_quant_agent.eval``     — financial metrics + regime + CoT scoring
- ``mcp_quant_agent.observability`` — Langfuse Cloud tracing wiring
"""

__version__ = "0.1.0"
