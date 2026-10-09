"""
Agents package for StockSage multi-agent system.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.analyst_agent import analyst_agent_node
    from src.agents.critic_node import critic_node, route_from_critic
    from src.agents.data_agent import data_agent_node
    from src.agents.news_agent import news_agent_node
    from src.agents.risk_agent import risk_agent_node
    from src.agents.supervisor import supervisor_node


def __getattr__(name: str):
    if name == "data_agent_node":
        from src.agents.data_agent import data_agent_node
        return data_agent_node
    if name == "news_agent_node":
        from src.agents.news_agent import news_agent_node
        return news_agent_node
    if name == "risk_agent_node":
        from src.agents.risk_agent import risk_agent_node
        return risk_agent_node
    if name == "analyst_agent_node":
        from src.agents.analyst_agent import analyst_agent_node
        return analyst_agent_node
    if name == "supervisor_node":
        from src.agents.supervisor import supervisor_node
        return supervisor_node
    if name in ("critic_node", "route_from_critic"):
        from src.agents import critic_node as cn
        return getattr(cn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "analyst_agent_node",
    "critic_node",
    "data_agent_node",
    "news_agent_node",
    "risk_agent_node",
    "route_from_critic",
    "supervisor_node",
]
