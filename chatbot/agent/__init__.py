import uuid
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from chatbot.constants import (
    GOOGLE_API_KEY, TAVILY_API_KEY,
    MODEL, ARTIFACT_PATHS,
    FUZZY_SEARCH_CFG, SIMILARITY_CFG, WEB_SEARCH_CFG,
)
from chatbot.tools import (
    FuzzyFindAnimeTool,
    GetAnimeInfoTool,
    SearchAnimeWebTool,
    FindSimilarAnimeTool,
)
from chatbot.nodes import make_agent_node


class AnimeAgent:
    """
    Session-based anime chatbot.

    Each conversation lives under a unique thread_id.
    Memory is in-process (MemorySaver) — resets when the process restarts.

    Usage
    -----
        agent    = AnimeAgent()
        tid      = agent.new_session("user_42")
        reply    = agent.chat("Tell me about Bleach", thread_id=tid)
        history  = agent.get_history(tid)
        agent.clear_session(tid)
    """

    def __init__(self) -> None:
        # ── Load anime data ────────────────────────────────────
        csv_path = Path(ARTIFACT_PATHS["anime_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Anime CSV not found: {csv_path}")
        anime_df = pd.read_csv(csv_path)

        # ── Instantiate tool classes ───────────────────────────
        _fuzzy   = FuzzyFindAnimeTool(anime_df, FUZZY_SEARCH_CFG)
        _info    = GetAnimeInfoTool(anime_df, ARTIFACT_PATHS["feature_bins"])
        _web     = SearchAnimeWebTool(TAVILY_API_KEY, WEB_SEARCH_CFG)
        _similar = FindSimilarAnimeTool(
            anime_df,
            SIMILARITY_CFG,
            vectorizer_path=ARTIFACT_PATHS["vectorizer"],
            index_path=ARTIFACT_PATHS["faiss_index"],
            id_map_path=ARTIFACT_PATHS["id_map"],
        )

        # Convert each class instance to a LangChain tool
        lc_tools = [
            _fuzzy.as_tool(),
            _info.as_tool(),
            _web.as_tool(),
            _similar.as_tool(),
        ]

        # ── LLM ────────────────────────────────────────────────
        llm            = ChatGoogleGenerativeAI(model=MODEL, google_api_key=GOOGLE_API_KEY)
        llm_with_tools = llm.bind_tools(lc_tools)

        # ── Graph ──────────────────────────────────────────────
        agent_node = make_agent_node(llm_with_tools)
        tool_node  = ToolNode(lc_tools)

        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")

        self._memory = MemorySaver()
        self._graph  = builder.compile(checkpointer=self._memory)

    # ── Public API ─────────────────────────────────────────────

    def new_session(self, user_id: str) -> str:
        """Generate a fresh thread_id tied to user_id."""
        return f"{user_id}:{uuid.uuid4().hex[:8]}"

    def chat(self, message: str, thread_id: str) -> str:
        """Send a message and return the agent reply.

        Args:
            message:   User's input text.
            thread_id: Session identifier from new_session().
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return result["messages"][-1].content

    def clear_session(self, thread_id: str) -> None:
        """Wipe all memory for a given session."""
        keys = [k for k in self._memory.storage if thread_id in str(k)]
        for k in keys:
            del self._memory.storage[k]

    def get_history(self, thread_id: str) -> list[dict]:
        """Return message history for a session (useful for debugging).

        Args:
            thread_id: Session identifier.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state  = self._graph.get_state(config)
        return [
            {"role": m.type, "content": m.content}
            for m in state.values.get("messages", [])
        ]