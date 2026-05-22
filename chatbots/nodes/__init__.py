
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.graph import MessagesState
from chatbots.constants import MEMORY_CFG

SYSTEM_PROMPT = """You are an expert anime assistant. You help users discover,
learn about, and find similar anime.

You have access to the following tools:
1. fuzzy_find_anime   – Find an anime by name in the local database.
                        Always call this FIRST when the user mentions an anime
                        by name, so you can resolve the correct anime_id.
2. get_anime_info     – Fetch detailed local info (synopsis, genres, rating,
                        episodes, era, studios …) using an anime_id.
3. search_anime_web   – Search the web for information not in the local data
                        (characters, news, airdate, sequels, etc.).
4. find_similar_anime – Recommend anime similar to a description + genre list.

Workflow
--------
• User names an anime  →  fuzzy_find_anime → get_anime_info
  → if info is missing / user wants more  → search_anime_web
• User wants recommendations  →  find_similar_anime (build description from
  what the user tells you; infer genres where reasonable)
• Keep answers friendly, concise, and well-formatted.
"""


def make_agent_node(llm_with_tools):
    """Return an agent node function bound to the given LLM."""

    def agent_node(state: MessagesState) -> dict:
        trimmed = trim_messages(
            state["messages"],
            max_tokens=MEMORY_CFG["max_tokens"],
            strategy="last",
            token_counter=llm_with_tools,
            include_system=False,
        )
        messages  = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed
        response  = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node