"""
demo.py — Quick smoke-test for the AnimeAgent.

Run from the project root:
    python demo.py
"""

from chatbots.agent import AnimeAgent


def separator(label: str = "") -> None:
    print("\n" + "─" * 60)
    if label:
        print(f"  {label}")
    print("─" * 60)


def run_demo() -> None:
    print("Initialising AnimeAgent …")
    agent = AnimeAgent()
    print("✅  Agent ready.\n")

    # ── Session 1: multi-turn conversation ────────────────────
    user_id   = "demo_user"
    thread_id = agent.new_session(user_id)
    print(f"Session started  →  thread_id = {thread_id}")

    turns = [
        "Tell me about Fullmetal Alchemist Brotherhood",
        "Who are the main characters?",           # agent should remember context
        "Recommend something similar to it",
    ]

    for msg in turns:
        separator(f"User: {msg}")
        reply = agent.chat(msg, thread_id=thread_id)
        print(f"\nAgent:\n{reply}")

    # ── Show stored history ───────────────────────────────────
    separator("Message history (session 1)")
    for entry in agent.get_history(thread_id):
        role    = entry["role"].upper()
        preview = str(entry["content"])[:120].replace("\n", " ")
        print(f"  [{role}]  {preview}")

    # ── Session 2: fresh memory, different user ───────────────
    separator("Session 2 — fresh start")
    tid2  = agent.new_session("another_user")
    reply = agent.chat("I want to watch a dark fantasy anime with magic", thread_id=tid2)
    print(f"\nAgent:\n{reply}")

    # ── Clear session 1 ───────────────────────────────────────
    separator("Clearing session 1")
    agent.clear_session(thread_id)
    print(f"Session {thread_id} cleared from memory.")


if __name__ == "__main__":
    run_demo()