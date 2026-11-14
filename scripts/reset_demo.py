"""Put the demo vault back to its starting state before a rehearsal or the defence.

    uv run python scripts/reset_demo.py

* removes the status lines the agent appended to request notes (everything from "---\\n**Агент");
* deletes the plan notes the agent wrote in demo-vault/Плани/ (they are agent output, git-ignored).
Touches nothing else. Obsidian picks up the changes by itself.
"""

from __future__ import annotations

from pathlib import Path

VAULT = Path(__file__).resolve().parents[1] / "demo-vault"
MARKER = "\n\n---\n**Агент"


def main() -> None:
    for note in sorted((VAULT / "Запити").glob("*.md")):
        text = note.read_text(encoding="utf-8")
        cut = text.find(MARKER)
        if cut != -1:
            note.write_text(text[:cut].rstrip() + "\n", encoding="utf-8")
            print(f"cleaned  {note.relative_to(VAULT)}")
    for plan in sorted((VAULT / "Плани").glob("*.md")):
        plan.unlink()
        print(f"deleted  {plan.relative_to(VAULT)}")
    print("demo vault reset")


if __name__ == "__main__":
    main()
