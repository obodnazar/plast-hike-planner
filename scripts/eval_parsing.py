"""Measure how well a model + our code understand free-text hike requests (eval/cases/*.yaml).

  uv run python scripts/eval_parsing.py --provider ollama --set dev --label baseline
  uv run python scripts/eval_parsing.py --provider openrouter --set dev --label baseline --reserve 15
  uv run python scripts/eval_parsing.py --replay baseline --label stage1
  uv run python scripts/eval_parsing.py --report

Each run is saved to eval/results/<label>_<set>_<provider>-<model>.json after every case. Running the
same command again resumes where it stopped, as long as the prompt, schema and parsing code are
unchanged (a fingerprint checks this). So the OpenRouter part can be spread over several days: before
each case the script asks OpenRouter how many free requests are left today and stops at --reserve.

--replay re-applies the CURRENT code (sanitise + understanding.refine) to the model answers saved in
earlier runs, without any model call: so a code-only change is measured for free and on exactly the
same model answers. It is valid while the prompt and the schema are unchanged.

For the TEST set only the totals are printed: its individual mistakes must not steer our fixes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plast_agent import llm as L  # noqa: E402
from plast_agent import understanding as U  # noqa: E402
from plast_agent import workflow as W  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from plast_agent.config import ConfigError, load_settings  # noqa: E402
from plast_agent.mcp_links import McpLinks  # noqa: E402
from plast_trail import places as gazetteer  # noqa: E402
from plast_trail.server import mcp as plast_trail_server  # noqa: E402
from plast_agent.evaluation import (FIELDS, describe_errors, load_cases, render_report,  # noqa: E402
                                    render_summary, score_case, summarise)

CASES = ROOT / "eval" / "cases"
RESULTS = ROOT / "eval" / "results"
REPORT = ROOT / "eval" / "RESULTS.md"


def _hash(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def model_fingerprint() -> str:
    """What the model sees: the prompt and the schema. Replaying saved answers is valid while it is unchanged."""
    return _hash([W.PARSE_SYSTEM, json.dumps(L.ParsedRequest.model_json_schema(), sort_keys=True, ensure_ascii=False)])


def fingerprint() -> str:
    """Changes whenever what decides the result changes: the prompt, the schema or the code after the model."""
    return _hash([model_fingerprint(), inspect.getsource(W.read_with_model), inspect.getsource(W.interpret),
                  inspect.getsource(W.sanitise_request), inspect.getsource(U),
                  inspect.getsource(W.resolve_named_places), inspect.getsource(gazetteer),
                  hashlib.sha256(gazetteer.PLACES_FILE.read_bytes()).hexdigest()])


@contextlib.asynccontextmanager
async def plast_trail_links():
    """The real plast-trail server in memory (no network) for resolve_places: the agent's own MCP path."""
    logging.disable(logging.INFO)   # the server logs every call; the evaluation prints its own lines
    async with create_connected_server_and_client_session(plast_trail_server) as session:
        links = McpLinks(SimpleNamespace(obsidian_mcp_url="-", plast_trail_url="memory://plast-trail"))
        links.sessions = {"plast-trail": session}
        yield links


async def after_model(raw: dict, text: str, today, links) -> tuple[dict, list[str]]:
    """Everything the agent does after the model's reading: code checks, then plast-trail.resolve_places."""
    parsed, notes, _ = W.interpret(raw, text, today)
    with contextlib.redirect_stdout(io.StringIO()):
        notes += await W.resolve_named_places(links, parsed, text)
    return parsed, notes


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return head + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_provider(settings, name: str, model: str | None) -> tuple[L.Provider, object]:
    if model:
        settings = dataclasses.replace(settings, **({"ollama_model": model} if name == "ollama"
                                                    else {"openrouter_model": model}))
    providers = {p.name: p for p in L.build_providers(settings, with_ollama=True)}
    return providers[name], settings


def save(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run["summary"] = summarise(run["results"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def write_report() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    if runs:
        REPORT.write_text(render_report(runs), encoding="utf-8")
        print(f"\nЗвіт: {REPORT.relative_to(ROOT)}")


async def evaluate(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(need_model=args.provider == "openrouter")
    except ConfigError as exc:
        print(exc)
        return 2
    provider, settings = make_provider(settings, args.provider, args.model)
    if provider.name == "ollama":
        ok, message = L.ollama_status(settings)
        if not ok:
            print(message)
            return 2
    llm = L.LLM(providers=[provider])   # one provider only: a silent switch would mix two models' scores
    fp = fingerprint()
    stopped = False

    for set_name in (["dev", "test"] if args.set == "all" else [args.set]):
        today, cases = load_cases(CASES / f"{set_name}.yaml")
        if args.ids:
            wanted = set(args.ids.split(","))
            cases = [c for c in cases if c.id in wanted]
        model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", provider.model).strip("-")
        out = RESULTS / f"{args.label}_{set_name}_{provider.name}-{model_tag}.json"
        run = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        if run and run.get("model_fingerprint") != model_fingerprint():
            print(f"{out.name}: промпт або схема змінились з того прогону; починаю цей набір заново.")
            run = None
        if run and args.retry_errors:
            kept = [r for r in run["results"] if not r["error"]]
            print(f"  повторюю {len(run['results']) - len(kept)} запит(и), де модель не відповіла")
            run["results"] = kept
        if run and run["fingerprint"] != fp:   # same model answers, newer code after the model: re-score them
            by_id = {c.id: c for c in cases}
            async with plast_trail_links() as links:
                run["results"] = [await rescore(by_id[r["id"]], r, today, links) for r in run["results"]
                                  if r["id"] in by_id]
            run.update(fingerprint=fp, commit=git_commit())
            print("  збережені відповіді моделі перераховано поточним кодом (без нових запитів)")
        if run is None:
            run = {"label": args.label, "set": set_name, "provider": provider.name, "model": provider.model,
                   "fingerprint": fp, "model_fingerprint": model_fingerprint(), "commit": git_commit(),
                   "today": today.isoformat(),
                   "started": datetime.now().isoformat(timespec="seconds"), "cases_total": len(cases),
                   "results": []}
        done = {r["id"] for r in run["results"]}
        todo = [c for c in cases if c.id not in done]
        print(f"\n━━ {set_name}: {len(cases)} запитів, уже є {len(done)}, лишилось {len(todo)} "
              f"· {provider.label} · відбиток {fp}")

        for case in todo:
            if provider.name == "openrouter":
                left = L.free_requests_left(settings)
                if left is not None and left <= args.reserve:
                    print(f"  Стоп: безкоштовних запитів на сьогодні лишилось {left} (резерв {args.reserve}). "
                          "Запустіть ту саму команду після 02:00 за Києвом — продовжить з цього місця.")
                    stopped = True
                    break
            started, calls_before = time.monotonic(), llm.calls
            try:
                raw = await W.read_with_model(llm, case.text, today)
                async with plast_trail_links() as links:
                    parsed, notes = await after_model(raw, case.text, today, links)
                record = score_case(case, parsed) | {"raw": raw, "notes": notes}
            except L.LLMError as exc:
                if "429" in str(exc):
                    print(f"  Стоп: OpenRouter відповів 429 (ліміт). Продовжимо пізніше тією ж командою.")
                    stopped = True
                    break
                record = score_case(case, None, error=str(exc))
            record["seconds"] = round(time.monotonic() - started, 1)
            record["calls"] = llm.calls - calls_before
            run["results"].append(record)
            save(out, run)
            right = sum(o == "correct" for o in record["outcomes"].values())
            line = f"  {case.id}  {record['seconds']:6.1f} с  {right:2d}/{len(FIELDS)} полів"
            if set_name == "dev" or args.details:
                line += f"  {describe_errors(case, record)}"
            print(line)

        if run["results"]:
            save(out, run)
            title = f"{set_name} · {args.label} · {provider.label} ({len(run['results'])}/{len(cases)})"
            print("\n" + render_summary(run["summary"], title))
        if stopped:
            break
    write_report()
    return 1 if stopped else 0


async def rescore(case, r: dict, today, links) -> dict:
    """One saved record scored again with the current code after the model (no model call)."""
    keep = {"seconds": r.get("seconds"), "calls": r.get("calls", 0)}
    if r["error"]:
        return score_case(case, None, error=r["error"]) | keep
    raw = r.get("raw") if r.get("raw") is not None else dict(r["got"])   # old runs saved only the sanitised answer
    parsed, notes = await after_model(raw, case.text, today, links)
    return score_case(case, parsed) | {"raw": raw, "notes": notes} | keep


async def replay(args: argparse.Namespace) -> int:
    """Score saved model answers with the current code: no model call at all."""
    sources = sorted(RESULTS.glob(f"{args.replay}_*.json"))
    if not sources:
        print(f"Немає збережених прогонів з етапом «{args.replay}» в {RESULTS.relative_to(ROOT)}.")
        return 2
    mfp = model_fingerprint()
    for src in sources:
        old = json.loads(src.read_text(encoding="utf-8"))
        if old.get("model_fingerprint", mfp) != mfp:
            print(f"{src.name}: промпт або схема змінились — збережені відповіді вже не ті, пропускаю.")
            continue
        today, cases = load_cases(CASES / f"{old['set']}.yaml")
        by_id = {c.id: c for c in cases}
        async with plast_trail_links() as links:
            results = [await rescore(by_id[r["id"]], r, today, links) for r in old["results"] if r["id"] in by_id]
        approximate = sum(1 for r in old["results"] if not r["error"] and r.get("raw") is None)
        run = {k: old[k] for k in ("set", "provider", "model", "today", "cases_total")} | {
            "label": args.label, "fingerprint": fingerprint(), "model_fingerprint": mfp, "commit": git_commit(),
            "started": datetime.now().isoformat(timespec="seconds"), "results": results,
            "replay_of": src.name, "replay_note": (f"{approximate} відповідей відтворено з очищених (sanitised), "
                                                   "бо сирі тоді ще не зберігались") if approximate else ""}
        out = RESULTS / src.name.replace(f"{args.replay}_", f"{args.label}_", 1)
        save(out, run)
        print("\n" + render_summary(run["summary"], f"{run['set']} · {args.label} (перерахунок {src.name})"))
        if run["set"] == "dev" or args.details:
            for r in results:
                print(f"  {r['id']}  {describe_errors(by_id[r['id']], r)}")
    write_report()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama")
    ap.add_argument("--model", help="override the model name (e.g. gemma4:12b for Ollama)")
    ap.add_argument("--set", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--label", default="baseline", help="what stage of the work this run measures")
    ap.add_argument("--ids", help="only these case ids, comma-separated (for debugging)")
    ap.add_argument("--reserve", type=int, default=15,
                    help="OpenRouter: stop when this many free requests are left today (default 15)")
    ap.add_argument("--details", action="store_true", help="print per-case mistakes for the test set too")
    ap.add_argument("--retry-errors", action="store_true", help="ask the model again where it failed last time")
    ap.add_argument("--report", action="store_true", help="only rebuild eval/RESULTS.md from saved runs")
    ap.add_argument("--replay", metavar="LABEL",
                    help="re-score the saved model answers of runs LABEL with the current code (no model calls)")
    args = ap.parse_args()
    if args.report:
        write_report()
        return
    if args.replay:
        sys.exit(asyncio.run(replay(args)))
    sys.exit(asyncio.run(evaluate(args)))


if __name__ == "__main__":
    main()
