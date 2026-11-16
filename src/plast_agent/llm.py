"""The model side: OpenRouter via the OpenAI-compatible ChatOpenAI, with structured outputs.

The agent asks the model for exactly three kinds of things, each with a typed schema:
  * ParsedRequest - read the leader's free-text note into fields;
  * Proposal      - choose a route, night spots and dates (re-asked with feedback);
  * a short conclusion paragraph for the plan note.
Every call is counted, because free models allow 50 requests a day.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from datetime import date
from typing import Any, Literal

import httpx
import openai
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from plast_agent.config import PROJECT_ROOT, Settings


class Lenient(BaseModel):
    """Small models sometimes write "None"/"null"/"" for a missing value: treat those as null."""

    @model_validator(mode="before")
    @classmethod
    def _nullish(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: (None if isinstance(v, str) and v.strip().lower() in ("none", "null", "", "n/a") else v)
                    for k, v in data.items()}
        return data


class ParsedRequest(Lenient):
    """Facts from the leader's request note. Use null when the note does not say; the code then asks for them."""

    section: Literal["novatstvo", "yunatstvo", "senior"] | None = Field(
        description="novatstvo = новацтво/УПН/новаки/рій/гніздо or ages 6-10; yunatstvo = юнацтво/УПЮ/юнаки/юначки/"
                    "гурток/курінь or ages 11-17; senior = старші пластуни/УСП/УПС/сеніори, any group of adults or ages 18+")
    participants: int | None = Field(default=None, ge=1, le=60,
        description="Participants WITHOUT accompanying adults ('12 разом з 2 виховниками' -> 10). For a senior/adult "
                    "group: everybody in the group ('я із сестрою' -> 2)")
    adults: int | None = Field(default=None, ge=0, le=30,
        description="Accompanying adult leaders (виховники, провідники, батьки-супровід); 0 for a senior/adult group")
    days: int | None = Field(default=None, ge=1, le=7,
        description="Hiking days. 'N ночівель' -> N+1; a range '2-3 дні' -> the larger; 'вихідні' -> 2")
    start_date: date | None = Field(default=None,
        description="First day YYYY-MM-DD. No year -> the nearest future date; weekdays -> use the calendar given")
    date_flexibility_days: int = Field(default=0, ge=0, le=31,
        description="How many days the start may move LATER ('можемо зсунути на 1-2 дні' -> 2); 0 if not said")
    region: Literal["chornohora", "svydovets", "gorgany", "marmarosy", "yaremche-vorokhta"] | None = Field(
        default=None, description="Чорногора (also Говерла, Петрос, Піп Іван, Несамовите) -> chornohora; Свидовець "
        "(Близниця, Драгобрат) -> svydovets; Горгани (Сивуля, Висока, Ґорґан) -> gorgany; Мармароси -> marmarosy; "
        "Яремче, Ворохта, Маковиця -> yaremche-vorokhta; null if not said")
    must_include: list[Literal["lake", "peak", "water", "hut", "shelter", "campsite"]] | None = Field(
        default=None, description="Wished features: озеро -> lake, вершина -> peak, джерело -> water, "
        "хатина/колиба -> hut, притулок -> shelter, табірне місце -> campsite")
    start_time: str | None = Field(default=None, pattern=r"^\d{1,2}:\d{2}$",
        description="Daily start time H:MM or HH:MM ('не пізніше 8:00' -> '8:00'); null if not said")
    places: list[str] | None = Field(default=None,
        description="Named peaks, lakes or villages the group wants to visit or pass, nominative case, at most 3 "
                    "('піднятися на Говерлу, спуститися біля Петроса' -> ['Говерла', 'Петрос'])")
    max_km_per_day: float | None = Field(default=None, description="Group's own km/day limit, null if not said")
    max_ascent_per_day_m: int | None = Field(default=None, description="Group's own ascent limit per day, m")
    max_walking_hours_per_day: float | None = Field(default=None, description="Group's own walking-hours limit")
    max_overnight_elevation_m: int | None = Field(default=None, description="Highest allowed night spot, m")
    lightest_body_weight_kg: float | None = Field(default=None, description="Weight of the lightest participant, kg")
    participant_base_gear_kg: float | None = Field(default=None, description="Weighed participant rucksack without food/water, kg")
    adult_base_gear_kg: float | None = Field(default=None, description="Weighed adult rucksack without food/water, kg")


class Proposal(Lenient):
    """One concrete plan to test with the tools. The code checks every field before any tool call."""

    route_id: str = Field(description="Route id from the candidates, e.g. 'r8613349'")
    overnight_waypoint_ids: list[str] = Field(
        description="Night spots from that route's overnight options, in increasing km; exactly days-1 ids")
    start_date: date = Field(description="First day YYYY-MM-DD, within the allowed range")
    reasoning: str = Field(description="1-2 sentences in Ukrainian: why this route, these night spots and dates")


class LLMError(RuntimeError):
    """The model could not be used. The message is for the person and names the real cause."""


def explain_llm(exc: BaseException, model: str) -> str:
    if isinstance(exc, openai.RateLimitError):
        return (f"OpenRouter: ліміт запитів вичерпано (HTTP 429) для {model}. Безкоштовні моделі мають спільний "
                "ліміт 20/хв і 50/день на акаунт: зачекайте, поповніть кредит або вкажіть платну модель у OPENROUTER_MODEL.")
    if isinstance(exc, openai.AuthenticationError):
        return "OpenRouter відхилив ключ (HTTP 401). Перевірте OPENROUTER_API_KEY у .env."
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        return f"OpenRouter: бракує кредиту для {model} (HTTP 402). Поповніть баланс або оберіть безкоштовну модель."
    if isinstance(exc, openai.NotFoundError):
        return f"OpenRouter не знає модель {model} (HTTP 404). Перевірте OPENROUTER_MODEL у .env."
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return f"OpenRouter не відповів ({type(exc).__name__}). Перевірте інтернет і повторіть."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenRouter повернув HTTP {exc.status_code}: {str(exc)[:200]}"
    return f"Модель не дала коректної відповіді: {type(exc).__name__}: {str(exc)[:200]}"


QUOTA_FILE = PROJECT_ROOT / ".cache" / "openrouter_quota.json"
MIN_FREE_REQUESTS = 4   # a planning run needs about 2-5 model calls
DIM, YELLOW, RESET = "\033[2m", "\033[33m", "\033[0m"


@dataclass
class Provider:
    """One place a model can answer from. The agent tries them in order."""

    name: str                 # "openrouter" | "ollama"
    model: str
    chat: Any                 # for structured answers
    quick: Any                # for the short closing paragraph
    method: str               # "function_calling" (OpenRouter) | "json_schema" (Ollama: constrained decoding)
    structured_deadline: int  # seconds, wall clock
    text_deadline: int

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"


class ProviderFailed(RuntimeError):
    """This provider cannot answer now (quota, timeout, down, broken schema twice); try the next one."""

    def __init__(self, message: str, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def next_quota_reset(now: datetime | None = None) -> datetime:
    """OpenRouter's free-model day ends at 00:00 UTC (02:00 Kyiv)."""
    now = now or datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def quota_exhausted_now() -> bool:
    try:
        until = datetime.fromisoformat(json.loads(QUOTA_FILE.read_text(encoding="utf-8"))["exhausted_until"])
    except (OSError, ValueError, KeyError):
        return False
    return datetime.now(timezone.utc) < until


def remember_quota_exhausted() -> None:
    QUOTA_FILE.parent.mkdir(exist_ok=True)
    QUOTA_FILE.write_text(json.dumps({"exhausted_until": next_quota_reset().isoformat()}), encoding="utf-8")


def free_requests_left(settings: Settings) -> int | None:
    """Today's remaining free-model requests on this OpenRouter key, or None if unknown."""
    try:
        r = httpx.get("https://openrouter.ai/api/v1/key", timeout=5,
                      headers={"Authorization": f"Bearer {settings.openrouter_api_key}"})
        return int(r.json()["data"]["free_model_daily_requests"]["remaining"])
    except Exception:  # unknown is fine: the run then simply tries OpenRouter first
        return None


def ollama_status(settings: Settings) -> tuple[bool, str]:
    """(usable, message). Usable = the Ollama server answers and the configured model is installed."""
    root = settings.ollama_base_url.rstrip("/").removesuffix("/v1")
    try:
        tags = httpx.get(f"{root}/api/tags", timeout=3).json().get("models", [])
    except Exception:
        return False, ("Ollama не запущена або не встановлена: відкрийте застосунок Ollama "
                       "(встановити: https://ollama.com).")
    names = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
    if settings.ollama_model not in names:
        return False, (f"Ollama працює, але моделі {settings.ollama_model} немає. Створіть її: "
                       "ollama pull gemma4:e4b && ollama create plast-gemma -f ollama/Modelfile")
    return True, f"Ollama готова: {settings.ollama_model}"


def build_providers(settings: Settings, with_ollama: bool) -> list[Provider]:
    """OpenRouter if there is a key, then the local Ollama model. No key and no Ollama gives an empty list."""
    providers = [] if not settings.openrouter_api_key else [Provider(
        name="openrouter", model=settings.openrouter_model,
        chat=ChatOpenAI(model=settings.openrouter_model, api_key=settings.openrouter_api_key,
                        base_url=settings.openrouter_base_url, temperature=0, timeout=75, max_retries=1,
                        default_headers={"X-Title": "plast-hike-planner"}),
        quick=ChatOpenAI(model=settings.openrouter_model, api_key=settings.openrouter_api_key,
                         base_url=settings.openrouter_base_url, temperature=0, timeout=45, max_retries=0,
                         default_headers={"X-Title": "plast-hike-planner"}),
        method="function_calling", structured_deadline=120, text_deadline=45)]
    if with_ollama:
        # gemma4 "thinks" before answering by default: hundreds of hidden tokens, 20+ s, a hot laptop
        # and sometimes no answer within 4 min. reasoning_effort="none" switches that off (measured:
        # 0.4 s instead of 20 s for a short reading); a token cap stops a runaway answer (hut, hut, hut...).
        local = dict(model=settings.ollama_model, api_key="ollama", base_url=settings.ollama_base_url,
                     temperature=0, max_retries=0, reasoning_effort="none")
        providers.append(Provider(
            name="ollama", model=settings.ollama_model,
            # langchain sends max_tokens as `max_completion_tokens`, which Ollama ignores: pass Ollama's own field
            chat=ChatOpenAI(timeout=180, extra_body={"max_tokens": 700}, **local),
            quick=ChatOpenAI(timeout=90, extra_body={"max_tokens": 900}, **local),
            # Ollama has no tool_choice: function_calling may return nothing. json_schema goes into
            # Ollama's `format` and constrains decoding, so the answer is always valid JSON.
            method="json_schema", structured_deadline=180, text_deadline=90))
    return providers


def describe_failure(exc: BaseException, p: Provider) -> ProviderFailed:
    if p.name == "ollama":
        if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError)):
            return ProviderFailed("Ollama не відповідає: чи запущений застосунок Ollama?")
        if isinstance(exc, openai.NotFoundError):
            return ProviderFailed(f"В Ollama немає моделі {p.model}: ollama create plast-gemma -f ollama/Modelfile")
        return ProviderFailed(f"Ollama: {type(exc).__name__}: {str(exc)[:200]}")
    return ProviderFailed(explain_llm(exc, p.model), quota=isinstance(exc, openai.RateLimitError))


class LLM:
    """Model calls with a provider chain: OpenRouter first, the local Ollama model when OpenRouter cannot answer."""

    def __init__(self, settings: Settings | None = None, providers: list[Provider] | None = None,
                 start_with_fallback: bool = False) -> None:
        self.providers = providers if providers is not None else build_providers(settings, settings.ollama_enabled)
        self.active = 1 if start_with_fallback and len(self.providers) > 1 else 0
        self.calls = 0
        self.calls_by: dict[str, int] = {}
        self.switches: list[str] = []

    @property
    def provider(self) -> Provider:
        return self.providers[self.active]

    @property
    def model_name(self) -> str:
        used = [p.label for p in self.providers if self.calls_by.get(p.label)]
        return ", ".join(used) or self.provider.label

    def _switch(self, reason: ProviderFailed) -> bool:
        if reason.quota:
            remember_quota_exhausted()
        if self.active + 1 >= len(self.providers):
            return False
        self.active += 1
        note = f"{reason} → перемикаюсь на {self.provider.label}"
        self.switches.append(note)
        print(f"    {YELLOW}↪ {note}{RESET}")
        return True

    def _count(self, p: Provider) -> None:
        self.calls += 1
        self.calls_by[p.label] = self.calls_by.get(p.label, 0) + 1

    async def _deadline(self, p: Provider, runnable, messages, seconds: int):
        """A hard wall-clock limit. OpenRouter keeps slow free-model connections alive with whitespace,
        so the HTTP read timeout alone may never fire."""
        try:
            return await asyncio.wait_for(runnable.ainvoke(messages), timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise ProviderFailed(f"{p.label} не відповіла за {seconds} с") from exc

    async def _structured_once(self, p: Provider, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if p.method == "json_schema":
            # Ollama: a plain JSON schema, checked by pydantic below. The strict form makes all 17 fields
            # required and gemma4 then loops on whitespace up to the token cap (measured: 11 of 30 readings
            # failed; with the plain schema the same readings take 3-4 s).
            runner = p.chat.with_structured_output(schema.model_json_schema(), method="json_schema", strict=False)
        else:
            runner = p.chat.with_structured_output(schema, method=p.method)
        messages = [("system", system), ("user", user)]
        last_problem = ""
        for _ in (1, 2):  # one retry with the reason if the answer breaks the schema
            self._count(p)
            try:
                result = await self._deadline(p, runner, messages, p.structured_deadline)
                if isinstance(result, dict):
                    result = schema.model_validate(result)
            except (ValidationError, OutputParserException) as exc:
                last_problem = f"{type(exc).__name__}: {str(exc)[:300]}"
                messages.append(("user", f"Your previous answer broke the schema: {last_problem}. "
                                         "Answer again following the schema; use JSON null for missing values."))
                continue
            except ProviderFailed:
                raise
            except Exception as exc:
                raise describe_failure(exc, p) from exc
            if result is None:
                last_problem = "no structured answer"
                messages.append(("user", "Answer with the structured object only, not in prose."))
                continue
            return result
        raise ProviderFailed(f"{p.label} двічі не дала відповіді за схемою ({last_problem})")

    async def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        """A typed answer from the first provider that can give one."""
        while True:
            try:
                return await self._structured_once(self.provider, schema, system, user)
            except ProviderFailed as exc:
                if not self._switch(exc):
                    raise LLMError(str(exc)) from exc

    async def text(self, system: str, user: str) -> str:
        """Free text with a short deadline: callers must have a fallback. Uses the current provider only."""
        p = self.provider
        self._count(p)
        try:
            msg = await self._deadline(p, p.quick, [("system", system), ("user", user)], p.text_deadline)
            return msg.content
        except ProviderFailed as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(describe_failure(exc, p))) from exc
