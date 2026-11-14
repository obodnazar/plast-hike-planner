"""The model side: OpenRouter via the OpenAI-compatible ChatOpenAI, with structured outputs.

The agent asks the model for exactly three kinds of things, each with a typed schema:
  * ParsedRequest - read the leader's free-text note into fields;
  * Proposal      - choose a route, night spots and dates (re-asked with feedback);
  * a short conclusion paragraph for the plan note.
Every call is counted, because free models allow 50 requests a day.
"""

from __future__ import annotations

import asyncio

from datetime import date
from typing import Any, Literal

import openai
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from plast_agent.config import Settings


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


STRUCTURED_DEADLINE_S = 120   # parsing a note / proposing a plan
TEXT_DEADLINE_S = 45          # the optional closing paragraph


class LLM:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.openrouter_model
        self.chat = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
            timeout=75,
            max_retries=1,
            default_headers={"X-Title": "plast-hike-planner"},
        )
        # The closing paragraph is optional: it gets a short leash, the plan never waits for it.
        self.quick_chat = ChatOpenAI(
            model=settings.openrouter_model, api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url, temperature=0, timeout=45, max_retries=0,
            default_headers={"X-Title": "plast-hike-planner"},
        )
        self.calls = 0

    async def _deadline(self, runnable, messages, seconds: int):
        """A hard wall-clock limit. OpenRouter keeps slow free-model connections alive with whitespace,
        so the HTTP read timeout alone may never fire."""
        try:
            return await asyncio.wait_for(runnable.ainvoke(messages), timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise LLMError(f"Модель {self.model_name} не відповіла за {seconds} с. Спробуйте ще раз або вкажіть "
                           "іншу модель в OPENROUTER_MODEL.") from exc

    async def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        """Ask for a typed answer; if it breaks the schema or comes back empty, retry once with the reason."""
        runner = self.chat.with_structured_output(schema, method="function_calling")
        messages = [("system", system), ("user", user)]
        last_problem = ""
        for attempt in (1, 2):
            self.calls += 1
            try:
                result = await self._deadline(runner, messages, STRUCTURED_DEADLINE_S)
            except (ValidationError, OutputParserException) as exc:
                last_problem = f"{type(exc).__name__}: {str(exc)[:300]}"
                messages.append(("user", f"Your previous answer broke the schema: {last_problem}. "
                                         "Answer again by calling the function; use JSON null for missing values."))
                continue
            except openai.OpenAIError as exc:
                raise LLMError(explain_llm(exc, self.model_name)) from exc
            if result is None:  # the model answered in prose instead of calling the schema
                last_problem = "no function call in the answer"
                messages.append(("user", "Answer by calling the function with the schema, not in prose."))
                continue
            return result
        raise LLMError(f"Модель {self.model_name} двічі не дала відповіді за схемою ({last_problem}).")

    async def text(self, system: str, user: str) -> str:
        """Free text with a short deadline: callers must have a fallback."""
        self.calls += 1
        try:
            msg = await self._deadline(self.quick_chat, [("system", system), ("user", user)], TEXT_DEADLINE_S)
            return msg.content
        except openai.OpenAIError as exc:
            raise LLMError(explain_llm(exc, self.model_name)) from exc
