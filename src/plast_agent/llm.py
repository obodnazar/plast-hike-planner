"""The model side: OpenRouter via the OpenAI-compatible ChatOpenAI, with structured outputs.

The agent asks the model for exactly three kinds of things, each with a typed schema:
  * ParsedRequest - read the leader's free-text note into fields;
  * Proposal      - choose a route, night spots and dates (re-asked with feedback);
  * a short conclusion paragraph for the plan note.
Every call is counted, because free models allow 50 requests a day.
"""

from __future__ import annotations

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
    """Facts from the leader's request note. Use null when the note does not say."""

    section: Literal["novatstvo", "yunatstvo", "senior"] = Field(
        description="novatstvo = новацтво (6-11), yunatstvo = юнацтво (11-18), senior = старші/сеніори (18+)")
    participants: int = Field(ge=1, le=60, description="Number of participants of the section (not adults)")
    adults: int = Field(ge=0, le=30, description="Number of adult leaders (виховники, провідники)")
    days: int = Field(ge=1, le=7, description="Number of hiking days")
    start_date: date = Field(description="First day, YYYY-MM-DD")
    date_flexibility_days: int = Field(ge=0, le=3, description="How many days the start may move later; 0 if not said")
    region: Literal["chornohora", "svydovets", "gorgany", "marmarosy", "yaremche-vorokhta"] | None = Field(
        description="Mountain area: Чорногора, Свидовець, Горгани, Мармароси, Яремче/Ворохта; null if not said")
    must_include: list[Literal["lake", "peak", "water", "hut", "shelter", "campsite"]] = Field(
        description="Wished features, e.g. lake for 'озеро'")
    start_time: str | None = Field(description="Daily start time HH:MM, e.g. '08:00'; null if not said")
    max_km_per_day: float | None = Field(description="Group's own km/day limit, null if not said")
    max_ascent_per_day_m: int | None = Field(description="Group's own ascent limit per day, null if not said")
    max_walking_hours_per_day: float | None = Field(description="Group's own walking-hours limit, null if not said")
    max_overnight_elevation_m: int | None = Field(description="Highest allowed night spot in metres, null if not said")
    lightest_body_weight_kg: float | None = Field(description="Weight of the lightest participant, null if not said")
    participant_base_gear_kg: float | None = Field(description="Weighed participant rucksack without food/water")
    adult_base_gear_kg: float | None = Field(description="Weighed adult rucksack without food/water")


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


class LLM:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.openrouter_model
        self.chat = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
            timeout=90,
            max_retries=2,
            default_headers={"X-Title": "plast-hike-planner"},
        )
        self.calls = 0

    def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        """Ask for a typed answer; if it breaks the schema or comes back empty, retry once with the reason."""
        runner = self.chat.with_structured_output(schema, method="function_calling")
        messages = [("system", system), ("user", user)]
        last_problem = ""
        for attempt in (1, 2):
            self.calls += 1
            try:
                result = runner.invoke(messages)
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

    def text(self, system: str, user: str) -> str:
        self.calls += 1
        try:
            return self.chat.invoke([("system", system), ("user", user)]).content
        except openai.OpenAIError as exc:
            raise LLMError(explain_llm(exc, self.model_name)) from exc
