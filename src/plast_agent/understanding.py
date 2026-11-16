"""Code that checks and completes what the model understood from a leader's free text.

The model reads the text once (llm.ParsedRequest). Whatever the words themselves decide is then
checked here, by plain code: it costs nothing, answers the same every time and is easy to explain.
  * refine()  - after the model: wishes, region, places, people counts, group consistency, missing dates;
  * parse_*() - short answers in the chat dialogue, so an answer needs no new model call.
The effect is measured on eval/cases (eval/RESULTS.md: baseline vs stage1).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any


def norm(text: str) -> str:
    """Lower case, ґ -> г, one kind of apostrophe: so a pattern is written once."""
    t = text.lower().replace("ґ", "г").replace("ё", "е")
    for a in "’ʼ`´":
        t = t.replace(a, "'")
    return t


B, E = r"(?<![\w'])", r"(?![\w'])"   # word start / word end (the apostrophe belongs to the word: п'ять)

# ------------------------------------------------------------------ numbers
NUMBER_WORDS = {
    "один": 1, "одна": 1, "одне": 1, "одного": 1, "одну": 1, "одним": 1, "одною": 1,
    "два": 2, "дві": 2, "двоє": 2, "двох": 2, "двома": 2, "две": 2, "двое": 2, "пару": 2, "пара": 2,
    "три": 3, "троє": 3, "трьох": 3, "трьома": 3, "трое": 3,
    "чотири": 4, "четверо": 4, "чотирьох": 4, "четыре": 4,
    "п'ять": 5, "п'ятеро": 5, "п'ятьох": 5, "пять": 5, "пятеро": 5,
    "шість": 6, "шестеро": 6, "шести": 6, "шесть": 6,
    "сім": 7, "семеро": 7, "семи": 7, "семь": 7,
    "вісім": 8, "восьмеро": 8, "восьми": 8, "восемь": 8,
    "дев'ять": 9, "дев'ятеро": 9, "девять": 9,
    "десять": 10, "десятеро": 10, "десяти": 10,
    "одинадцять": 11, "одинадцятеро": 11, "дванадцять": 12, "дванадцятеро": 12,
    "тринадцять": 13, "тринадцятеро": 13, "чотирнадцять": 14, "п'ятнадцять": 15, "шістнадцять": 16,
    "сімнадцять": 17, "вісімнадцять": 18, "дев'ятнадцять": 19, "двадцять": 20,
}
NUM = r"(\d{1,3}|" + "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)) + r")"


def to_int(token: str) -> int | None:
    return int(token) if token.isdigit() else NUMBER_WORDS.get(token)


# ------------------------------------------------------------------ people
PEOPLE = (r"(?:юнак\w*|юнач\w*|діт(?:ей|и|ьми)|дитин\w*|учасни\w*|осіб|особи|особа|чол(?:ов\w*|\.|" + E +
          r")|чєл\w*|чел\w*|люд(?:ей|и|ини)|дівч\w*|хлопц\w*|хлопч\w*|новак\w*|новач\w*|пластун\w*)")
ADULTS = r"(?:виховни\w*|доросл\w*|взросл\w*|провідни\w*|батьк\w*|батьків|інструктор\w*|супровід\w*)"
# the number after a noun is a count only if it is not an age or a measure («юнаків 13 років», «учасник — 5 кг»)
NOT_AGE = (r"(?!\s*[-–—]\s*\d)(?!\s*(?:і|та|,)\s*\d+\s*(?:рок|рік))(?!\s*(?:рок|рік|лєт|лет|р\.))"
           r"(?!\s*(?:кг|км|м|год|л|%)(?![\w']))")
COUNT_BEFORE = re.compile(B + NUM + E + r"\s+" + PEOPLE)            # «10 юнаків», «двоє дітей»
COUNT_AFTER = re.compile(B + PEOPLE + r"\s*[:\-—]?\s*" + NUM + E + NOT_AGE)   # «Новачків 10», not «юнаків 13 років»
TOGETHER = re.compile(B + NUM + E + r"\s+(?:людей\s+|осіб\s+)?(?:разом|включно)\s+(?:з|із|зі)\s+" + NUM + E +
                      r"\s+" + ADULTS)                               # «12 разом з двома виховниками»
NO_ADULTS = re.compile(r"без\s+(?:доросл|супровод|виховник|провідник)|" + B + r"0\s+доросл|дорослих\s*[:\-—]?\s*0" + E)


ADULT_BEFORE = re.compile(B + NUM + E + r"\s+" + ADULTS + r"(?:\s+" + ADULTS + r")?")   # «2 дорослі виховники»
ADULT_AFTER = re.compile(B + ADULTS + r"\s*[:\-—]?\s*" + NUM + E + NOT_AGE)                # «дорослих 2», «батьків 4»
ADULT_WORD = re.compile(B + ADULTS + r"|" + B + r"(?:інструктор|мам|тат)\w*")
AND_ME = re.compile(PEOPLE + r"\s+(?:і|та|плюс)\s+я" + E)                                # «8 юнаків і я»
ME_WITH_GROUP = re.compile(B + r"я\s+(?:і|та|з|із|зі)\s+(?:сво\w+\s+)?(?:юнацтв|юнак|юнач|новацтв|новак|новач|гурт|ро[єю]|"
                           r"діть|дітьми|хлопц|дівч)\w*")                       # «я із юнацтвом»: the leader is the adult
PLUS = re.compile(r"(?<![\d.,:])(\d{1,3})\s*\+\s*(\d{1,2})(?!\d|[.,:]\d)")          # «10 + 2»: children + adults
ALONE = re.compile(B + r"я\s+(?:сам|сама|одна|один)" + E + r"|" + B + r"сам\s+іду" + E)


def _past_spans(t: str) -> list[tuple[int, int]]:
    """Sentences about earlier trips («Минулого року … було 20 людей»): their numbers are history."""
    spans, start = [], 0
    for part in re.split(r"(?<=[.!?\n])", t):
        if PAST.search(part):
            spans.append((start, start + len(part)))
        start += len(part)
    return spans


def present_text(text: str) -> str:
    """The text without sentences about earlier trips."""
    t = norm(text)
    for a, b in reversed(_past_spans(t)):
        t = t[:a] + " " + t[b:]
    return t


def people_count(text: str) -> tuple[int | None, int | None]:
    """(participants, adults) that the words state without doubt; None where they don't."""
    t = norm(text)
    past = _past_spans(t)
    now = lambda m: not any(a <= m.start() < b for a, b in past)   # noqa: E731
    m = TOGETHER.search(t)
    if m:
        total, adults = to_int(m.group(1)), to_int(m.group(2))
        if total and adults is not None and total > adults:
            return total - adults, adults
    m = PLUS.search(t)
    if m and not list(ADULT_WORD.finditer(t)):
        return int(m.group(1)), int(m.group(2))
    found = {to_int(m.group(1)) for rx in (COUNT_BEFORE, COUNT_AFTER) for m in rx.finditer(t) if now(m)} - {None}
    participants = found.pop() if len(found) == 1 else (1 if not found and ALONE.search(t) else None)
    # adults only when every adult word in the text is part of one counted phrase («інструктор і одна мама» is not «1»)
    phrases = [m for rx in (ADULT_BEFORE, ADULT_AFTER) for m in rx.finditer(t) if now(m)]
    counts = {to_int(m.group(1)) for m in phrases} - {None}
    covered = all(any(p.start() <= w.start() < p.end() for p in phrases) for w in ADULT_WORD.finditer(t))
    adults = counts.pop() if len(counts) == 1 and covered else None
    if adults is None and not phrases and not list(ADULT_WORD.finditer(t)) and (AND_ME.search(t) or ME_WITH_GROUP.search(t)):
        adults = 1
    if adults is None and not phrases and NO_ADULTS.search(t):   # «без дорослих», «дорослих: 0»
        adults = 0
    return participants, adults


def people_candidates(text: str) -> set[int]:
    """Every number the text puts next to people («60 учасників», «9 осіб»): when they disagree, code cannot
    choose, but a model reading that picked one of them is backed by the text."""
    t = norm(text)
    return {to_int(m.group(1)) for rx in (COUNT_BEFORE, COUNT_AFTER) for m in rx.finditer(t)} - {None}


NAS = re.compile(B + r"нас\s+(?:буде\s+|будє\s+|усього\s+|всього\s+)?" + NUM + E)          # «нас троє», «нас 5»
ME_WITH_N = re.compile(B + r"я\s+(?:і|та|з|із|зі)\s+" + NUM + E + r"\s+(?:друз|подруг|товариш|колег)\w*")
ME_WITH_ONE = re.compile(B + r"я\s+(?:і|та|з|із|зі)\s+(?:сестр|брат|друг|подруг|товариш)\w*")


def people_fallback(text: str) -> int | None:
    """How many people go, from phrases about the whole party, when nothing else said it: «нас троє друзів»
    -> 3, «я з двома друзями» -> 3, «я із сестрою» -> 2. Only fills an empty value, never overrides."""
    t = norm(text)
    m = NAS.search(t)
    if m:
        return to_int(m.group(1))
    m = ME_WITH_N.search(t)
    if m and to_int(m.group(1)):
        return to_int(m.group(1)) + 1
    return 2 if ME_WITH_ONE.search(t) else None


RANGE = re.compile(B + NUM + r"\s*(?:[-–—]|або|чи)\s*" + NUM + E)


def parse_count(answer: str) -> int | None:
    """A short answer with one number: «2», «я думаю 1 буду», «двоє». A range («6-7», «шість або сім») counts as
    its larger end: for people that is the safe side (enough food, enough adults)."""
    t = norm(answer)
    rng = RANGE.search(t)
    if rng:
        low, high = to_int(rng.group(1)), to_int(rng.group(2))
        if low is not None and high is not None and low < high:
            t = t[:rng.start()] + str(high) + t[rng.end():]
    numbers = {to_int(m.group(1)) for m in re.finditer(B + NUM + E, t)} - {None}
    return numbers.pop() if len(numbers) == 1 else None


# ------------------------------------------------------------------ section
# Words that support each group. The model's group is kept only if the text supports THAT group.
SECTION_WORDS = {
    "novatstvo": r"новац|новак|новач|упн|рій|роєм|рою|гнізд|діт|дитин|мал",
    "yunatstvo": r"юнац|юнак|юнач|упю|гурт|курін|підліт|діт|хлопц|дівч|мал",
    "senior": r"старш|сеньйор|сеніор|сенйор|упс|усп|доросл|взросл|друз|дружин|чоловік|сестр|брат|родин|сам",
}
AGES = re.compile(r"(?:по|за|всім|усім)\s+(\d{1,2})" + E + r"|" + B + r"(\d{1,2})(?:\s*[-–—]\s*\d{1,2})?\s*(?:рок|рік|лєт|лет)")


# Ages for deciding the group: «по 14», «усім за 30», «13–15 років», but not a date («з 27 по 29 листопада»).
AGES_STATED = re.compile(r"(?:по|за|всім|усім)\s+(\d{1,2})" + E + r"(?!\s*(?:-?(?:го|ого)\s+)?(?:січ|лют|берез|квіт|трав|черв|"
                         r"лип|серп|верес|жовт|листоп|груд|числ|км|кіломет|дн|год|хв))|" + B +
                         r"(\d{1,2})(?:\s*[-–—]\s*\d{1,2})?\s*(?:рок|рік|лєт|лет)")


def age_section(text: str) -> str | None:
    """The group the stated ages point to («13–15 років» -> юнацтво), if they all point to one."""
    t = present_text(text)
    found = set()
    for m in AGES_STATED.finditer(t):
        age = int(m.group(1) or m.group(2))
        if 6 <= age <= 10:
            found.add("novatstvo")
        elif 11 <= age <= 17:
            found.add("yunatstvo")
        elif age >= 18:
            found.add("senior")
    return found.pop() if len(found) == 1 else None


CHILD_WORDS = re.compile(B + "(?:" + SECTION_WORDS["novatstvo"] + "|" + SECTION_WORDS["yunatstvo"] + ")")


def section_supported(text: str, section: str) -> bool:
    t = norm(text)
    if age_section(text) == section:
        return True
    if section == "senior" and CHILD_WORDS.search(t):
        # «діти … і 2 дорослі»: the adults go with the children; senior needs its own words, not «дорослі»
        return bool(re.search(B + r"(?:старш|сеньйор|сеніор|сенйор|упс|усп|друз|дружин|чоловік|сестр|брат|родин)", t)) \
            and age_section(text) in (None, "senior")
    if re.search(B + "(?:" + SECTION_WORDS[section] + ")", t):
        return True
    for m in AGES.finditer(t):
        age = int(m.group(1) or m.group(2))
        if (section == "novatstvo" and 6 <= age <= 10) or (section == "yunatstvo" and 11 <= age <= 17) or \
                (section == "senior" and age >= 18):
            return True
    return False


# ------------------------------------------------------------------ region
SECTION_NAME = {"novatstvo": "новацтво", "yunatstvo": "юнацтво", "senior": "старші"}
REGION_UA = {"chornohora": "Чорногора", "svydovets": "Свидовець", "gorgany": "Горгани", "marmarosy": "Мармароси",
             "yaremche-vorokhta": "Яремче–Ворохта"}
# Cyrillic only: a Latin "region to marmarosy" inside the note is more likely an injected instruction than a wish.
REGION_NAMES = {"chornohora": ("чорногор", "черногор"), "svydovets": ("свидов",),
                "gorgany": ("горган",), "marmarosy": ("мармарос", "мармарош")}
REGION_HINTS = {
    "chornohora": ("говерл", "петрос", "піп іван", "піпіван", "піп-іван", "несамовит", "бребенеск", "туркул", "шпиц",
                   "брескул", "пожижевськ", "заросляк", "дземброн", "смотрич", "кукул", "маришевськ"),
    "svydovets": ("близниц", "драгобрат", "трояск"),
    "gorgany": ("сивул", "висок", "грофа", "парашк", "хом'як", "синяк", "довбушанк", "ігровец"),
    "marmarosy": ("мармароськ",),
    "yaremche-vorokhta": ("яремч", "ворохт", "маковиц", "буковел", "микуличин", "татарів"),
}


def regions_named(text: str) -> set[str]:
    t = norm(text)
    return {r for r, words in REGION_NAMES.items() if any(w in t for w in words)}


def region_hinted(text: str, region: str) -> bool:
    t = norm(text)
    return any(w in t for w in REGION_NAMES.get(region, ()) + REGION_HINTS.get(region, ()))


# ------------------------------------------------------------------ places
PLACE_PREFIXES = ("г.", "гора ", "оз.", "озеро ", "с.", "село ", "смт ", "пол.", "полонина ", "пер.", "вершина ")
GENERIC_PLACES = {"карпати", "карпатах", "гори", "гора", "озеро", "вершина", "полонина", "ліс", "табір", "похід",
                  "мандрівка", "гірське озеро", "гірське", "джерело", "колиба", "притулок"}


def stems(place: str) -> list[str]:
    """The server's matching rule (plast_trail.routes.place_stems), repeated here so the agent does not
    import the server's code: 'Говерлу' -> ['говер'], 'Піп Іван' -> ['піп', 'іван']."""
    t = norm(place).strip()
    for pre in PLACE_PREFIXES:
        if t.startswith(pre):
            t = t[len(pre):].strip()
    words = [w for w in t.replace("-", " ").split() if len(w) >= 3]
    return [w if len(w) <= 4 else w[: max(4, len(w) - (2 if len(w) >= 7 else 1))] for w in words]


PLACE_AFTER = re.compile(B + r"(?:на|до|через|біля|повз|під|від|поблизу|коло)\s+"
                         r"(?:(?:оз\.|озер\w*|гор[ауи]|г\.|пол\.|полонин\w*|с\.|сел[оа])\s+)?"
                         r"([а-яіїєґ][а-яіїєґ'\-]{2,})(?=(?:\s+([а-яіїєґ][а-яіїєґ'\-]{2,}))?)")   # 2nd word: peek only
PLACE_KIND_FIRST = re.compile(B + r"(?:озеро|озера|оз\.|гора|гору|г\.|вершин[аиуі]|полонин[аиуі]|пол\.)\s+"
                              r"([а-яіїєґ][а-яіїєґ'\-]{2,})")   # «озеро Несамовите» with no preposition
NOT_A_PLACE = ("вихідн", "пару", "ден", "дні", "дня", "днів", "тижд", "субот", "неділ", "п'ятниц", "понеділ",
               "вівтор", "серед", "четвер", "джерел", "вод", "озер", "вершин", "гор", "ліс", "табір", "колиб",
               "хатин", "притул", "стоянк", "ночів", "маршрут", "похід", "мандрівк", "сніг", "дорог", "стежк",
               "річк", "карпат", "чорногор", "черногор", "горган", "свидов", "мармарос", "сестр", "друз", "брат",
               "діт", "гурт", "курін", "рій", "роєм", "юнак", "юнач", "новак", "новач", "нас", "них", "мене",
               "себе", "все", "всі", "місц", "супров", "захід", "схід", "тиждень", "ранок", "вечір", "свят",
               "відпуст", "канікул", "природ", "пікнік", "одн", "дв", "три", "чотир", "п'ят", "двох", "ночі")


def place_candidates(text: str) -> list[str]:
    """Words after «на / до / через / біля…» that may be a place the model did not list: «хочемо на Близницю»
    -> «Близницю». plast-trail.resolve_places then keeps only confident finds. Sentences about earlier trips
    are skipped («минулого року ходили на Петрос»)."""
    found: list[str] = []
    now = present_text(text)
    matches = sorted([(m.start(), m.group(1), m.group(2)) for m in PLACE_AFTER.finditer(now)] +
                     [(m.start(), m.group(1), None) for m in PLACE_KIND_FIRST.finditer(now)])
    for _, first, second in matches:
        if first.startswith(NOT_A_PLACE) or first in NUMBER_WORDS:
            continue
        name = f"{first} {second}" if second and len(first) <= 4 and not second.startswith(NOT_A_PLACE) else first
        if name not in found:
            found.append(name)
    return found[:3]


def clean_places(places: list[str], text: str) -> tuple[list[str], list[str]]:
    """Keep real place names that are in the text. Returns (places, what was dropped and why)."""
    squashed = re.sub(r"[\s\-'.]", "", norm(text))
    kept, dropped, seen = [], [], set()
    for p in places:
        pn = norm(p).strip()
        key = tuple(stems(p))
        if not re.search(r"[а-яіїє]", pn):
            dropped.append(f"«{p}» (не українська назва)")
        elif any(w in pn for words in REGION_NAMES.values() for w in words):
            dropped.append(f"«{p}» (це район, а не місце)")
        elif pn in GENERIC_PLACES or not key:
            dropped.append(f"«{p}» (загальне слово, не назва)")
        elif not all(s.replace("'", "") in squashed for s in key):
            dropped.append(f"«{p}» (немає в тексті)")
        elif key not in seen:
            seen.add(key)
            kept.append(p.strip())
    return kept[:3], dropped


# ------------------------------------------------------------------ wishes
FEATURES = {
    "lake": r"озер|озір|оз\.",
    "peak": r"вершин",
    "water": r"джерел|криниц|струмк|струмок|річк|(?:біля|поруч|коло|поблизу|з|із|до)\s+вод(?:и|ою|а)" + E,
    "hut": r"колиб|колыб|хатин|хижк",
    "shelter": r"притул",
    "campsite": r"табірн\w*\s+місц|табірне|стоянк|кемпінг",
}
FEATURE_UA = {"lake": "озеро", "peak": "вершина", "water": "вода поруч", "hut": "колиба", "shelter": "притулок",
              "campsite": "табірне місце"}


def wished_features(text: str) -> list[str]:
    t = norm(text)
    return [f for f, pattern in FEATURES.items() if re.search(B + "(?:" + pattern + ")", t)]


# ------------------------------------------------------------------ dates and days
MONTHS = {"січ": 1, "лют": 2, "берез": 3, "квіт": 4, "трав": 5, "черв": 6, "лип": 7, "серп": 8, "верес": 9,
          "жовт": 10, "лист": 11, "груд": 12, "январ": 1, "феврал": 2, "марта": 3, "апрел": 4, "мая": 5,
          "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}
MONTH = r"(" + "|".join(MONTHS) + r")\w*"
WEEKDAYS = {0: r"понеділ\w*|понедельник", 1: r"вівтор\w*|вторник", 2: r"середу|середа|среду",
            3: r"четвер|четверг", 4: r"п'ятниц\w*|пятниц\w*", 5: r"субот\w*|суббот\w*", 6: r"неділ\w*|воскресен\w*"}
DAY_SUFFIX = r"(?:\s*-?\s*(?:го|ого|е))?"
SHIFT = re.compile(r"зсун|посун|перен|зміст|через" + E)
FLEX_WORDS = re.compile(r"зсун|посун|перен|зміст|пізніше|гнучк|±|\+/-|не\s+жорстк")
PAST = re.compile(r"минул|торік|вже\s+бул|ходили|були\s+на")


def _month_of(word: str) -> int:
    return next(v for k, v in MONTHS.items() if word.startswith(k))


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _nearest(today: date, day: int, month: int, year: int | None = None) -> date | None:
    if year:
        return _safe(year + 2000 if year < 100 else year, month, day)
    d = _safe(today.year, month, day)
    if d and d < today:
        d = _safe(today.year + 1, month, day)
    return d


def _day_only(today: date, day: int) -> date | None:
    d = _safe(today.year, today.month, day)
    if d is None or d < today:
        nxt = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        d = _safe(nxt.year, nxt.month, day)
    return d


def next_weekday(today: date, weekday: int) -> date:
    return today + timedelta(days=(weekday - today.weekday()) % 7 or 7)


def parse_date(text: str, today: date) -> date | None:
    """The first day of the trip as people write it: «21.11», «21 листопада», «з 27 по 29 листопада»,
    «28-го», «в суботу», «завтра», «через тиждень», «на вихідних». None if the text has no date."""
    t = norm(text)
    m = re.search(r"(?<!\d)(\d{1,2})\s*[-–]\s*\d{1,2}\s+" + MONTH, t) or \
        re.search(r"з\s+(\d{1,2})" + DAY_SUFFIX + r"\s+(?:по|до)\s+\d{1,2}" + DAY_SUFFIX + r"\s+" + MONTH, t)
    if m:
        return _nearest(today, int(m.group(1)), _month_of(m.group(2)))
    m = re.search(r"(?<![\d.,])(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)(?![.,]\d)(?!\s*(?:км|м|год|кг|л)" + E + ")", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return _nearest(today, int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)
    m = re.search(r"(?<!\d)(\d{1,2})" + DAY_SUFFIX + r"\s+" + MONTH, t)
    if m:
        return _nearest(today, int(m.group(1)), _month_of(m.group(2)))
    m = re.search(r"(?<!\d)(\d{1,2})\s*-?\s*(?:го|ого)" + E + r"|(?<!\d)(\d{1,2})\s+числа", t)
    if m:
        return _day_only(today, int(m.group(1) or m.group(2)))
    for word, shift in (("післязавтра", 2), ("послезавтра", 2), ("завтра", 1), ("сьогодні", 0), ("сегодня", 0)):
        if re.search(B + word + E, t):
            return today + timedelta(days=shift)
    m = re.search(r"через\s+" + NUM + E + r"\s+(дн|тиж)", t)
    if m:
        n = to_int(m.group(1)) or 0
        return today + timedelta(days=n * (7 if m.group(2) == "тиж" else 1))
    if re.search(r"через\s+тиждень", t):
        return today + timedelta(days=7)
    for weekday, pattern in WEEKDAYS.items():
        if re.search(B + "(?:" + pattern + ")" + E, t):
            return next_weekday(today, weekday)
    if re.search(B + r"вихідн|выходн", t):
        return next_weekday(today, 5)
    return None


def stated_times(text: str) -> list[str]:
    """Start times written in the text, as HH:MM: «не пізніше 8:00», «о 7:30», «о 7», «7 ранку». Only 04:00-12:59:
    «збір о 20:00» is a meeting, not the morning start, and a stray «01:29» is never a hiking start."""
    out: list[str] = []
    t = norm(text)
    for m in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)|" + B + r"о\s+(\d{1,2})" + E + r"(?!\s*[:.]?\s*\d)|"
                         r"(?<!\d)(\d{1,2})\s*(?:год\w*\s+)?ранку", t):
        h = int(m.group(1) or m.group(3) or m.group(4))
        mins = int(m.group(2) or 0)
        if 4 <= h <= 12 and 0 <= mins < 60 and f"{h:02d}:{mins:02d}" not in out:
            out.append(f"{h:02d}:{mins:02d}")
    return out


def explicit_date(text: str, today: date) -> date | None:
    """Only a date written with a day and a month («12.12.2026», «21 листопада»): it beats the model's reading."""
    t = norm(text)
    if any(PAST.search(part) and re.search(r"\d", part) for part in re.split(r"(?<=[.!?\n])", t)
           if re.search(MONTH, part) or re.search(r"\d{1,2}[./]\d{1,2}", part)):
        return None   # a date in a sentence about an earlier trip: leave it to the model
    m = re.search(r"(?<!\d)(\d{1,2})\s*[-–]\s*\d{1,2}\s+" + MONTH, t) or \
        re.search(r"з\s+(\d{1,2})" + DAY_SUFFIX + r"\s+(?:по|до)\s+\d{1,2}" + DAY_SUFFIX + r"\s+" + MONTH, t)
    if m:
        return _nearest(today, int(m.group(1)), _month_of(m.group(2)))
    m = re.search(r"(?<![\d.,])(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)(?![.,]\d)(?!\s*(?:км|м|год|кг|л)" + E + ")", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return _nearest(today, int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)
    m = re.search(r"(?<!\d)(\d{1,2})" + DAY_SUFFIX + r"\s+" + MONTH, t)
    return _nearest(today, int(m.group(1)), _month_of(m.group(2))) if m else None


def parse_shift(text: str) -> int | None:
    """How many days the start may move: «зсунути на 1–2 дні» -> 2, «на день пізніше» -> 1, «на тиждень» -> 7."""
    t = norm(text)
    m = re.search(r"(?:зсун|посун|перен|зміст)\w*[^.]{0,25}?на\s+(" + NUM[1:-1] + r"|день|тиждень)"
                  r"(?:\s*[-–]\s*(" + NUM[1:-1] + r"))?", t)
    if not m:
        m = re.search(r"на\s+(" + NUM[1:-1] + r"|день|тиждень)()(?:\s*дн\w*)?\s+пізніше", t)
    if not m:
        return None
    first, second = m.group(1), m.group(2)
    n = 1 if first == "день" else 7 if first == "тиждень" else to_int(first) or 0
    return max(n, to_int(second) or 0) if second else n


# Group limits, each only with its own words around the number, so one number is never taken for another.
LIMITS = {
    "max_km_per_day": re.compile(r"(?:не\s+більше|до|максимум|ліміт\w*)\s+(\d{1,2}(?:[.,]\d)?)\s*(?:км|кілометр)|"
                                 r"(\d{1,2}(?:[.,]\d)?)\s*(?:км|кілометр)\w*\s+(?:на|за|в)\s+день"),
    "max_ascent_per_day_m": re.compile(r"(\d{3,4})\s*(?:м|метр)\w*\s+(?:набор|набір|підйом|набра)|"
                                       r"(?:набор|набір|підйом)\w*\s+(?:до|не більше|максимум)\s+(\d{3,4})"),
    "max_walking_hours_per_day": re.compile(r"(\d{1,2})\s*(?:год|годин)\w*\s+ход|ход\w*\s+(?:до|не більше)\s+(\d{1,2})\s*год"),
    "max_overnight_elevation_m": re.compile(r"(?:ночув|ночівл)\w*[^.]{0,20}?(?:не\s+вище|нижче|до)\s+(\d{3,4})\s*м"),
    "lightest_body_weight_kg": re.compile(r"(?:найлегш)\w*[^.]{0,30}?(\d{2,3})\s*кг"),
}
LIMIT_UA = {"max_km_per_day": "Ліміт км/день", "max_ascent_per_day_m": "Ліміт набору висоти, м",
            "max_walking_hours_per_day": "Ліміт годин ходу", "max_overnight_elevation_m": "Найвища ночівля, м",
            "lightest_body_weight_kg": "Вага найлегшого, кг"}


def parse_days(text: str) -> int | None:
    """Trip length: «3 дні», «2-3 дні» -> 3, «дві ночівлі» -> 3, «одноденка» -> 1, «з 27 по 29» -> 3,
    «вихідні» -> 2. None if not said."""
    t = norm(text)
    for m in re.finditer(B + NUM + r"(?:\s*[-–]\s*" + NUM + r")?\s*-?\s*(?:дн(?:і|ів|я|ей)|день)" + E, t):
        sentence = t[max(t.rfind(".", 0, m.start()), t.rfind("\n", 0, m.start()), 0):m.start()]
        if SHIFT.search(t[max(0, m.start() - 30):m.start()]) or PAST.search(sentence):
            continue   # «зсунути старт на 1–2 дні» is the date flexibility; «минулого року … 4 дні» is history
        return max(to_int(m.group(1)) or 0, to_int(m.group(2) or "0") or 0) or None
    m = re.search(B + NUM + r"\s+ночів", t)
    if m and to_int(m.group(1)) is not None:
        return to_int(m.group(1)) + 1
    if re.search(r"одноден|без\s+ночів", t):
        return 1
    if re.search(r"на\s+(?:цілий\s+)?тиждень", t) and not re.search(r"через\s+тиждень", t):
        return 7
    m = re.search(r"з\s+(\d{1,2})" + DAY_SUFFIX + r"\s+(?:по|до)\s+(\d{1,2})" + DAY_SUFFIX + r"\s+" + MONTH, t) or \
        re.search(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})\s+" + MONTH, t)
    if m and 0 <= int(m.group(2)) - int(m.group(1)) <= 6:
        return int(m.group(2)) - int(m.group(1)) + 1
    if re.search(B + r"(?:вихідн|выходн)", t):
        return 2
    return None


DONT_CARE = re.compile(r"не\s*важлив|не\s+важно|все\s?одно|всеодно|байдуже|як\s+скажете|скільки\s+скажете|"
                       r"на\s+(?:ваш|твій)\s+розсуд|без\s+різниці|будь[- ]?(?:як|коли|скільки)|пофіг|"
                       r"не\s+має\s+значення|неважно")


def dont_care(answer: str) -> bool:
    return bool(DONT_CARE.search(norm(answer)))


# ------------------------------------------------------------------ the latest statement wins
# Answers and corrections are appended to the description («Уточнення:», «Виправлення:», «Зміна після плану:»);
# a fact read by code is taken from the LATEST part that states it, so «зміни старт на 25 листопада» wins
# over the 21 November of the first message.
PARTS = re.compile(r"\n\s*(?=Уточнення:|Виправлення|Зміна після плану:)")


def latest(fn, text: str, *args):
    """fn(part, *args) on the parts of the text from the last one back; the first non-None answer."""
    for part in reversed(PARTS.split(text)):
        found = fn(part, *args)
        if found not in (None, [], (None, None)):
            return found
    return None


def latest_people(text: str) -> tuple[int | None, int | None]:
    """people_count, each number from the latest part that states it."""
    participants = adults = None
    for part in reversed(PARTS.split(text)):
        p, a = people_count(part)
        participants = p if participants is None else participants
        adults = a if adults is None else adults
    return participants, adults


# ------------------------------------------------------------------ the check after the model
def group_consistency(d: dict[str, Any], text: str = "") -> list[str]:
    """Adults travelling on their own have no accompanying adults; a children's group with 0 adults
    is only believed when the text says so. Changes d in place; returns notes."""
    notes = []
    if d.get("section") == "senior":
        if d.get("participants") is None and d.get("adults"):
            d["participants"] = d["adults"]
        d["adults"] = 0
    elif d.get("section") in ("novatstvo", "yunatstvo") and d.get("adults") == 0 and text and \
            not NO_ADULTS.search(norm(text)):
        d["adults"] = None
        notes.append("Дорослих: у тексті не сказано, тому перепитаю (0 дорослих для дитячої групи не беру).")
    return notes


def refine(d: dict[str, Any], text: str, today: date) -> tuple[dict[str, Any], list[str]]:
    """Check the model's reading against the words of the text. Changes d; returns notes for the leader."""
    notes: list[str] = []

    by_age = age_section(text)
    if by_age and d.get("section") and by_age != d["section"]:
        notes.append(f"Група: за віком у тексті — {SECTION_NAME[by_age]} (модель прочитала {SECTION_NAME[d['section']]}).")
        d["section"] = by_age
    elif d.get("section") in SECTION_WORDS and not section_supported(text, d["section"]):
        notes.append("Групу (вік) у тексті не знайдено, тому перепитаю.")
        d["section"] = None
    if (d.get("date_flexibility_days") or 0) > 0 and not FLEX_WORDS.search(norm(text)):
        notes.append("Зсув дати: у тексті не сказано, тому 0 (модель вказала "
                     f"{d['date_flexibility_days']} дн.).")
        d["date_flexibility_days"] = 0

    participants, adults = latest_people(text)
    if participants is None:
        participants = latest(people_fallback, text)
    if participants is not None and participants != d.get("participants"):
        if d.get("participants") is not None:
            notes.append(f"Учасників: {participants} — так написано в тексті (модель прочитала {d['participants']}).")
        d["participants"] = participants
    elif participants is None and d.get("participants") is not None and d.get("section") != "senior" \
            and d["participants"] not in people_candidates(text):
        # the model's number with nothing in the text behind it: ask instead of guessing
        notes.append(f"Учасників: у тексті числа не знайдено, тому перепитаю (модель вказала {d['participants']}).")
        d["participants"] = None
    if adults is not None and adults != d.get("adults"):
        if d.get("adults") is not None:
            notes.append(f"Дорослих: {adults} — так написано в тексті (модель прочитала {d['adults']}).")
        d["adults"] = adults
    elif adults is None and d.get("adults") and d.get("section") in ("novatstvo", "yunatstvo") \
            and not ADULT_WORD.search(norm(text)):
        notes.append(f"Дорослих: у тексті не сказано, тому перепитаю (модель вказала {d['adults']}).")
        d["adults"] = None
    notes += group_consistency(d, text)

    times = latest(stated_times, text) or []
    model_time = d.get("start_time") or "08:00"
    if model_time not in times and (times or model_time != "08:00"):
        new = times[0] if len(times) == 1 else "08:00"   # the text's own time, or the usual one
        if model_time != new:
            notes.append(f"Час виходу: {new} — " + ("так написано в тексті" if len(times) == 1 else "у тексті не "
                         "сказано, тому звичайний") + (f" (модель вказала {model_time})." if model_time != "08:00" else "."))
        d["start_time"] = new
    stated = latest(explicit_date, text, today)
    if stated and d.get("start_date") and stated.isoformat() != d["start_date"]:
        notes.append(f"Дата старту: {stated:%d.%m.%Y} — так написано в тексті (модель прочитала {d['start_date']}).")
        d["start_date"] = stated.isoformat()
    elif stated is None and d.get("start_date") and latest(parse_date, text, today) is None:
        notes.append(f"Дата старту: у тексті не знайдено, тому перепитаю (модель вказала {d['start_date']}).")
        d["start_date"] = None
    if not d.get("date_flexibility_days"):
        shift = parse_shift(text)
        if shift:
            d["date_flexibility_days"] = min(shift, 3)
    for field, rx in LIMITS.items():
        found = {float(next(g for g in m.groups() if g).replace(",", ".")) for m in rx.finditer(norm(text))}
        value = d.get(field)
        if len(found) == 1:
            new = found.pop()
            new = int(new) if new.is_integer() and field != "max_km_per_day" else new
            if value is not None and float(value) != float(new):
                notes.append(f"{LIMIT_UA[field]}: {new:g} — так написано в тексті (модель вказала {value:g}).")
            d[field] = new
        elif value is not None and not found:
            notes.append(f"{LIMIT_UA[field]} {value:g} у тексті не знайдено, тому не враховую.")
            d[field] = None

    named = regions_named(text)
    now = present_text(text)   # «минулого року ходили на Петрос» says nothing about this trip
    model_region = d.get("region")
    if len(named) == 1:
        region = next(iter(named))
    elif named:
        region = model_region if model_region in named else None
    elif model_region and region_hinted(now, model_region):
        region = model_region
    else:   # «піпівана», «бребенеско» point to one region only
        hinted = {r for r in REGION_HINTS if region_hinted(now, r)}
        region = hinted.pop() if len(hinted) == 1 else None
    if region != model_region and model_region:
        notes.append(f"Район: {REGION_UA.get(region, 'не вказано')} — за словами тексту "
                     f"(модель вказала {REGION_UA.get(model_region, model_region)}).")
    elif region != model_region:
        how = "так написано в тексті" if named else "за назвами місць у тексті"
        notes.append(f"Район: {REGION_UA[region]} — {how}.")
    d["region"] = region

    wishes = wished_features(text)
    extra = [f for f in (d.get("must_include") or []) if f not in wishes]
    if extra:
        notes.append("Прибрано побажання, яких немає в тексті: " + ", ".join(FEATURE_UA[f] for f in dict.fromkeys(extra)) + ".")
    d["must_include"] = wishes

    places, dropped = clean_places(d.get("places") or [], text)
    if dropped:
        notes.append("Не беру як місця: " + "; ".join(dropped) + ".")
    d["places"] = places

    if d.get("start_date") is None:
        found = latest(parse_date, text, today)
        if found and found >= today:
            d["start_date"] = found.isoformat()
    days = latest(parse_days, text)
    if days and 1 <= days <= 7 and days != d.get("days"):
        if d.get("days") is not None:
            notes.append(f"Днів: {days} — так написано в тексті (модель прочитала {d['days']}).")
        d["days"] = days
    elif days is None and d.get("days") is not None and not re.search(r"\d", norm(text)):
        # nothing in the text to count days from: the model invented them («Юнацтво, хочемо в гори» -> 3 дні)
        notes.append(f"Днів: у тексті не сказано, тому перепитаю (модель вказала {d['days']}).")
        d["days"] = None
    return d, notes
