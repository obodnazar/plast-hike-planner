# Ollama як запасна модель і розуміння українських запитів: дослідження

Складено 15.11.2026. Нічого не встановлювалось і не запускалось (Ollama на Mac ще немає). Цифри швидкості й пам'яті взято з джерел. Здебільшого це оцінки або заміри на іншому залізі, тож їх треба переміряти на своєму MacBook Air M4 / 16 ГБ.
Пов'язаний план: `../PLAN-ollama-and-understanding.md`.

---

## 1. Короткий висновок і рекомендація

1. **Для Ollama використовувати `method="json_schema"`, а не `function_calling`.** Ollama на `/v1/chat/completions` перекладає `response_format.json_schema.schema` у свій параметр `format`, тобто в граматику, яка обмежує декодування [2]. `tool_choice` Ollama не підтримує [1][2]. А LangChain у режимі `function_calling` якраз змушує модель викликати «інструмент-схему» через `tool_choice` [14]. Ollama це ігнорує, модель може відповісти звичайним текстом, і тоді парсер поверне `None` (мій висновок з коду [14]). Моделі на основі Gemma 3 (MamayLM, Lapa) в Ollama взагалі не мають tools і отримають помилку 400 [16]. Для OpenRouter `function_calling` можна залишити.
2. **Клієнт: залишити `ChatOpenAI` з `base_url="http://localhost:11434/v1"`.** Це той самий клас, що вже є в коді. Запасний варіант — `ChatOllama` (langchain-ollama 1.1.0, за замовчуванням json_schema): він потрібен, якщо захочеться задавати `num_ctx`, `keep_alive` чи вимикати мислення в кожному запиті [11][12][13].
3. **Основна локальна модель — `gemma4:12b`** (Q4_K_M, 8,0 ГБ на диску [20]; Google позиціонує її для ноутбуків з 16 ГБ [21][23]).
   - Родина Gemma лідирує в українських тестах. Gemma 4 26B займає 1-ше місце в Ukrainian LLM Leaderboard, а похідні від Gemma 3 MamayLM і Lapa — 2-ге і 3-тє [29].
   - Керівник Lapa називає Gemma 4 найкращою для української серед моделей до ~30B [28].
   - MMMLU 83,4 % [22]. Є tools, а мислення вимкнене за замовчуванням [22].

   **Легша альтернатива — `gemma4:e4b`** (6,6 ГБ, MMMLU 76,6 %, у кілька разів швидша) [20][22].
   **Претендент для порівняння на eval — `qwen3.5:9b`** (7,6 ГБ, MMMLU 81,2 %, сильна в tools/JSON). Але мислення в неї увімкнене за замовчуванням, а доказів щодо української менше [24][25][29].
4. **Обов'язкові налаштування:**
   - Контекст 8192. На Mac з менш ніж 24 ГБ за замовчуванням лише 4096 токенів, і задовгий промпт мовчки обрізається з початку, разом із системним промптом [6][7][17].
   - `temperature=0` задавати явно. Вбудовані налаштування gemma4 і qwen3.5 мають temperature 1, а `/v1` їх більше не перезаписує [10][20][24].
   - `reasoning_effort="none"`, щоб вимкнути мислення [2].
   - Ollama не старіша за 0.31.2: у ранніших версіях `format` ігнорувався, коли мислення вимкнене [9]. Поточна версія — 0.40.16 [10].
   - `keep_alive` 30 хв і «розігрів» моделі перед прогоном [7].
5. **Реалістична швидкість на Air M4.**
   - 12B: орієнтовно 6–15 токенів/с генерації і близько 150–230 токенів/с обробки промпту (оцінки) [36][37]. Отже, один розбір запиту триватиме приблизно 30–60 с.
   - e4b: у кілька разів швидше.
   - MLX-рушій Ollama вимагає понад 32 ГБ пам'яті. На 16 ГБ, за неофіційним джерелом, Ollama працює через llama.cpp/Metal [18][19].
   - Звідси: промпт тримати коротким, а статичну частину (інструкції й приклади) ставити на початок.
6. **«Навчання» за 2 тижні складається з чотирьох частин:** граматика через json_schema, 4–8 прикладів у промпті, детерміновані розбирачі в коді і набір для перевірки (eval). LoRA відкласти на потім:
   - для 12B на 16 ГБ RAM і 21 ГБ вільного диска шлях «злиття → GGUF» не вміщається, бо лише bf16-база важить близько 24 ГБ;
   - реально це лише для моделей класу 4B;
   - вигоду від LoRA без eval не видно [38][40][43].
7. **Eval: 50–60 «золотих» прикладів.** Справжні запити кладемо лише в тестову частину. Метрики: валідний JSON, точність по кожному полю, «усі поля правильні», затримка. При n=50 похибка частки близько ±11 п.п., тому моделі порівнювати за сумою полів, а не за одним полем.
8. **Дати й числа розбирати кодом.**
   - `dateparser` 1.5.0 знає українську: знахідний відмінок днів тижня, «завтра», «через N днів» [49][50]. Але правило «наступна/ця субота» реалізоване лише для англійської [49]. Тож потрібен свій маленький розбирач днів тижня, а dateparser — для «19 листопада».
   - Числівники: `number-parser` знає «двоє», «дві», «двох», «трьох» [52]. Альтернатива — свій словник на 1–30.
   - Для вибору маршруту варто в схемі задати динамічний `enum` з наявних id. Тоді модель фізично не зможе вигадати неіснуючий id.

---

## 2. Розділи A–E

### A. Можливості Ollama (стан на листопад 2026)

**Версія.** Останній стабільний реліз — v0.40.16 від 09.11.2026 [10]. У v0.40.14 OpenAI-сумісні ендпоінти перестали перезаписувати вбудовані параметри семплювання моделі [10]. Тож якщо клієнт не передає `temperature`, діє вбудоване значення моделі, а воно часто дорівнює 1.

**Нативний API (`/api/chat`, поле `format`).**
- `format` приймає JSON-схему (наприклад, `Model.model_json_schema()` з Pydantic) і обмежує вихід граматикою [3].
- Документація радить дублювати схему текстом у промпті і ставити `temperature=0` [3].
- `format: "json"` — це лише «будь-який JSON», без схеми [3].

**OpenAI-сумісний `/v1/chat/completions`.**
- Підтримувані поля: `model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `response_format`, `tools`, `reasoning_effort` [1].
- **Не підтримуються:** `tool_choice`, `logprobs`, `n`, `logit_bias`, `user` [1]. Розробники llama-stack через це взагалі прибирають `tools` із запитів до Ollama [15].
- `num_ctx` через OpenAI API задати не можна. Документація радить створити модель через Modelfile з `PARAMETER num_ctx` [1].
- У коді Ollama (`openai/openai.go`) `response_format.type="json_schema"` перетворюється на `format = схема`, а `json_object` — на `format="json"` [2]. Значення `reasoning_effort: "none"` перетворюється на `think=false` [2].
- Суперечність у джерелах: сторонній сайт сумісності стверджує, що json_schema ігнорується [4]. Тест pydantic-ai на Ollama 0.17.7 показав, що схема працює [5]. Код на `main` теж передає схему [2]. **Висновок: перевірити smoke-тестом (розділ 3) на своїй версії.**

**Що робить `ChatOpenAI.with_structured_output` (langchain-openai 1.8.0, встановлена в `.venv`) [14].**
- Метод за замовчуванням — `function_calling`. Він прив'язує схему як інструмент і передає `tool_choice=<назва>` та `parallel_tool_calls=False`. Через `/v1` Ollama `tool_choice` ігнорує, тож модель **може** не викликати інструмент, і тоді `PydanticToolsParser(first_tool_only=True)` поверне `None`.
- Аргументи інструмента граматикою не обмежуються: Ollama лише розбирає їх з тексту моделі (за кодом [2] схема йде в `format` тільки з `response_format`).
- `json_schema` передає Pydantic-клас у `client.chat.completions.parse(...)`. OpenAI SDK робить зі схеми «strict»-варіант: усі поля обов'язкові, `additionalProperties: false`. Для граматики Ollama це добре.
  - Помилка валідації Pydantic (наприклад, число поза межами) вилітає як виняток, тож виклик варто обгорнути одним повтором із текстом помилки.
  - Якщо `max_tokens` занадто малий, теж буде виняток (обрізаний JSON).

**`ChatOllama` (langchain-ollama 1.1.0, квітень 2026) [11][12][13].**
- `with_structured_output` за замовчуванням використовує `method="json_schema"`, тобто нативний `format`. Ця зміна з'явилася з версії 0.3.0 [11].
- Є параметри `num_ctx`, `keep_alive`, `num_predict`, `temperature`, `reasoning`:
  - `reasoning=False` вимикає мислення;
  - `reasoning=None` залишає поведінку моделі за замовчуванням, і тоді теги `<think>` можуть потрапити у відповідь [12].

**Порівняння шляхів**

| Шлях | Як Ollama обмежує вихід | Плюси | Мінуси |
|---|---|---|---|
| `ChatOpenAI` → `/v1`, `function_calling` | Ніяк: аргументи інструмента розбираються з тексту, `tool_choice` ігнорується [1][2] | Той самий виклик, що для OpenRouter | Можливий `None`; Gemma 3-моделі отримують 400 «does not support tools» [16] |
| **`ChatOpenAI` → `/v1`, `json_schema`** (рекомендовано) | `response_format` → `format` → граматика [2] | Мінімальна зміна коду; синтаксис і enum гарантовані | `num_ctx`, `keep_alive` і мислення задаються лише змінними середовища, Modelfile або `reasoning_effort` [1]; є одне суперечливе свідчення [4] |
| `ChatOllama`, `json_schema` (за замовчуванням) | `format=schema` у `/api/chat` [11] | Усі параметри Ollama доступні в коді [12] | Ще одна залежність і інший клас моделі |

**Відомі пастки**

1. **Контекст.**
   - Документація суперечить сама собі: FAQ каже, що за замовчуванням 4096 токенів [7]; сторінка про довжину контексту — що при пам'яті GPU менше 24 ГБ це 4k, а 32k лише при 24–48 ГБ [6]; Modelfile-документація для `num_ctx` досі пише 2048 [8].
   - Для цього Mac у будь-якому разі вийде близько 4096.
   - Задовгий промпт обрізається **мовчки, з початку** (тобто разом із системним промптом), а повідомлення є лише в debug-журналі [17].
   - Що робити: `OLLAMA_CONTEXT_LENGTH=8192` або Modelfile з `PARAMETER num_ctx 8192`, потім перевірити колонку `CONTEXT` в `ollama ps` [6]. У відповіді `/v1` поле `usage.prompt_tokens` має бути помітно меншим за ліміт.
2. **Мислення (thinking).**
   - Qwen3.5 за замовчуванням думає [25]. На Air це десятки секунд зайвої генерації.
   - Gemma 4 за замовчуванням не думає, але для 12B у відповіді лишається порожній блок думок [22].
   - Був баг: `format` ігнорувався, коли мислення вимкнене (qwen3.5, qwen3.6, gemma4). Його виправили 07.07.2026, а в 0.31.2 користувач підтвердив, що він зник [9]. Потрібна Ollama не старіша за 0.31.2.
   - Через `/v1` мислення вимикається параметром `reasoning_effort="none"` [2]. Для моделі без можливості мислення (наприклад, MamayLM з hf.co) цей параметр краще не передавати.
3. **Параметри семплювання.**
   - Вбудовані налаштування: `gemma4:12b` має temperature 1, top_k 64, top_p 0.95 [20]; `qwen3.5:9b` — temperature 1 і presence_penalty 1.5 [24].
   - Для розбору тексту задати `temperature=0`. `presence_penalty=0` теж задати явно: штраф за повтори може спотворювати значення в JSON (мій висновок, не перевірено).
4. **Холодний старт і `keep_alive`.**
   - Модель вивантажується через 5 хв бездіяльності [7].
   - Перший запит після цього включає завантаження ~8 ГБ з диска.
   - Що робити: `OLLAMA_KEEP_ALIVE=30m` і «розігрів» порожнім запитом [7]; тайм-аут клієнта не менше 180 с.
5. **Пам'ять.**
   - Потрібна пам'ять росте пропорційно `OLLAMA_NUM_PARALLEL × контекст` [7]. Залишити `OLLAMA_NUM_PARALLEL=1`.
   - MLX-рушій (прев'ю 0.19, стабільний з 0.30) вимагає понад 32 ГБ [18]. На 16 ГБ, за статтею dev.to, Ollama тихо переходить на llama.cpp/Metal [19].
   - Перевірка: `ollama ps` має показувати `100% GPU`. Якщо там частка CPU, буде в рази повільніше [6].
6. **Обмеження граматики (JSON-схема → GBNF у llama.cpp).** Що вважаю застосовним до Ollama на шляху llama.cpp (мій висновок):
   - непідтримувані ключові слова **мовчки пропускаються**;
   - вкладені `$ref` зламані;
   - `minimum` і `maximum` працюють лише для цілих чисел;
   - `pattern` має починатися з `^` і закінчуватися `$`;
   - формати рядків `uri` і `email` відсутні;
   - не можна змішувати `properties` з `anyOf` в одному типі [46].

   Тому схему тримати пласкою (не глибше одного рівня вкладених моделей), використовувати enum і `Optional[...]` (тобто anyOf з null).
7. **Моделі на Gemma 3 (MamayLM, Lapa) в Ollama не мають tools.** Шаблон не підтримує інструменти, тож запит з `tools` отримає 400 [16]. Режим `format` при цьому працює для будь-якої моделі.

### B. Моделі для української + JSON на 16 ГБ M4

Бюджет: на Mac з 16 ГБ GPU за замовчуванням доступно приблизно 10,5–11,5 ГБ (оцінки калькуляторів [37]). Вільного диска 21 ГБ, тобто одночасно на диску вміщаються щонайбільше 2 моделі, і ще треба залишити запас для системи.

| Модель | Тег Ollama | Диск | Пам'ять (оцінка) | Швидкість на M4 (оцінки) | Докази щодо української | JSON / tools | Вердикт |
|---|---|---|---|---|---|---|---|
| **Gemma 4 12B** (червень 2026) | `gemma4:12b` | 8,0 ГБ (7,4 ГБ модель Q4_K_M + 175 МБ зір + 465 МБ драфт-модель) [20] | ~9–10 ГБ з контекстом 8K | 6,3 т/с (спільнота, oMLX, M4 16 ГБ) … 13–25 т/с (оцінки) [36][37] | Gemma 4 — найкраща до ~30B за словами керівника Lapa [28]; Gemma 4 26B — №1 у UA-лідерборді [29]; MMMLU 83,4 % [22] | tools і мислення (за замовчуванням вимкнене) [20][22] | **Основна** |
| **Gemma 4 E4B** | `gemma4:e4b` | 6,6 ГБ [20] | ~7–8 ГБ | 25–45 т/с на Air (оцінка) [36] | MMMLU 76,6 % [22] | tools, мислення [20] | **Легка** |
| Qwen3.5 9B (лютий 2026) | `qwen3.5:9b` | 7,6 ГБ (5,6 ГБ модель + 0,9 ГБ зір) [24] | ~8–9 ГБ | 10–16 т/с (оцінка й заміри спільноти) | MMMLU 81,2 %, 201 мова [25]; попередня Qwen3-8B слабка в UA-лідерборді (у інструктивної версії нулі на ЗНО, ймовірно через формат відповіді) [29]; токенізатор Qwen витрачає більше токенів на українську (коментар читача) [28] | tools сильні, мислення **увімкнене** за замовчуванням [25] | Претендент для порівняння |
| Qwen3.5 4B | `qwen3.5:4b` | 3,3–4,0 ГБ [24] | ~5 ГБ | швидко | MMMLU 76,1 % [25]; у статті про витягування полів з фінансових транзакцій отримала F1 96,6 % лише з промптом «тільки JSON» [43] | tools | Найменший запасний варіант |
| MamayLM v2.0 12B (INSAIT, Gemma 3) | `hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` | 7,3 ГБ [31] | ~9 ГБ | як Gemma 3 12B | v1.0 12B — №2 у UA-лідерборді [29]; v2.0 заявляє кращу українську й tool calling, але без цифр [32] | Через Ollama лише `format`, бо шаблон Gemma 3 не має tools [16] | Для порівняння якості абзацу |
| Lapa v0.1.2/0.1.3 12B (УКУ/КПІ, Gemma 3) | GGUF лише для v0.1.2 [33] | ~7 ГБ | ~9 ГБ | — | Токенізатор потребує в ~1,5 раза менше токенів для української [34][35]; інструктивна версія має IFEval-UA 31,8 % (слабке виконання інструкцій) [29] | без tools | Не для JSON |
| Ministral 3 8B/14B | `ministral-3:8b` / `:14b` | 6,0 / 9,1 ГБ [26] | — | — | Української немає в переліку мов [26] | Нативний JSON і tools [26] | Не пріоритет |
| Llama 3.1 8B | — | — | — | — | Слабка: інструктивна версія має середній ранг 9,1 у UA-лідерборді [29] | tools | Ні |
| Qwen3.6 27B / Qwen3.8 27B / Gemma 4 26B/31B / Mistral Small 24B | `qwen3.8:27b` 18 ГБ, `gemma4:26b` 16–19 ГБ [20][27] | — | > 16 ГБ | — | — | — | Не вміщаються |

Примітки:
- **Ukrainian LLM Leaderboard** запустили 24.08.2026 центр WINWIN AI Мінцифри, УКУ і lang-uk [30]. Сам лідерборд вантажиться динамічно, тому цифри взято з дзеркала BenchmarkList (оновлення 06.05.2026) [29]. Там немає ні Gemma 4 12B/E4B, ні Qwen3.5, тож для нашої задачі прямих українських замірів **немає**. Висновок робиться за родиною моделей, а наш eval і є головним аргументом.
- Ukrainian-tuned моделі (MamayLM, Lapa) досі побудовані на Gemma 3. Найсвіжішу Lapa (v0.1.3) оновили в червні [33], а v1.0 лише анонсована. Обидві поступаються Gemma 4 у tools і в підтримці в Ollama.
- У `gemma4:12b` є драфт-модель для спекулятивного декодування [20]. Чи прискорює вона генерацію разом із граматикою — невідомо.

**Рекомендація:** основна `gemma4:12b`, легка `gemma4:e4b`. На eval прогнати ще `qwen3.5:9b` (з `reasoning_effort="none"`). Якщо залишиться час і місце, додати MamayLM v2 12B. Моделі тягнути по одній і видаляти після прогону (`ollama rm`).

### C. Як «вчити» модель

| Підхід | Скільки даних | Час | Що дає | Ризики |
|---|---|---|---|---|
| **(1) Приклади в промпті (few-shot)** | 4–8 добре підібраних прикладів | Години | Стиль розбору, правила для «не важливо», нормалізація назв | Кожен приклад додає ~150–250 токенів. На 12B це +7–15 с обробки промпту без кешу (оцінка з [37]). Приклади з eval у промпт не брати |
| **(2) LoRA на Mac (mlx-lm)** | 200–500 чистих прикладів для вузької задачі [40]; статті показують ефект уже від 100–1000 [44]; ранг 8 майже не поступається рангу 32 [43] | 500 ітерацій на 300 прикладах — 15–45 хв для моделі класу 8B на 16 ГБ (QLoRA ~7 ГБ) [40]; 12B на 16 ГБ — на межі | Стабільний формат, менший промпт, швидші відповіді | Диск: ланцюжок злиття → GGUF потребує bf16-бази (12B ≈ 24 ГБ) плюс f16 GGUF такого ж розміру. При 21 ГБ вільного місця для 12B це **неможливо**, для 4B — на межі |
| **(3) Обмежене декодування (json_schema → граматика)** | 0 | Хвилини | Гарантовано валідний JSON, типи, enum, межі масивів [3][46] | Не гарантує **правильних значень**. Непідтримувані ключові слова схеми мовчки ігноруються [46]. Жорсткий формат може погіршувати міркування [45] |

**LoRA у деталях (якщо колись дійде).**
- `mlx_lm.lora --train` з даними `train.jsonl` / `valid.jsonl` у форматі `chat`. Прапорець `--mask-prompt` рахує втрату лише на відповіді [38].
- Квантована модель на вході автоматично вмикає QLoRA. Для економії пам'яті: `--batch-size 1`, `--num-layers 8`, `--grad-checkpoint` [38].
- Підтримувані родини: Llama, Mistral, Qwen2, Gemma та інші [38]. Для Gemma 4 спільнота навчала адаптери через mlx-lm, але є скарги на помилки під час злиття [42].
- Експорт: `mlx_lm.fuse --dequantize` → GGUF. Вбудований `--export-gguf` вміє лише Llama, Mistral і Mixtral [38], тому для Gemma і Qwen потрібен `convert_hf_to_gguf.py` з llama.cpp, потім `llama-quantize` і Modelfile з `FROM ./model.gguf` [40].
- Директива `ADAPTER` у старій документації Ollama підтримувала лише Llama, Mistral і Gemma 1/2 [41]. У поточній документації цей розділ відсутній. Для Gemma 4 / Qwen3.5 вважати, що **не підтримується**.
- Без конвертації можна віддавати модель з адаптером через `mlx_lm.server` (OpenAI-подібний API). Але там немає `response_format` чи граматики [39], тож гарантію JSON доведеться робити валідацією Pydantic і повтором.

**Обмежене декодування і якість.**
- Tam et al. (EMNLP 2024): жорсткий формат (JSON-режим) погіршує задачі на міркування, але на класифікаційних задачах він конкурентний або кращий [45]. Наше витягування полів ближче до класифікації, тож граматика тут допомагає.
- Для вибору маршруту варто додати в схему **першим** коротке поле `rationale` (1–2 речення) перед `route_id`: якщо поле відповіді стоїть раніше за поле з поясненням, модель часто пропускає міркування [45]. Ціна — додаткові токени на повільному Air.

**Прагматичний шлях на 2 тижні:**
1. Граматика (`json_schema`) для обох структурованих задач. Для плану — динамічні `Literal`/`enum` з id маршрутів і точок, які реально є в даних.
2. Коротка пласка схема з описами полів (`Field(description=...)`) і 4–8 прикладів у промпті, які покривають важкі явища: суржик, помилки в назвах, відносні дати, «не важливо», виправлення.
3. Нормалізацію робить код: дати, числа, газетир назв, вік → група.
4. Eval на 50–60 прикладах для 2–3 моделей.
5. LoRA лише після захисту і лише якщо eval покаже **систематичну** помилку одного поля, яку не виправляють промпт і код. Робити на моделі класу 4B.

### D. Набір для перевірки (eval) і синтетичні дані

**Склад «золотого» набору (50–60 прикладів).**
- Тестова частина: справжні запити (з журналу діалогів, від знайомих провідників) і ручні варіації. Вона **ніколи** не потрапляє в промпт.
- Dev-частина: синтетика й решта прикладів. На ній підбирати few-shot і формулювання.
- Шари (на кожен по 5–8 прикладів):
  - розмовний стиль;
  - суржик;
  - помилки в назвах («піпівана», «бребенеско»);
  - транслітерація латиницею;
  - відносні дати («в суботу», «через тиждень»);
  - абсолютні дати («19 листопада»);
  - «не важливо», «всеодно»;
  - лише діти без дорослих;
  - суперечності й виправлення;
  - мінімальна інформація;
  - кілька фактів в одному реченні.
- Кожному прикладу дати теги шарів, щоб рахувати точність окремо по шарах.

Формат (назви полів умовні, взяти з реальної Pydantic-моделі):
```json
{"id": "real-001",
 "text": "хочу повести своїх дітей, наш гурток 10 хлопців десь 11-13 років, думав почати із піпівана і закінчити бребенеско",
 "today": "2026-11-15",
 "expected": {"youth_count": 10, "age_min": 11, "age_max": 13, "adults": null,
              "start_place_id": "pip-ivan-chornohirskyi", "end_place_id": "brebeneskul",
              "start_date": null, "nights": null, "region": "chornohora"},
 "tags": ["розмовний", "назви з помилками", "немає дат", "немає дорослих"]}
```
- Розрізняти `null` («не сказано») і явне значення-«будь-яке» для «не важливо». Це різні випадки, і обидва треба перевіряти.
- Для назв порівнювати **id з газетиру**, а не рядок.

**Метрики (рахувати окремо, не змішувати).**
- Частка валідного JSON / відповідей, які пройшли валідацію Pydantic. З граматикою має бути близько 100 %.
- Точність по кожному полю: точний збіг для enum, цілих чисел і дат; precision/recall/F1 для списків (множини id); id газетиру для назв. Окремо рахувати макро-середнє (кожне поле має однакову вагу) і мікро-середнє (усі пари «приклад–поле» разом).
- «Усі поля правильні» на прикладі: найсуворіша метрика.
- Затримка p50/p95 і кількість запитів до моделі на діалог.
- Статистика: при n=50 і точності 80 % 95-відсотковий інтервал приблизно ±11 п.п. (√(0,8·0,2/50) ≈ 0,057). Різниця між моделями в 5 п.п. на одному полі — це шум. Дивитися на суму полів і на шари.
- Процес: після кожної суттєвої зміни переглядати 20–50 відповідей очима і записувати типи помилок. Саме з них виростають нові приклади й правила [47].

**Синтетичні дані без «синтетичного стилю».**
1. Спершу задати **виміри** запиту і генерувати комбінації (кортежі) з них, а не просити «придумай 100 запитів» [47]. Виміри для нас:
   - хто пише (провідник-підліток, батько, вчитель, вожатий);
   - регістр (офіційний, розмовний, суржик, без розділових знаків, голосовий набір);
   - повнота інформації;
   - тип дати;
   - склад групи;
   - назви (правильні, з помилками, народні).
2. **Зворотна генерація:** код спершу генерує правильний JSON (значення відомі), а сильна модель пише текст, що виражає саме ці факти в заданому стилі. Мітки тоді правильні за побудовою. Окремо треба перевірити, що модель не додала зайвих фактів: другий прохід іншою моделлю і ручний перегляд вибірки.
3. **Шум додавати кодом,** а не просити модель «зробити помилки»:
   - пропущені апострофи;
   - ї→і, є→е, и↔і;
   - російські літери (ы, э, ъ);
   - злиті слова;
   - відмінки назв;
   - латиниця.
4. **Різні генератори** (2–3 моделі різних провайдерів) і обмеження довжини. Дослідження показують, що синтетичні промпти помітно менш різноманітні за людські, а детальні «персони» додають мало різноманіття порівняно з простим обмеженням довжини [48].
5. Прибрати майже-дублікати (n-грами або ембединги). Синтетика — лише для dev і навчання, тест — лише справжні запити. Окремо міряти розрив «синтетика → справжні». Synthetic не показує, наскільки часта помилка в реальності [47].
6. Приватність: не відправляти справжні імена й контакти у зовнішні API. Журнал діалогів знеособлювати.

### E. Дати й числа українською в коротких відповідях

**`dateparser`** (1.5.0 від 04.11.2026 [49]).
- Локаль `uk` є в списку, без регіональних варіантів [51].
- Словник `uk.py` містить [50]:
  - дні тижня в називному і знахідному відмінках («суботу», «неділю», «п'ятницю», «середу») і скорочення;
  - «завтра», «післязавтра»;
  - «через N днів/тижнів»;
  - «наступного тижня/місяця/року»;
  - форми «наступної», як у «наступної суботи», у словнику **немає**.
- Правило «last/this/next + день тижня» задокументоване лише для англійської [49].
- `search_dates` (пошук дати в довшому тексті) автори самі називають обмеженим [49].
- Як поводиться саме «в суботу» (найближча субота чи помилка) — **не перевірено**, потрібен тест з фіксованим `RELATIVE_BASE`.
- Налаштування: `languages=["uk"]`, `settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": today, "DATE_ORDER": "DMY"}`.
- Небезпека: голе число («2») dateparser може прочитати як дату. Тому викликати його лише там, де очікується дата.

**Альтернатива — `ovos-date-parser`** (0.29.0, липень 2026). Має окрему українську реалізацію `extract_datetime` і `extract_duration` [53]. Якість для української не перевірено.

**Числа.**
- `number-parser` підтримує кількісні числівники англійською, гінді, іспанською, українською й російською [52]. Український словник містить «двоє», «дві», «двох», «трьох», «троє», «четверо» [52].
- Останній реліз 0.3.2 вийшов 28.03.2023, тож бібліотека стабільна, але не розвивається.
- Порядкові числівники й дроби — лише англійською [52].
- Перед розбором нормалізувати апострофи (’ ʼ ' → ').

**Рекомендація.** Маленькі детерміновані розбирачі у своєму коді (усе одно потрібне підтвердження «Ось що я зрозумів»):
```python
import re
from datetime import date, timedelta

WEEKDAYS = {"понеділ": 0, "вівтор": 1, "серед": 2, "четвер": 3,
            "п'ятниц": 4, "субот": 5, "неділ": 6}
ANY = {"не важливо", "неважливо", "всеодно", "все одно", "будь-коли", "як скажете", "без різниці"}
SMALL = {"один": 1, "одна": 1, "двоє": 2, "два": 2, "дві": 2, "троє": 3, "три": 3,
         "четверо": 4, "чотири": 4, "п'ятеро": 5, "п'ять": 5, "шестеро": 6, "шість": 6}

def norm(s: str) -> str:
    return re.sub(r"[’ʼ`]", "'", s.lower()).strip()

def parse_weekday(text: str, today: date) -> date | None:
    t = norm(text)
    for stem, wd in WEEKDAYS.items():
        if stem in t:
            delta = (wd - today.weekday()) % 7 or 7   # найближчий такий день у майбутньому
            return today + timedelta(days=delta)       # «наступної суботи» неоднозначне, тож підтвердити датою
    return None

def parse_nights(text: str) -> int | None:            # «дві ночівлі» → 2 (днів = ночей + 1)
    t = norm(text)
    m = re.search(r"(\d+)\s*ноч", t)
    if m:
        return int(m.group(1))
    for w, n in SMALL.items():
        if re.search(rf"\b{w}\b\s+ноч", t):
            return n
    return None
```
Потім ланцюжок: `ANY` → цифри → `SMALL` / number-parser → dateparser (лише для дат) → якщо нічого не вийшло, уточнювальне питання. Модель на цьому етапі не потрібна.

---

## 3. Команди й налаштування (не виконувались)

**Встановлення і моделі** (це твій крок; кожна модель важить кілька ГБ):
```bash
# застосунок з ollama.com або: brew install ollama
ollama --version                 # має бути ≥ 0.31.2 (поточна 0.40.x)
ollama pull gemma4:12b           # ~8,0 ГБ
ollama show gemma4:12b           # у Capabilities мають бути tools, thinking
ollama run gemma4:12b --verbose "Привіт! Скільки буде двоє плюс троє?"   # eval rate = токени/с
ollama ps                        # CONTEXT і PROCESSOR (має бути 100% GPU)
# за потреби легша: ollama pull gemma4:e4b ; порівняння: ollama pull qwen3.5:9b
# прибрати після прогону: ollama rm qwen3.5:9b
```

**Змінні середовища для застосунку Ollama на macOS** (через `launchctl`, потім перезапустити застосунок) [7]:
```bash
launchctl setenv OLLAMA_CONTEXT_LENGTH 8192
launchctl setenv OLLAMA_KEEP_ALIVE 30m
launchctl setenv OLLAMA_NUM_PARALLEL 1
# необов'язково, економія пам'яті на KV-кеші:
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
```

**Або окрема модель через Modelfile** (не залежить від глобальних змінних):
```
FROM gemma4:12b
PARAMETER num_ctx 8192
PARAMETER temperature 0
```
```bash
ollama create plast-gemma4 -f Modelfile     # далі в коді model="plast-gemma4"
```

**Smoke-тест: чи справді `/v1` дотримується json_schema:**
```bash
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "gemma4:12b", "temperature": 0, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "наш гурток 10 хлопців 11-13 років, дорослих двоє"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "group", "schema": {
     "type": "object", "additionalProperties": false,
     "properties": {"youth": {"type": "integer"}, "adults": {"type": "integer"},
                    "age_band": {"type": "string", "enum": ["7-10", "11-13", "14-17", "18+"]}},
     "required": ["youth", "adults", "age_band"]}}}
}'
```
Очікувано: лише JSON з трьома ключами, `age_band` з переліку. Для порівняння нативний виклик: `POST /api/chat` з `"format": {...схема...}`, `"think": false`, `"options": {"temperature": 0, "num_ctx": 8192}`, `"stream": false`.

**Python: запасна модель через той самий `ChatOpenAI`:**
```python
import openai
from langchain_openai import ChatOpenAI

local = ChatOpenAI(
    base_url="http://localhost:11434/v1", api_key="ollama",   # ключ будь-який, Ollama його не перевіряє
    model="gemma4:12b", temperature=0, presence_penalty=0,
    reasoning_effort="none",          # → think=false; для моделей без мислення не передавати
    max_tokens=800, timeout=180, max_retries=0,
)
extract_local = local.with_structured_output(HikeRequest, method="json_schema")

# якщо вистачить стандартного механізму LangChain (у плані є свій ланцюжок у класі LLM);
# extract_remote = ChatOpenAI(<OpenRouter>).with_structured_output(HikeRequest, method="function_calling")
extract = extract_remote.with_fallbacks(
    [extract_local], exceptions_to_handle=(openai.RateLimitError, openai.APITimeoutError))
```

**Альтернатива: `ChatOllama`:**
```python
from langchain_ollama import ChatOllama   # pip/uv: langchain-ollama (1.1.0)
local = ChatOllama(model="gemma4:12b", temperature=0, num_ctx=8192,
                   keep_alive="30m", reasoning=False, num_predict=800)
extract_local = local.with_structured_output(HikeRequest)   # за замовчуванням json_schema → format
```

**Динамічний enum для вибору маршруту і точок:**
```python
from typing import Literal
from pydantic import Field, create_model

def make_plan_choice(route_ids: list[str], waypoint_ids: list[str]):
    RouteId = Literal[tuple(route_ids)]          # Literal[("a","b")] == Literal["a","b"]
    WaypointId = Literal[tuple(waypoint_ids)]
    return create_model(
        "PlanChoice",
        rationale=(str, Field(description="1–2 речення: чому цей маршрут")),   # першим полем
        route_id=(RouteId, ...),
        waypoint_ids=(list[WaypointId], Field(min_length=1, max_length=8)),
    )
```
Схему тримати пласкою: вкладені `$ref` у граматиці llama.cpp зламані [46].

**LoRA (лише на потім, класу 4B; назву репозиторію mlx-community перевірити):**
```bash
uv pip install "mlx-lm[train]"
mlx_lm.lora --model mlx-community/gemma-4-e4b-it-4bit --train --data data/lora \
  --iters 500 --batch-size 1 --num-layers 8 --grad-checkpoint --mask-prompt \
  --adapter-path adapters/extract-v1
mlx_lm.lora --model mlx-community/gemma-4-e4b-it-4bit --adapter-path adapters/extract-v1 --data data/lora --test
# експорт в Ollama: fuse (--dequantize) → llama.cpp convert_hf_to_gguf.py --outtype f16 → llama-quantize Q4_K_M
#                  → Modelfile "FROM ./model-q4_k_m.gguf" → ollama create
```
Формат `data/lora/train.jsonl`: `{"messages": [{"role": "system", ...}, {"role": "user", "content": "<текст>"}, {"role": "assistant", "content": "<JSON>"}]}` [38].

**Скрипт eval (`scripts/eval_parsing.py`, ідея):** для кожної моделі прогнати всі приклади і зберегти сирі відповіді, час і `usage.prompt_tokens`. Потім порахувати метрики з розділу D по полях і шарах і записати таблицю в `research/eval-<дата>.md`. Прогони на Air робити блоками до ~20 хв, бо через ~20–30 хв навантаження Air без вентилятора скидає частоти [36].

---

## 4. Ризики й невідоме

- **Справжня швидкість на Air M4 невідома.** Цифри 6–25 т/с для 12B суперечать одна одній [36][37]. Якщо один розбір на 12B займе понад 60 с, основною запасною зробити `gemma4:e4b`.
- **Пам'ять.** 12B з контекстом 8K займає ~9–10 ГБ при ~10,5–11,5 ГБ доступних GPU. Відкриті браузер чи IDE можуть витіснити частину моделі на CPU або в swap. Зменшувати контекст або перейти на e4b.
- **Диск.** 21 ГБ вільного місця вистачить лише на 2 моделі одночасно. LoRA для 12B неможлива без звільнення місця або зовнішнього диска (`OLLAMA_MODELS` можна перенести).
- **Суперечливі свідчення про `/v1` + json_schema** [4] проти [2][5]. Знімається smoke-тестом. Якщо не працює — `ChatOllama`.
- **Регресії Ollama з мисленням і `format`.** Баг #14645 повертався в окремих версіях (повідомлення про 0.30.x) [9]. Після оновлення Ollama щоразу проганяти smoke-тест.
- **Немає українських бенчмарків саме для Gemma 4 12B/E4B і Qwen3.5** на витягування полів. Вибір моделі — гіпотеза, яку перевіряє наш eval.
- **MLX-рушій на 16 ГБ:** офіційно лише «понад 32 ГБ» [18]. Поведінку на 16 ГБ описує неофіційне джерело [19]. Чи працюють граматики на MLX-рушії — невідомо.
- **dateparser і «в суботу» / «наступної суботи»** — не перевірено. Самописний розбирач надійніший.
- **Граматика гарантує форму, а не зміст.** Модель може впевнено поставити правильне за типом, але хибне значення (наприклад, `adults=1`, коли дорослих не згадано). Ловиться eval-ом і підтвердженням людиною.
- **Ліцензії:** Gemma 4 — Apache 2.0 [21]. MamayLM — Gemma Terms of Use [31]. Для студентського проєкту обидві підходять.
- Частина джерел — блоги й калькулятори (позначено «оцінка»). Офіційні документи Ollama місцями суперечать один одному щодо контексту за замовчуванням [6][7][8].

---

## 5. Джерела

Дата в дужках — дата публікації або оновлення. «Переглянуто» означає, що дата публікації не вказана і сторінку відкрито 15.11.2026.

1. Ollama Docs — OpenAI compatibility. https://docs.ollama.com/api/openai-compatibility (переглянуто 15.11.2026)
2. Ollama, вихідний код `openai/openai.go` (гілка main). https://github.com/ollama/ollama/blob/main/openai/openai.go (переглянуто 15.11.2026)
3. Ollama Docs — Structured outputs. https://docs.ollama.com/capabilities/structured-outputs (переглянуто 15.11.2026)
4. «Am I OpenAI compatible?» — сторінка Ollama. https://heiervang-technologies.github.io/am-i-openai-compatible/implementations/ollama/ (дата й версія не вказані)
5. DEV Community — «Does pydantic-ai's structured output API actually work against Ollama's OpenAI-compatible endpoint?» (Ollama 0.17.7, 2026). https://dev.to/kasir-barati/does-pydantic-ais-structured-output-api-actually-work-against-ollamas-openai-compatible-endpoint-44aa
6. Ollama Docs — Context length. https://docs.ollama.com/context-length (переглянуто 15.11.2026)
7. Ollama Docs — FAQ (keep_alive, OLLAMA_CONTEXT_LENGTH, launchctl, KV cache). https://docs.ollama.com/faq (переглянуто 15.11.2026)
8. Ollama Docs — Modelfile reference. https://docs.ollama.com/modelfile (переглянуто 15.11.2026)
9. GitHub ollama/ollama issue #14645 «format is ignored when think is disabled for qwen3.5 series» (відкрито 05.03.2026, закрито 07.07.2026; підтверджено в 0.31.2). https://github.com/ollama/ollama/issues/14645
10. Ollama Releases (v0.40.16 від 09.11.2026; v0.40.14 — параметри семплювання в /v1). https://github.com/ollama/ollama/releases
11. LangChain reference — `ChatOllama.with_structured_output`. https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama/with_structured_output (переглянуто 15.11.2026)
12. LangChain reference — `ChatOllama` (параметри `reasoning`, `num_ctx`, `keep_alive`). https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama (переглянуто 15.11.2026)
13. PyPI — langchain-ollama 1.1.0 (07.04.2026). https://pypi.org/project/langchain-ollama/
14. langchain-openai 1.8.0, `langchain_openai/chat_models/base.py` у `.venv` проєкту plast-hike-planner (методи `with_structured_output`, `_convert_to_openai_response_format`, `_oai_structured_outputs_parser`; переглянуто локально 15.11.2026)
15. llama-stack, коміт «ollama isn't respecting tool_choice» (прибирають `tools` для Ollama). https://git.kvant.cloud/phoenix-oss/llama-stack/commit/934446ddb4cf8801e011083fd6ae8d4a65724831 (дата не вказана)
16. BetterClaw — «Ollama models tool calling support» (18.08.2026; gemma3 без tools, таблиця tools-моделей). https://www.betterclaw.io/blog/ollama-models-tool-calling-support
17. Faun.dev — «Silent truncation: the trap» (Ollama num_ctx). https://faun.dev/sensei/academy/go/local-ai-engineering-with-ollama-05b200-3b93fc-3ae/the-context-window-bc58b0/silent-truncation-the-trap-796745/ (дата не вказана)
18. Ollama Blog — «Ollama is now powered by MLX on Apple Silicon in preview» (30.03.2026). https://ollama.com/blog/mlx
19. DEV Community — «Ollama v0.30 on Apple Silicon: what the stable MLX release actually changed» (05.07.2026). https://dev.to/jovan_chan_9500711396d4e6/ollama-v030-on-apple-silicon-what-the-stable-mlx-release-actually-changed-from-the-preview-25h7
20. Ollama Library — gemma4, gemma4:12b, gemma4:e4b (теги оновлено ~жовтень 2026). https://ollama.com/library/gemma4 , https://ollama.com/library/gemma4:12b , https://ollama.com/library/gemma4:e4b
21. Google Blog — «Introducing Gemma 4 12B» (03.06.2026). https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
22. Hugging Face — google/gemma-4-12b-it, картка моделі (MMMLU, мислення, мови). https://huggingface.co/google/gemma-4-12b-it (переглянуто 15.11.2026)
23. The Decoder — «Gemma 4 12B squeezes multimodal AI onto a laptop with just 16 GB of RAM» (03.06.2026). https://the-decoder.com/google-deepminds-gemma-4-12b-squeezes-multimodal-ai-onto-a-laptop-with-just-16-gb-of-ram/
24. Ollama Library — qwen3.5, qwen3.5:9b. https://ollama.com/library/qwen3.5 , https://ollama.com/library/qwen3.5:9b (переглянуто 15.11.2026)
25. Hugging Face — Qwen/Qwen3.5-9B, картка моделі (лютий 2026). https://huggingface.co/Qwen/Qwen3.5-9B
26. Ollama Library — ministral-3. https://ollama.com/library/ministral-3 (теги ~грудень 2025)
27. Ollama Library — qwen3.8 (лише 27B, 18 ГБ), qwen3.6 (27B/35B). https://ollama.com/library/qwen3.8 , https://registry.ollama.ai/library/qwen3.6 (переглянуто 15.11.2026)
28. DOU — інтерв'ю з Юрієм Панівим (LapaLLM): «Чому Gemma знає українську краще за інших…» (25.09.2026). https://dou.ua/lenta/interviews/lapallm-paniv/
29. BenchmarkList — дзеркало Ukrainian LLM Leaderboard (оновлено 06.05.2026). https://benchmarklist.com/benchmarks/ukrainian_llm_leaderboard/ ; оригінал: https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard
30. DOU — «В Україні створили рейтинг ШІ-моделей…» (24.08.2026). https://dou.ua/lenta/news/ai-comparing-of-ukrainian-language/
31. Hugging Face — INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF (оновлено 03.06.2026). https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF
32. DOU — «Команда MamayLM випустила друге покоління українських моделей…» (04.06.2026). https://dou.ua/lenta/news/new-version-mamaylm/
33. Hugging Face — організація lapa-llm (v0.1.3 від 06.06.2026; GGUF лише для v0.1.2). https://huggingface.co/lapa-llm
34. Forbes Україна — про Lapa LLM (28.10.2025). https://forbes.ua/innovations/vikoristali-obchislyuvalni-resursi-na-500-000-naukovtsi-z-uku-y-kpi-stvorili-model-lapa-llm-optimizovanu-pid-ukrainsku-movu-shcho-v-niy-osoblivogo-28102025-33666
35. UNLP 2026 (ACL Anthology) — стаття Lapa про адаптацію токенізатора. https://aclanthology.org/2026.unlp-1.14/
36. Gemma4All — «Gemma 4 on MacBook Air: The Fanless Reality Check» (11.07.2026; переважно оцінки). https://gemma4all.com/blog/gemma-4-on-macbook-air
37. Заміри спільноти й калькулятори: oMLX community benchmarks (M4 10-core GPU / 16 ГБ: gemma-4-12B-it 6,3 т/с; Llama-3.1-8B 226 т/с обробки промпту і 19,9 т/с генерації). https://omlx.ai/benchmarks ; оцінки WillItRunAI / LLMCheck: https://willitrunai.com/de/can-run/gemma-4-12b-on-m4-16gb , https://llmcheck.net/models/gemma-4-12b-on-m4/ (переглянуто 15.11.2026)
38. mlx-lm — LORA.md. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md (переглянуто 15.11.2026)
39. mlx-lm — SERVER.md. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md (переглянуто 15.11.2026)
40. InsiderLLM — «Fine-Tuning on Mac: LoRA & QLoRA with MLX» (26.02.2026, оновлено 04.11.2026). https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/
41. Ollama Docs (старе дзеркало) — Importing a model / adapters. https://ollama.readthedocs.io/en/import/ (може бути застарілим; поточна сторінка https://docs.ollama.com/import не описує ADAPTER)
42. Hugging Face Forum — «Gemma4-e4b adaptors fuse after training, how?» (2026). https://discuss.huggingface.co/t/gemma4-e4b-adaptors-fuse-after-training-how/176429
43. Huang et al. — «How Small Can You Go? LoRA Fine-Tuning 270M–8B Models for Merchant Information Extraction» (arXiv 2606.08051, 06.06.2026). https://arxiv.org/abs/2606.08051
44. Chih, Hou — «Low-Resource Fine-Tuning for Multi-Task Structured Information Extraction with a Billion-Parameter Instruction-Tuned Model» (arXiv 2509.08381, 10.09.2025). https://arxiv.org/abs/2509.08381
45. Tam et al. — «Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of LLMs» (EMNLP 2024 Industry; arXiv 2408.02442). https://arxiv.org/abs/2408.02442
46. llama.cpp — grammars/README.md (JSON Schema → GBNF, обмеження). https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md (переглянуто 15.11.2026)
47. Hamel Husain, Shreya Shankar — AI Evals FAQ: синтетичні дані (01.06.2025), аналіз помилок (27.06.2025). https://hamel.dev/blog/posts/evals-faq/ , https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html
48. «Measuring Lexical Diversity of Synthetic Data Generated through Fine-Grained Persona Prompting» (EMNLP 2025 Findings; arXiv 2505.17390). https://arxiv.org/abs/2505.17390
49. PyPI — dateparser 1.5.0 (04.11.2026). https://pypi.org/project/dateparser/
50. dateparser — `date_translation_data/uk.py`. https://github.com/scrapinghub/dateparser/blob/master/dateparser/data/date_translation_data/uk.py (переглянуто 15.11.2026)
51. dateparser — Supported languages and locales. https://dateparser.readthedocs.io/en/latest/supported_locales.html
52. number-parser (scrapinghub): README і `data/uk.py`; PyPI 0.3.2 (28.03.2023). https://github.com/scrapinghub/number-parser , https://pypi.org/project/number-parser/
53. PyPI — ovos-date-parser 0.29.0 (21.07.2026). https://pypi.org/project/ovos-date-parser/
