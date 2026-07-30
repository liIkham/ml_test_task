# Refusal Direction Research

Исследовательское тестовое задание по интерпретируемости языковых моделей. Проект исследует направление отказа в residual stream instruction-tuned модели и сравнивает два вида интервенции: ручную аблитерацию весов и activation steering.

## Модель

Используется `Qwen/Qwen2.5-1.5B-Instruct` в `float16`. Сохранённый прогон выполнен на Tesla T4 16 GB.

## Данные и split

Для построения и оценки refusal direction используются вручную составленные harmful и harmless prompts:

- 30 harmful и 30 harmless prompts для train;
- 10 harmful и 10 harmless prompts для held-out evaluation;
- harmful train дополнительно делится на две половины по 15 примеров для проверки cosine stability.

Perplexity оценивается на отдельном нейтральном корпусе: 32 первых фрагмента из test split `Salesforce/wikitext`, конфигурация `wikitext-2-raw-v1`, содержащих не менее 20 слов. Каждый фрагмент ограничивается 256 токенами. Baseline и ablated perplexity считаются на одних и тех же текстах.

## Структура

- `notebooks/experiment.ipynb` — полный эксперимент;
- `results/refusal_direction.pt` — сохранённое направление и метаданные;
- `results/steering_results.csv` — refusal rate для activation steering;
- `results/ablation_summary.txt` — сводка ablation и next-token KL;
- `requirements.txt` — зависимости окружения.

## Запуск

Запускайте Jupyter из корня репозитория, чтобы относительные пути вели в `results/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab notebooks/experiment.ipynb
```

При первом запуске модель и WikiText загружаются с Hugging Face, поэтому требуется доступ в интернет.

## GPU-требования

Рекомендуется NVIDIA GPU с CUDA и не менее 8 GB VRAM. Сохранённые результаты получены на Tesla T4 16 GB с PyTorch `2.11.0+cu128`. CPU-only запуск не является целевым и будет существенно медленнее.

## Воспроизводимость

В notebook используется единый `SEED = 42` для Python, NumPy и PyTorch. Это уменьшает вариативность повторных запусков, но не гарантирует идентичные результаты на разных GPU, версиях CUDA или версиях библиотек.

## Основные результаты

Уже полученные результаты:

- выбранный слой: 20;
- split cosine similarity: 0.9968;
- harmful refusal rate после ablation: 0.9 → 0.8;
- mean next-token KL на harmless held-out: 0.00435;
- harmful refusal rate при steering: `+20 → 1.0`, `-20 → 0.3`;
- harmless refusal rate при steering: `+20 → 0.6`, `-20 → 0.0`.

В совокупности результаты показывают, что найденное направление связано с поведением отказа: steering в положительную сторону усиливает отказы, а в отрицательную — снижает их. Эффект аблитерации оказался слабее.

Perplexity на 32 нейтральных фрагментах WikiText:

- baseline perplexity: `16.7509`;
- ablated perplexity: `16.7468`;
- relative change: `-0.02%`;
- число предсказываемых токенов: `4037`.

Perplexity практически не изменилась. На этой выборке аблитерация не показала заметной деградации по данной метрике. Небольшое снижение на `0.02%` не следует интерпретировать как улучшение модели, поскольку разница слишком мала.

## Ограничения

- небольшие вручную составленные выборки;
- последовательный, а не случайный split;
- общий harmless mean в проверке split stability;
- эвристический refusal classifier;
- next-token KL не заменяет sequence-level оценку;
- perplexity измерена только на 32 фрагментах WikiText, поэтому вывод об отсутствии деградации ограничен этой небольшой выборкой;
- систематический alpha sweep не проводился;
- результаты относятся только к одной модели.

## Responsible use

Проект предназначен только для исследования механизмов безопасности. Метод имеет двойственное применение: он позволяет анализировать отказы, но также способен ослаблять защитное поведение модели.

Модифицированные веса не должны публиковаться или распространяться. Код и результаты не должны использоваться для получения опасного, незаконного или иного вредоносного контента. Выводы нельзя без дополнительной проверки переносить на другие модели и условия.
