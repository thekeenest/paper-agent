# 📊 Инструкция по созданию Gold Standard датасета

## Обзор

Gold Standard датасет — это набор вручную размеченных статей, который используется для оценки качества работы системы. Рекомендуемый размер: **50-100 статей**.

## Структура датасета

Файл: `data/gold_standard.json`

```json
{
  "version": "1.0",
  "created": "2024-12-13T12:00:00",
  "total_papers": 50,
  "total_authors": 250,
  "papers": [
    {
      "paper_id": "2401.12345",
      "title": "Attention Is All You Need",
      "authors": [
        {
          "name": "Ashish Vaswani",
          "raw_affiliation": "Google Brain",
          "normalized_affiliation": "Google",
          "country": "United States",
          "country_code": "US",
          "org_type": "company"
        }
      ],
      "source": "manual",
      "annotator": "Your Name",
      "annotation_date": "2024-12-13",
      "notes": ""
    }
  ]
}
```

## Пошаговая инструкция

### Шаг 1: Выбор статей для разметки

**Критерии отбора:**
1. Разнообразие конференций (NeurIPS, ICML, CVPR, ACL, ICLR)
2. Разнообразие организаций (academia + industry)
3. Разнообразие стран (США, Китай, Европа, Канада)
4. Разный размер авторских коллективов (2-15 авторов)
5. Разные форматы аффилиаций в PDF

**Рекомендуемое распределение (50 статей):**
- NeurIPS 2023/2024: 15 статей
- ICML 2023/2024: 10 статей
- ICLR 2024: 10 статей
- CVPR 2024: 10 статей
- ACL 2024: 5 статей

**Как найти ArXiv ID:**
1. Откройте статью на arxiv.org
2. ID указан в URL: `arxiv.org/abs/2401.12345` → ID = `2401.12345`

### Шаг 2: Создание шаблона

```bash
cd conf_agent

# Создать шаблон для разметки
python evaluate.py --create-template \
  --papers "2401.12345,2401.12346,2401.12347"
```

Это создаст файл `data/gold_standard_template.json` с заготовками.

### Шаг 3: Ручная разметка

#### 3.1 Откройте PDF статьи

1. Скачайте PDF с ArXiv: `https://arxiv.org/pdf/2401.12345.pdf`
2. Откройте первую страницу — там указаны авторы и аффилиации

#### 3.2 Найдите информацию об авторах

**Где искать:**
- Под заголовком статьи
- В сносках (footnotes) внизу страницы
- В супер/подстрочных индексах (^1, *, †)
- В секции "Author Affiliations"

**Пример типичного формата:**
```
Ashish Vaswani^1    Noam Shazeer^1    Niki Parmar^1
Illia Polosukhin^2  ...

^1 Google Brain
^2 Google Research
```

#### 3.3 Заполните поля для каждого автора

| Поле | Что указывать | Пример |
|------|---------------|--------|
| `name` | Полное имя как в статье | `Ashish Vaswani` |
| `raw_affiliation` | Точно как написано в PDF | `Google Brain` |
| `normalized_affiliation` | Каноническое название | `Google` |
| `country` | Полное название страны | `United States` |
| `country_code` | ISO 3166-1 alpha-2 | `US` |
| `org_type` | Тип организации | `company` |

#### 3.4 Типы организаций (`org_type`)

| Тип | Описание | Примеры |
|-----|----------|---------|
| `university` | Университеты и колледжи | MIT, Stanford, Tsinghua |
| `company` | Компании | Google, Meta, Microsoft, OpenAI |
| `research_institute` | НИИ и лаборатории | INRIA, Max Planck, Shanghai AI Lab |
| `government` | Государственные организации | NASA, NSF |
| `hospital` | Медицинские учреждения | Mayo Clinic |
| `nonprofit` | Некоммерческие | Allen AI, MILA |

#### 3.5 Коды стран (часто используемые)

| Код | Страна |
|-----|--------|
| `US` | United States |
| `CN` | China |
| `GB` | United Kingdom |
| `DE` | Germany |
| `FR` | France |
| `CA` | Canada |
| `JP` | Japan |
| `KR` | South Korea |
| `IL` | Israel |
| `SG` | Singapore |
| `CH` | Switzerland |
| `AU` | Australia |
| `NL` | Netherlands |
| `HK` | Hong Kong |

### Шаг 4: Правила нормализации

#### 4.1 Название организации

**Правила:**
- Используйте каноническое английское название
- Не включайте department/lab (только организацию верхнего уровня)
- Для компаний с дочками: используйте бренд (`Google`, не `Alphabet`)

**Примеры:**
| raw_affiliation | normalized_affiliation |
|-----------------|------------------------|
| `Google Brain` | `Google` |
| `Meta AI Research` | `Meta` |
| `MIT CSAIL` | `Massachusetts Institute of Technology` |
| `Stanford NLP Group` | `Stanford University` |
| `DeepMind, London` | `Google DeepMind` |
| `Tsinghua Univ.` | `Tsinghua University` |
| `ETH Zürich` | `ETH Zurich` |

#### 4.2 Особые случаи

**Множественные аффилиации:**
- Указывайте ПЕРВУЮ (основную) аффилиацию
- Если автор из Google Brain и Stanford — выбирайте по контексту

**Неизвестная аффилиация:**
- Если не указана в PDF: `raw_affiliation = ""`, `normalized_affiliation = ""`
- Не угадывайте!

**Совместные лаборатории:**
- `Google DeepMind` → `Google DeepMind` (уже объединены)
- `MIT-IBM Watson AI Lab` → `Massachusetts Institute of Technology`

### Шаг 5: Валидация

После заполнения проверьте:

1. **Все авторы учтены** — сравните количество с PDF
2. **Имена корректны** — проверьте транслитерацию китайских/корейских имен
3. **Страны верны** — Hong Kong ≠ China (используйте HK)
4. **Типы правильные** — OpenAI = company (не nonprofit)

### Шаг 6: Сохранение

Переименуйте шаблон в рабочий файл:
```bash
mv data/gold_standard_template.json data/gold_standard.json
```

## Пример полной записи

```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": [
    {
      "name": "Ashish Vaswani",
      "raw_affiliation": "Google Brain",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Noam Shazeer",
      "raw_affiliation": "Google Brain",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Niki Parmar",
      "raw_affiliation": "Google Brain",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Jakob Uszkoreit",
      "raw_affiliation": "Google Brain",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Llion Jones",
      "raw_affiliation": "Google Research",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Aidan N. Gomez",
      "raw_affiliation": "University of Toronto",
      "normalized_affiliation": "University of Toronto",
      "country": "Canada",
      "country_code": "CA",
      "org_type": "university"
    },
    {
      "name": "Łukasz Kaiser",
      "raw_affiliation": "Google Brain",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    },
    {
      "name": "Illia Polosukhin",
      "raw_affiliation": "Google Research",
      "normalized_affiliation": "Google",
      "country": "United States",
      "country_code": "US",
      "org_type": "company"
    }
  ],
  "source": "manual",
  "annotator": "Your Name",
  "annotation_date": "2024-12-13",
  "notes": "Landmark paper on Transformer architecture"
}
```

## Автоматизация с помощью GPT-4

Для ускорения можно использовать GPT-4 для первичной разметки:

**Промпт:**
```
Extract all authors and their affiliations from this paper header.
Return JSON format:
{
  "authors": [
    {
      "name": "Full Name",
      "raw_affiliation": "As written",
      "normalized_affiliation": "Canonical name",
      "country": "Full country name",
      "country_code": "ISO 2-letter",
      "org_type": "university|company|research_institute|government|hospital|nonprofit"
    }
  ]
}

Paper text:
[ВСТАВЬТЕ ТЕКСТ ПЕРВОЙ СТРАНИЦЫ PDF]
```

**⚠️ Важно:** Всегда проверяйте результаты GPT-4 вручную!

## Запуск оценки

После создания датасета:

```bash
# Проверить статистику датасета
python evaluate.py --stats

# Запустить оценку
python evaluate.py --evaluate --csv output/affiliations_*.csv

# Сохранить отчёт
python evaluate.py --evaluate --csv output/affiliations_*.csv -o report.json
```

## Чеклист качества датасета

- [ ] Минимум 50 статей
- [ ] Минимум 200 авторов
- [ ] Минимум 5 разных конференций
- [ ] Минимум 10 разных стран
- [ ] Баланс academia/industry (примерно 60/40)
- [ ] Все поля заполнены корректно
- [ ] Нет дубликатов paper_id
- [ ] JSON валиден (проверить через jsonlint)
