# NLP-SQL

## توضیح پروژه

این پروژه یک سیستم deterministic و rule-based برای تبدیل پرسش زبان طبیعی انگلیسی
به SQL امن است. سیستم از LLM، مدل هوش مصنوعی، embedding، vector database یا سرویس
AI خارجی استفاده نمی‌کند.

هدف پروژه این است که کاربر بتواند روی دیتابیس واقعی Nashville Housing سؤال‌هایی
مثل این‌ها بپرسد:

```text
show top 5 properties by sale price
show properties with bedrooms at least 4 and sale price less than 300000
average sale price
show properties where land use is single family
```

سیستم query را به AST معنایی تبدیل می‌کند، AST را با schema دیتابیس validate
می‌کند، SQL پارامتری می‌سازد و بعد روی SQLite اجرا می‌کند.

## شمای کلی و مدل کار سیستم

مسیر اصلی سیستم:

```text
Natural Language Query
        ↓
Text Normalizer
        ↓
Tokenizer / Phrase Matcher
        ↓
Rule-Based Semantic Parser
        ↓
Schema Resolver
        ↓
Typed Query AST
        ↓
AST Validator
        ↓
Parameterized SQL Compiler
        ↓
SQL Safety Validator
        ↓
SQLite Executor
        ↓
Rows
```

مدل کار با دیتابیس واقعی:

```text
Kaggle CSV Dataset
        ↓
SQLite Importer
        ↓
SQLite Schema Introspection
        ↓
Schema Registry
        ↓
NLP-to-AST
        ↓
Validated SQL
```

نکته‌های مهم:

- parser هیچ‌وقت SQL خام از متن کاربر نمی‌سازد.
- contract اصلی سیستم `QueryAST` است.
- table، column، operator و aggregation همگی whitelist و validate می‌شوند.
- valueها مستقیم داخل SQL قرار نمی‌گیرند و به صورت parameter پاس داده می‌شوند.
- API اختیاری است و فقط wrapper روی همین parser deterministic است؛ مدل AI نیست.

## Tree کل ریپو

```text
NLP-SQL/
├── ARCHITECTURE.md
├── NO_AI_POLICY.md
├── README.md
├── pyproject.toml
├── src/
│   └── nlp_sql/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── cli.py
│       ├── compiler.py
│       ├── datasets.py
│       ├── date_parser.py
│       ├── engine.py
│       ├── executor.py
│       ├── normalizer.py
│       ├── number_parser.py
│       ├── parser.py
│       ├── query_ast.py
│       ├── result.py
│       ├── safety.py
│       ├── schema.py
│       ├── tokenizer.py
│       ├── validator.py
│       └── config/
│           ├── __init__.py
│           └── default_schema.json
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_cli.py
    ├── test_compiler_validator_security.py
    ├── test_datasets.py
    ├── test_date_parser.py
    ├── test_engine_integration.py
    ├── test_housing_engine.py
    ├── test_no_ai_policy.py
    ├── test_normalizer.py
    ├── test_number_parser.py
    ├── test_package.py
    ├── test_schema.py
    └── test_tokenizer.py
```

## نحوه clone و run

Clone:

```bash
git clone https://github.com/rashidiff/NLP-SQL.git
cd NLP-SQL
```

ساخت virtual environment:

```bash
python -m venv .venv
```

فعال‌سازی در Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

فعال‌سازی در macOS / Linux:

```bash
source .venv/bin/activate
```

نصب پروژه:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

دانلود دیتاست Kaggle و ساخت دیتابیس SQLite:

```bash
nlp-sql prepare-data
```

اجرای query واقعی روی دیتابیس:

```bash
nlp-sql query "show top 5 properties by sale price"
```

چند نمونه query دیگر:

```bash
nlp-sql query "show properties built after 2000"
nlp-sql query "show properties with bedrooms at least 4 and sale price less than 300000"
nlp-sql query "average sale price"
nlp-sql query "show properties where land use is single family"
```

دیدن AST و SQL بدون اجرا:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
```

دیدن تفسیر semantic:

```bash
nlp-sql explain "average sale price"
```

اجرای API اختیاری:

```bash
nlp-sql serve
```

اجرای تست‌ها:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

