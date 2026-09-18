from nlp_sql.normalizer import TextNormalizer


def test_normalizer_lowercases_and_collapses_whitespace() -> None:
    result = TextNormalizer().normalize("  Show   ALL Customers  ")

    assert result.text == "show all customers"


def test_normalizer_removes_punctuation_without_creating_sql() -> None:
    result = TextNormalizer().normalize("show customers; DELETE FROM customers!")

    assert result.text == "show customers delete from customers"


def test_normalizer_expands_common_contractions() -> None:
    result = TextNormalizer().normalize("What's the total revenue?")

    assert result.text == "what is the total revenue"


def test_normalizer_resolves_currency_and_commas() -> None:
    result = TextNormalizer().normalize("Show orders above $5,000,000")

    assert result.text == "show orders above 5000000"
    assert result.number_replacements == (("$5000000", 5000000),)


def test_normalizer_resolves_scaled_numeric_value() -> None:
    result = TextNormalizer().normalize("Show orders above 2.5 million")

    assert result.text == "show orders above 2500000"


def test_normalizer_resolves_word_numbers() -> None:
    result = TextNormalizer().normalize("Show customers who spent five million")

    assert result.text == "show customers who spent 5000000"
    assert result.number_replacements == (("five million", 5000000),)
