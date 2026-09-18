from nlp_sql.tokenizer import PhraseKind, PhraseMatcher, Tokenizer


def test_tokenizer_splits_normalized_text() -> None:
    tokens = Tokenizer().tokenize("show orders greater than 1000")

    assert [token.text for token in tokens] == ["show", "orders", "greater", "than", "1000"]
    assert [token.position for token in tokens] == [0, 1, 2, 3, 4]


def test_phrase_matcher_prefers_longest_operator_phrase() -> None:
    tokens = Tokenizer().tokenize("show orders not equal to 1000")

    matches = PhraseMatcher().find_matches(tokens)

    assert len(matches) == 1
    assert matches[0].phrase == "not equal to"
    assert matches[0].kind == PhraseKind.OPERATOR
    assert matches[0].value == "!="
    assert matches[0].start == 2
    assert matches[0].end == 5


def test_phrase_matcher_finds_multiple_business_phrases() -> None:
    tokens = Tokenizer().tokenize("show top 10 orders from last month order by amount")

    matches = PhraseMatcher().find_matches(tokens)

    assert [(match.phrase, match.kind, match.value) for match in matches] == [
        ("top", PhraseKind.LIMIT, "TOP"),
        ("last month", PhraseKind.DATE, "LAST_MONTH"),
        ("order by", PhraseKind.ORDERING, "ORDER_BY"),
    ]
