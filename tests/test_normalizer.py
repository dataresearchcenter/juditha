from juditha.normalizer import (
    icu_normalize,
    tokenize,
)


def test_icu_normalize():
    # NFKC + casefold + Latin-ASCII folding
    assert icu_normalize("café") == "cafe"
    assert icu_normalize("ﬁ") == "fi"  # ligature decomposition
    assert icu_normalize("Straße") == "strasse"  # ß→ss
    assert icu_normalize("Müller") == "muller"  # ü→u
    assert icu_normalize("Élève") == "eleve"  # accent stripping
    assert icu_normalize("Łódź") == "lodz"  # special char folding
    assert icu_normalize("GmbH") == "gmbh"  # casefold


def test_tokenize_basic():
    tokens = tokenize("Jane Doe")
    assert len(tokens) == 2
    assert tokens[0].form == "jane"
    assert tokens[0].original == "Jane"
    assert tokens[1].form == "doe"
    assert tokens[1].original == "Doe"


def test_tokenize_offsets():
    tokens = tokenize("Hello World")
    assert tokens[0].start == 0
    assert tokens[0].end == 5
    assert tokens[1].start == 6
    assert tokens[1].end == 11


def test_tokenize_punctuation():
    tokens = tokenize("Mr. Smith, Jr.")
    forms = [t.form for t in tokens]
    assert "mr" in forms
    assert "smith" in forms
    assert "jr" in forms


def test_tokenize_unicode():
    tokens = tokenize("Müller Straße")
    # ICU folding: ü→u, ß→ss
    assert tokens[0].form == "muller"
    assert tokens[1].form == "strasse"
