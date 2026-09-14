from optimization.query_compatibility import (
    classify_spj_compatibility,
)


def test_classifies_plain_spj_query() -> None:
    result = classify_spj_compatibility(
        "SELECT a.x, b.y FROM a, b WHERE a.id = b.id AND a.x > 1"
    )
    assert result == {"is_spj_compatible": True, "non_spj_features": []}


def test_allows_simple_result_aggregates() -> None:
    assert classify_spj_compatibility("SELECT MIN(a.x) FROM a") == {
        "is_spj_compatible": True,
        "non_spj_features": [],
    }
    assert classify_spj_compatibility("SELECT COUNT(DISTINCT a.x) FROM a") == {
        "is_spj_compatible": True,
        "non_spj_features": [],
    }


def test_reports_complex_aggregation() -> None:
    result = classify_spj_compatibility(
        "SELECT a.id, SUM(b.x) FROM a, b WHERE a.id = b.id "
        "GROUP BY a.id ORDER BY SUM(b.x)"
    )
    assert result == {"is_spj_compatible": True, "non_spj_features": []}

    result = classify_spj_compatibility("SELECT SUM(a.x * a.y) FROM a")
    assert result == {
        "is_spj_compatible": False,
        "non_spj_features": ["complex_aggregation"],
    }


def test_reports_subquery_and_ignores_keywords_in_literals_and_comments() -> None:
    result = classify_spj_compatibility(
        "SELECT a.x FROM a WHERE a.y IN (SELECT b.y FROM b) "
        "AND a.note = 'group by limit' -- order by"
    )
    assert result == {
        "is_spj_compatible": False,
        "non_spj_features": ["subquery"],
    }
