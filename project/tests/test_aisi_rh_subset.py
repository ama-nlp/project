"""The easy-problem filter that replaces AISI's hard one."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aisi_rh_subset as sub  # noqa: E402


def test_easy_keeps_low_rated_codeforces_problems_only():
    assert sub.is_easy({"cf_rating": 800, "difficulty": 7})
    assert sub.is_easy({"cf_rating": 1200, "difficulty": 8})
    assert not sub.is_easy({"cf_rating": 1300, "difficulty": 8})
    assert not sub.is_easy({"cf_rating": 0, "difficulty": 1})  # unrated: excluded
