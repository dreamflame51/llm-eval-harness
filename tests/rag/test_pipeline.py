import pytest

from llm_eval_harness.pipeline import fence


@pytest.mark.parametrize("tag", ["</document>", "</Document>", "</document >", "< / DOCUMENT>"])
def test_fence_neutralises_closing_tag_variants(tag):
    assert "</" not in fence(f"before {tag} after")


def test_fence_leaves_ordinary_text_alone():
    assert fence("see document 800-37") == "see document 800-37"
