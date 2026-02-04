"""Tests for rag.eval.judges module."""

from rag.eval.judges import make_gold_prompt


def test_make_gold_prompt_includes_metadata():
    """make_gold_prompt should format all placeholders including metadata."""
    prompt = make_gold_prompt(
        query="What is X?",
        expected_answer="X is a thing.",
        generated_answer="X is something.",
        query_type="factual",
        difficulty="easy",
        requires_synthesis=False,
    )

    # Verify core fields are present
    assert "What is X?" in prompt
    assert "X is a thing." in prompt
    assert "X is something." in prompt

    # Verify metadata fields are present (not raw placeholders)
    assert "{query_type}" not in prompt
    assert "{difficulty}" not in prompt
    assert "{requires_synthesis}" not in prompt

    # Verify actual values appear
    assert "factual" in prompt
    assert "easy" in prompt
    assert "False" in prompt


def test_make_gold_prompt_defaults():
    """make_gold_prompt should use defaults when metadata not provided."""
    prompt = make_gold_prompt(
        query="Test query",
        expected_answer="Expected",
        generated_answer="Generated",
    )

    # Should not raise, defaults should be used
    assert "Test query" in prompt
    assert "unknown" in prompt  # default query_type and difficulty
    assert "False" in prompt  # default requires_synthesis
