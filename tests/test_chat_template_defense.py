"""Integration test: the defensive system prompt actually reaches each model.

The paper (Sections 4.4, 8.5, Appendix G.6) asserts that under the
``with_defense`` condition the defensive prompt text reaches every model -- in
the *system* slot for the three models whose chat templates accept a system
role (Llama-3.2-3B, Phi-3-mini, Qwen2.5-3B) and in the *first user message* for
Gemma-2 (whose template rejects a system role, so the harness merges the system
content into the first user turn via ``prepare_messages``). This is the single
check that catches the highest-severity silent failure in the audit: a
``with_defense`` run that, because of a placement bug or a chat-template change,
sent the model no defensive text at all -- which would masquerade as a "defense
that does nothing" rather than as a bug.

Two layers:

* ``test_placement_*`` -- pure-function checks on ``build_messages`` /
  ``prepare_messages`` (the backend-agnostic placement logic shared by the
  transformers and Ollama backends). No downloads; always run.
* ``test_rendered_template_contains_defense`` -- the full check Appendix G.6
  describes: render each model's *actual* chat template and assert the
  defensive text survives. This needs the tokenizers (network + gated-model
  access), so it is opt-in: set ``RUN_TOKENIZER_TESTS=1`` to run it.

Run directly (``python tests/test_chat_template_defense.py``) or under pytest.
"""

from __future__ import annotations

import importlib.util
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_exp = _import("experiment_under_test", os.path.join(_ROOT, "05_experiment.py"))


SYSTEM_ROLE_MODELS = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)
GEMMA = "google/gemma-2-2b-it"
USER_PROMPT = "How do I water my tomato plants?"


def _defense_text() -> str:
    path = os.path.join(_ROOT, "defensive_prompts", "primary.txt")
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def test_placement_system_role_models():
    """System-role models carry the defense in a single system message, intact."""
    defense = _defense_text()
    for model_id in SYSTEM_ROLE_MODELS:
        prepared = _exp.prepare_messages(
            _exp.build_messages(USER_PROMPT, "with_defense", defense), model_id)
        system = [m for m in prepared if m["role"] == "system"]
        assert len(system) == 1, f"{model_id}: expected exactly one system message"
        assert system[0]["content"] == defense, \
            f"{model_id}: system message is not the verbatim defensive prompt"
        assert any(USER_PROMPT in m["content"]
                   for m in prepared if m["role"] == "user"), \
            f"{model_id}: user prompt missing after preparation"


def test_placement_gemma_merges_into_first_user_turn():
    """Gemma has no system role: the defense is prepended to the first user msg."""
    defense = _defense_text()
    prepared = _exp.prepare_messages(
        _exp.build_messages(USER_PROMPT, "with_defense", defense), GEMMA)
    assert all(m["role"] != "system" for m in prepared), \
        "Gemma must never be handed a system role"
    first_user = next(m for m in prepared if m["role"] == "user")
    assert defense in first_user["content"], \
        "defensive text must survive the system->user merge for Gemma"
    assert USER_PROMPT in first_user["content"], \
        "user prompt must survive the system->user merge for Gemma"


def test_no_defense_has_no_defensive_text():
    """Under no_defense, the defensive text appears in no message, for any model."""
    defense = _defense_text()
    for model_id in SYSTEM_ROLE_MODELS + (GEMMA,):
        prepared = _exp.prepare_messages(
            _exp.build_messages(USER_PROMPT, "no_defense", defense), model_id)
        assert all(m["role"] != "system" for m in prepared), \
            f"{model_id}: no_defense must not introduce a system message"
        assert all(defense not in m["content"] for m in prepared), \
            f"{model_id}: defensive text leaked into the no_defense condition"


def test_rendered_template_contains_defense():
    """Full Appendix G.6 check: render each model's actual chat template and
    assert the defensive text survives. Opt-in (needs network + gated-model
    access): set RUN_TOKENIZER_TESTS=1. Skipped otherwise."""
    if os.environ.get("RUN_TOKENIZER_TESTS") != "1":
        print("  [skip] test_rendered_template_contains_defense "
              "(set RUN_TOKENIZER_TESTS=1 to render real chat templates)")
        return
    from transformers import AutoTokenizer

    defense = _defense_text()
    revisions = dict(_exp.DEFAULT_MODELS)
    for model_id in SYSTEM_ROLE_MODELS + (GEMMA,):
        tok = AutoTokenizer.from_pretrained(
            model_id, revision=revisions.get(model_id))
        prepared = _exp.prepare_messages(
            _exp.build_messages(USER_PROMPT, "with_defense", defense), model_id)
        rendered = tok.apply_chat_template(
            prepared, add_generation_prompt=True, tokenize=False)
        assert defense in rendered, \
            f"{model_id}: defensive text missing from the rendered chat template"


if __name__ == "__main__":
    test_placement_system_role_models()
    test_placement_gemma_merges_into_first_user_turn()
    test_no_defense_has_no_defensive_text()
    test_rendered_template_contains_defense()
    print("chat-template defense-placement tests passed")
