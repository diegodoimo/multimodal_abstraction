"""Shared prompt text formatting helpers."""


def format_user_turn_text(question: str, guide_text: str = "") -> str:
    """User-turn text with optional guide and a terminal newline.

    Known inconsistency, kept so that the paper's similarity and label-overlap data reproduce: the
    representation experiments (similarity, label overlap, logit lens) of LLaVA, OneVision and
    InternVL2, and InternVL2's compositionality prompt, end the user turn with this newline; the
    benchmarks, interventions, text sweeps, the other compositionality prompts and every Cambrian
    prompt (utils/models/cambrian/cambrian_utils.py) do not, and neither does any model's own template.
    """
    question_text = str(question).rstrip()
    guide = str(guide_text).strip()
    if guide:
        return f"{question_text}\n{guide}\n"
    return f"{question_text}\n"
