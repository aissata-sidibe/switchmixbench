from dataclasses import dataclass
from switchmixbench.generate.code_switch_rules import mix_sentences
from switchmixbench.generate.noise_injectors import inject_informal_noise

@dataclass
class Example:
    uid: str
    task: str
    split: str
    variant: str  # clean or switchmix
    text_a: str
    text_b: str
    label: str

def build_switchmix_pair(uid, task, split, fr_a, en_a, fr_b=None, en_b=None, label=""):
    """
    For NLI: text_a=premise, text_b=hypothesis
    For QA:  text_a=context, text_b=question
    """
    clean_a = fr_a
    clean_b = fr_b if fr_b is not None else ""

    mixed_a = inject_informal_noise(
        mix_sentences(fr_a, en_a, seed=hash(uid) % 10000),
        seed=hash(uid) % 10000
    )

    mixed_b = ""
    if fr_b is not None and en_b is not None:
        mixed_b = inject_informal_noise(
            mix_sentences(fr_b, en_b, seed=(hash(uid) + 7) % 10000),
            seed=(hash(uid) + 7) % 10000
        )

    clean = Example(uid=uid, task=task, split=split, variant="clean", text_a=clean_a, text_b=clean_b, label=label)
    switchmix = Example(uid=uid, task=task, split=split, variant="switchmix", text_a=mixed_a, text_b=mixed_b, label=label)

    return [clean, switchmix]
