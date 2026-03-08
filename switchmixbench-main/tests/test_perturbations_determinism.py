from switchmixbench.generate.perturbations import (
    apply_switchmix_perturbation,
    inject_informal_noise_with_meta,
    mix_sentences_with_meta,
)


def test_mix_sentences_with_meta_deterministic():
    base = "Le chat est sur le tapis."
    insert = "The cat is on the rug."
    seed = 123

    a, meta_a = mix_sentences_with_meta(base, insert, seed=seed)
    b, meta_b = mix_sentences_with_meta(base, insert, seed=seed)

    assert a == b
    assert meta_a == meta_b


def test_inject_informal_noise_with_meta_deterministic():
    text = "you are ne pas here"
    seed = 7

    a, meta_a = inject_informal_noise_with_meta(text, p=0.5, seed=seed)
    b, meta_b = inject_informal_noise_with_meta(text, p=0.5, seed=seed)

    assert a == b
    assert meta_a == meta_b


def test_apply_switchmix_perturbation_deterministic():
    base = "Le chat est sur le tapis."
    insert = "The cat is on the rug."
    seed = 99

    a, meta_a = apply_switchmix_perturbation(base, insert, seed=seed)
    b, meta_b = apply_switchmix_perturbation(base, insert, seed=seed)

    assert a == b
    assert meta_a == meta_b

