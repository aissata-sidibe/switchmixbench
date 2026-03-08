import switchmixbench.generate.build_pairs as bp


def test_build_switchmix_pair_basic_invariants():
    """Each seed example must yield one clean and one perturbed variant with label preservation."""
    uid = "nli_0001"
    fr_p = "Le chat est sur le tapis."
    en_p = "The cat is on the rug."
    fr_h = "Un animal est sur le tapis."
    en_h = "An animal is on the rug."
    label = "entailment"

    examples = bp.build_switchmix_pair(
        uid=uid,
        task="nli",
        split="test",
        fr_a=fr_p,
        en_a=en_p,
        fr_b=fr_h,
        en_b=en_h,
        label=label,
    )

    # One clean + one perturbed
    assert len(examples) == 2
    variants = {e.variant for e in examples}
    assert variants == {"clean", "switchmix"}

    # Labels are preserved
    assert {e.label for e in examples} == {label}

    # Pair-generation over multiple seeds yields 2x output size
    seeds = [
        ("id1", "A", "B", "C", "D", "entailment"),
        ("id2", "E", "F", "G", "H", "neutral"),
        ("id3", "I", "J", "K", "L", "contradiction"),
    ]
    all_out = []
    for s in seeds:
        all_out.extend(
            bp.build_switchmix_pair(
                uid=s[0],
                task="nli",
                split="test",
                fr_a=s[1],
                en_a=s[2],
                fr_b=s[3],
                en_b=s[4],
                label=s[5],
            )
        )
    assert len(all_out) == 2 * len(seeds)

