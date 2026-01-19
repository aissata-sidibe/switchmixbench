def format_nli_prompt(premise: str, hypothesis: str):
    return (
        "Task: Natural Language Inference (NLI)\n"
        "Decide if the hypothesis is entailed by, contradicts, or is neutral with respect to the premise.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Answer with one label: entailment, contradiction, neutral."
    )
