def format_nli_prompt(premise: str, hypothesis: str) -> str:
    return (
        "You are an NLI classifier.\n"
        "Return exactly ONE label from: entailment, contradiction, neutral.\n"
        "Do not explain.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Label:"
    )

