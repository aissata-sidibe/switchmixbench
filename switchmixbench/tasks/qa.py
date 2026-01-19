def format_qa_prompt(context: str, question: str):
    return (
        "Task: Question Answering\n"
        "Answer the question using only the provided context.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
