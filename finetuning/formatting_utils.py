# formatting_utils.py
from typing import Dict, List, Any

def formatting_prompts_func(examples: Dict[str, List[Any]], tokenizer: Any) -> Dict[str, List[str]]:
    """
    Batched formatting function.
    Returns a dictionary with a "text" column containing the formatted strings.
    """
    output_texts = []

    lang_map = {"de": "German", "fr": "French", "it": "Italian"}

    # We iterate over the lists provided in the batch
    for i in range(len(examples['decision'])):
        decision_text = examples['decision'][i]
        headnote_text = examples['headnote'][i]
        lang_code = examples['headnote_language'][i]

        full_lang = lang_map.get(lang_code, "German")

        system_content = (
            "You are a legal expert specializing in Swiss Federal Supreme Court decisions "
            "with extensive knowledge of legal terminology and conventions in German, French, and Italian. "
            "Your task is to generate a headnote for a provided leading decision. "
            "A headnote is a concise summary that captures the key legal points and significance of the decision. "
            "It is not merely a summary of the content but highlights the aspects that make the decision \"leading\" "
            "and important for future legislation.\n\n"
            "When generating the headnote:\n"
            "1. Focus on the core legal reasoning and key considerations that establish the decision's significance.\n"
            "2. Include any relevant references to legal articles (prefixed with \"Art.\") and considerations "
            "(prefixed with \"E.\" in German or \"consid.\" in French/Italian).\n"
            "3. Use precise legal terminology and adhere to the formal and professional style typical of "
            "Swiss Federal Supreme Court headnotes.\n"
            "4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand "
            "in the specified language.\n\n"
            f"Your response should consist solely of the headnote in {full_lang}."
        )

        user_content = (
            "Leading decision:\n"
            f"```\n{decision_text}\n```\n"
            f"Generate a headnote in {full_lang} for the leading decision above."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": headnote_text}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)

    # CRITICAL CHANGE: Return a DICTIONARY, not a list.
    # This aligns with dataset.map(batched=True)
    return {"text": output_texts}
