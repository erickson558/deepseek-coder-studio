from app.models.dataset import Message


def render_messages(messages: list[Message], tokenizer: object | None = None) -> str:
    serialised_messages = [message.model_dump() for message in messages]
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                serialised_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:  # noqa: BLE001
            pass

    rendered_parts: list[str] = []
    for message in messages:
        rendered_parts.append(f"[{message.role.upper()}]\n{message.content}")
    return "\n\n".join(rendered_parts)
