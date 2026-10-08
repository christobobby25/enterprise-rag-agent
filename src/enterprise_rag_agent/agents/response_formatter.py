import re


def clean_agent_response(text: str) -> str:
    """Remove model-generated reasoning tags from user-facing answers."""
    text = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"</?response>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()
