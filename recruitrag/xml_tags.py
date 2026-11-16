"""Pull the text out of an <tag>...</tag> pair in an LLM reply."""
import re
from typing import Optional


def tag_text(reply: str, tag: str) -> Optional[str]:
    """Return the stripped contents of the first <tag> element, or None when absent."""
    name = re.escape(tag)
    found = re.search(rf"<{name}>(.*?)</{name}>", reply, flags=re.DOTALL | re.IGNORECASE)
    return found.group(1).strip() if found else None
