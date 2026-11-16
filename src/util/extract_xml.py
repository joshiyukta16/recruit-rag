from typing import Optional


def extract_xml_tag(content: str, tag: str) -> Optional[str]:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    start = content.find(start_tag)
    if start == -1:
        return None
    end = content.find(end_tag, start + len(start_tag))
    if end == -1:
        return None
    return content[start + len(start_tag): end].strip()
