import re

def standardize_name(name: str) -> str:
    name = re.sub(r"[/-]", " ", name).upper()
    name = re.sub(r"[^A-Z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name