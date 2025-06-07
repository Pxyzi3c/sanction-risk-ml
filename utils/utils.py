from rapidfuzz import fuzz

def get_fuzz_ratio(name1: str, name2: str, ratio_type: str) -> float:
    name1 = " ".join(sorted(name1.split()))
    name2 = " ".join(sorted(name2.split()))

    try:
        if ratio_type == "ratio":
            return round(fuzz.ratio(name1, name2), 2)
        elif ratio_type == "token_sort_ratio":
            return fuzz.token_sort_ratio(name1, name2)
    except:
        return 0