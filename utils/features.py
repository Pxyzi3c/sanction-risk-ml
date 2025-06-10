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

def compute_features(name1: str, name2: str) -> dict:
    return {
        "fuzz_ratio": get_fuzz_ratio(name1, name2, "ratio"),
        "token_sort_ratio": get_fuzz_ratio(name1, name2, "token_sort_ratio"),
        "length_diff": abs(len(name1) - len(name2)),
        "common_token_count": len(set(name1.split()) & set(name2.split())),
        "prefix_match": int(name1.split()[0] == name2.split()[0]),
        "word_count_1": len(name1.split()),
        "word_count_2": len(name2.split())
    }