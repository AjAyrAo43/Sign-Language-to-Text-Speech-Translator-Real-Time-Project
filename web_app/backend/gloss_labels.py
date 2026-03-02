# WLASL100 Gloss Labels
# The first 100 glosses from the WLASL dataset, ordered by gloss_number (0-99).
# This ordering matches the label indices used during training.

WLASL100_GLOSSES = [
    "book", "drink", "computer", "before", "chair",
    "go", "clothes", "who", "candy", "cousin",
    "deaf", "fine", "help", "no", "thin",
    "walk", "year", "yes", "all", "black",
    "cool", "finish", "hot", "like", "many",
    "mother", "now", "orange", "table", "thanksgiving",
    "what", "woman", "bed", "blue", "bowling",
    "can", "dog", "family", "fish", "graduate",
    "hat", "hearing", "kiss", "language", "later",
    "man", "shirt", "study", "tall", "teacher",
    "ugly", "white", "above", "accident", "apple",
    "bird", "birthday", "brown", "but", "cheat",
    "city", "cook", "cute", "dance", "dark",
    "doctor", "eat", "enjoy", "forget", "give",
    "glove", "green", "hard", "hungry", "jacket",
    "letter", "lost", "medicine", "meet", "name",
    "need", "paint", "paper", "pen", "pick",
    "pizza", "pull", "purse", "rain", "same",
    "school", "sign", "sorry", "spring", "story",
    "time", "want", "weather", "work", "wrong",
]

def get_gloss(index: int) -> str:
    """Get the gloss word for a given class index."""
    if 0 <= index < len(WLASL100_GLOSSES):
        return WLASL100_GLOSSES[index]
    return f"unknown_{index}"

def get_all_glosses() -> list:
    """Return the full list of glosses."""
    return WLASL100_GLOSSES
