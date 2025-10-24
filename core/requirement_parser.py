import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

def extract_action_object_polarity(text: str):
    actions = set()
    objects = set()
    polarity = 1
    negation_words = {"not", "no", "never", "prohibit", "forbid", "deny", "without", "except", "cannot", "must not", "shall not"}
    text_l = text.lower()
    if any(w in text_l for w in negation_words):
        polarity = -1
    if _NLP:
        doc = _NLP(text)
        for token in doc:
            if token.pos_ == "VERB":
                actions.add(token.lemma_)
            if token.dep_ in {"dobj", "pobj", "nsubj", "attr"}:
                objects.add(token.lemma_)
    else:
        actions.update(re.findall(r"\b[a-z]{4,}ing\b|\b[a-z]{4,}ed\b|\b[a-z]{4,}\b", text_l))
        objects.update(re.findall(r"\b[a-z]{4,}\b", text_l))
    return actions, objects, polarity

def enrich_requirement(item):
    # item: dict or object with .text
    actions, objects, polarity = extract_action_object_polarity(item.text)
    item.actions = actions
    item.objects = objects
    item.polarity = polarity
    return item
