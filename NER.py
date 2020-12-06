import anserini as anserini
import spacy
nlp = spacy.load("en_core_web_sm")


def get_entities(text):
    doc_text = nlp(text)
    map_ent_id = {"PERSON": 18, "NORP": 1, "FAC": 2, "ORG": 3, "GPE":4, "LOC":5 , "PRODUCT":6 , "EVENT":7 , "WORK_OF_ART":8, "LAW":9, "LANGUAGE":10, "DATE":11, "TIME":12, "PERCENT": 13, "MONEY": 14, "QUANTITY":15, "ORDINAL":16, "CARDINAL": 17 }
    entity_words = {}
    for ent in doc_text.ents:
        for word in anserini.tokenizeString(ent.text):
            if ent.label_ != "CARDINAL" and ent.label_ != "QUANTITY":
                entity_words[word] = map_ent_id[ent.label_]
    return entity_words

