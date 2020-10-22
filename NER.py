import anserini as anserini
import spacy
nlp = spacy.load("en_core_web_sm")


def get_entities(text):
    doc_text = nlp(text)
    entity_words = set()
    for ent in doc_text.ents:
        for word in anserini.tokenizeString(ent.text):
            if not ent.label_ == 'CARDINAL' and not ent.label_ == 'DATE':
                entity_words.add(word)
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return entity_words
