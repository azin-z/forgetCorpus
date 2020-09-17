import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

def get_pos_tags(terms):
    return [pos_tag for _, pos_tag in nltk.pos_tag(terms, tagset='universal')]