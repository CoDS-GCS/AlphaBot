from textblob import TextBlob
from spacy import displacy


class LanguageProcessing:
    def __init__(self):
        pass

    @staticmethod
    def spacy_entity_detector(tokens):
        entities = [(i, i.label_, i.label) for i in tokens.ents]
        return entities

    @staticmethod
    def spacy_dependency_parser_in_noun_phrases(tokens):
        dependencies = []
        for chunk in tokens.noun_chunks:
            dependencies.append([chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text])
        return dependencies

    @staticmethod
    def spacy_dependency_parser(tokens):
        dependencies = []
        for token in tokens:
            dependencies.append([token.text, token.tag_, token.head.text, token.dep_])
        return dependencies

    @staticmethod
    def spacy_display_render(tokens, style, jupyter=True):  # style = 'dep' or 'ent'
        displacy.render(tokens, style=style, jupyter=jupyter)

    @staticmethod
    def textblob_noun_phrase_detector(text):
        text_blob_processed = TextBlob(text)
        noun_phrases = text_blob_processed.noun_phrases
        return noun_phrases

    @staticmethod
    def textblob_sentiment_analyzer(text):
        text_blob_processed = TextBlob(text)
        sentiment = text_blob_processed.sentiment
        return sentiment

    @staticmethod
    def textblob_spell_corrector(text):
        text_blob_processed = TextBlob(text)
        corrected = text_blob_processed.correct()
        return corrected

    @staticmethod
    def is_subject(tok):
        subject_deps = {"csubj", "nsubj", "nsubjpass"}
        return tok.dep_ in subject_deps

    @staticmethod
    def is_wh_question(tokens):
        # "What is your name?"
        wh_tags = ["WDT", "WP", "WP$", "WRB"]
        wh_words = [t for t in tokens if t.tag_ in wh_tags]
        sent_initial_is_wh = wh_words and wh_words[0].i == 0

        # Include pied-piped constructions: "To whom did she read the article?"
        pied_piped = wh_words and wh_words[0].head.dep_ == "prep"

        # Exclude pseudoclefts: "What you say is impossible."
        pseudocleft = wh_words and wh_words[0].head.dep_ in ["csubj", "advcl"]
        if pseudocleft:
            return False

        return sent_initial_is_wh or pied_piped

    @staticmethod
    def is_polar_question(tokens):  # Is that for real?: polar
        root = [t for t in tokens if t.dep_ == "ROOT"][0]  # every spaCy parse as a root token.
        subj = [t for t in root.children if LanguageProcessing.is_subject(t)]

        if LanguageProcessing.is_wh_question(tokens):
            return False

        # Type I: In a non-copular sentence, "is" is an aux.
        # "Is she using spaCy?" or "Can you read that article?"
        aux = [t for t in root.lefts if t.dep_ == "aux"]
        if subj and aux:
            return aux[0].i < subj[0].i

        # Type II: In a copular sentence, "is" is the main verb.
        # "Is the mouse dead?"
        root_is_inflected_copula = root.pos_ == "VERB" and root.tag_ != "VB"
        if subj and root_is_inflected_copula:
            return root.i < subj[0].i

        return False

    @staticmethod
    def get_question_type(tokens):
        is_wh = LanguageProcessing.is_wh_question(tokens)
        is_polar = LanguageProcessing.is_polar_question(tokens)
        if is_wh:
            return 'wh'  # wh question
        elif is_polar:
            return 'polar'  # polar
        else:
            return 'nq'  # not a question

    @staticmethod
    def exist_in_text(subtext_to_find, text):
        if text.lower().find(subtext_to_find.lower()) != -1:
            return True
        else:
            return False
