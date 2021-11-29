import spacy
from bs4 import BeautifulSoup
import unidecode
from word2number import w2n
import contractions


# The base class of all preprocessing steps
class Preprocessing:
    def __init__(self, nlp_object=None, remove_html=True, remove_extra_whitespace=True,
                 remove_accented_char=True, expand_contraction=True, to_lowercase=True,
                 remove_stop_word=True, remove_punctuation=True, remove_special_char=True, remove_number=True,
                 convert_number_to_string=True, lemmatization=True, deselect_stop_words=None,
                 specific_contractions=None):
        if nlp_object is None:
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = nlp_object
        self.remove_html = remove_html
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_accented_char = remove_accented_char
        self.expand_contraction = expand_contraction
        self.to_lowercase = to_lowercase
        self.remove_stop_word = remove_stop_word
        self.remove_punctuation = remove_punctuation
        self.remove_special_char = remove_special_char
        self.remove_number = remove_number
        self.convert_number_to_string = convert_number_to_string
        self.lemmatization = lemmatization
        if deselect_stop_words is None:
            deselect_stop_words = ['not']
        for word in deselect_stop_words:  # Excluding words from spacy stopwords list
            self.nlp.vocab[word].is_stop = False
        self.contraction = contractions
        if specific_contractions is not None:
            for contraction, expanded_from in specific_contractions.items():
                # Adding project-specific contractions: contractions.add('mychange', 'my change')
                self.contraction.add(contraction, expanded_from)

    def do_preprocess(self, text):
        preprocessed_text = ''
        if self.remove_html:
            preprocessed_text = self.__remove_html(text)
        if self.remove_extra_whitespace:
            preprocessed_text = self.__remove_extra_whitespace(preprocessed_text)
        if self.remove_accented_char:
            preprocessed_text = self.__remove_accented_char(preprocessed_text)
        if self.expand_contraction:
            preprocessed_text = self.__expand_contraction(preprocessed_text)
        if self.to_lowercase:
            preprocessed_text = self.__to_lowercase(preprocessed_text)
        tokens = self.tokenize_text(preprocessed_text)
        counter = 0
        clean_text = {}
        for token in tokens:
            if self.remove_stop_word and self.__is_stop_word(token):
                counter += 1
                continue
            if self.remove_punctuation and self.__is_punctuations(token):
                counter += 1
                continue
            if self.remove_special_char and self.__is_special_char(token):
                counter += 1
                continue
            if self.remove_number and self.__is_number(token):
                counter += 1
                continue
            if self.convert_number_to_string and self.__is_number(token):
                text = self.__convert_number_to_string(token.text)
                clean_text[counter] = text
                counter += 1
                continue
            lemmatization = self.__lemmatization(token)
            if self.lemmatization and lemmatization:
                clean_text[counter] = lemmatization
                counter += 1

        return clean_text

    def tokenize_text(self, text):
        return self.nlp(text)

    @staticmethod
    def __remove_html(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    @staticmethod
    def __remove_extra_whitespace(text):
        text = text.strip()
        return " ".join(text.split())

    @staticmethod
    def __remove_accented_char(text):
        text = unidecode.unidecode(text)
        return text

    @staticmethod
    def __to_lowercase(text):
        text = text.lower()
        return text

    @staticmethod
    def __is_stop_word(token):
        if token.is_stop and token.pos_ != 'NUM':
            return True

    @staticmethod
    def __is_punctuations(token):
        if token.pos_ == 'PUNCT':
            return True

    @staticmethod
    def __is_special_char(token):
        if token.pos_ == 'SYM':
            return True

    @staticmethod
    def __is_number(token):
        if token.pos_ == 'NUM' or token.text.isnumeric():
            return True

    def __expand_contraction(self, text):
        return self.contraction.fix(text, slang=False)

    @staticmethod
    def __convert_number_to_string(text):
        return w2n.word_to_num(text)

    @staticmethod
    def __lemmatization(token):
        if token.lemma_ != "-PRON-":
            return token.lemma_
        else:
            return False
