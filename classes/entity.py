import re
import socket
import requests
from classes.language_processing import LanguageProcessing


class EntityDetector:
    language_processing = LanguageProcessing()

    def __init__(self, regular_expressions=None, use_duckling=True, duckling_address='0.0.0.0',
                 duckling_port=8000):
        self.regular_expressions = {}
        if regular_expressions is None:
            self.regular_expressions = {'filename': r"[^\\ ]*\.(\w+)", 'number': r"[0-9]+"}
        else:
            for regular_expression_name, regular_expression in regular_expressions.items():
                self.regular_expressions[regular_expression_name] = regular_expression
        self.use_duckling = use_duckling
        self.duckling_address = duckling_address
        self.duckling_port = duckling_port

    def detect_entities(self, text, clean_text=None, spacy_tokens=None):
        entities = []
        entities.extend(self.__re_entities(text))
        entities.extend(self.__time_entity_using_duckling(text))
        entities.extend(self.__issue_number_entity_detector(text, clean_text, spacy_tokens))
        entities.extend(self.__issue_status_entity_detector(text, clean_text, spacy_tokens))
        return entities

    @staticmethod
    def __issue_number_entity_detector(text, clean_text, spacy_tokens):
        entity_name = 'issue_number'

        def re_patterns():
            entities = []
            regular_expressions = [r'issue\s\w{1,}\snumber\s\d{1,}',
                                   r'pull-request\s\w{1,}\snumber\s\d{1,}',
                                   r'pull\srequest\s\w{1,}\snumber\s\d{1,}',
                                   r'pr\s\w{1,}\snumber\s\d{1,}',
                                   r'issue\snumber\s\d{1,}',
                                   r'pr\snumber\s\d{1,}',
                                   r'pull\srequest\snumber\s\d{1,}',
                                   r'pull-request\snumber\s\d{1,}',
                                   r'issue\s\d{1,}',
                                   r'pull-request\s\d{1,}',
                                   r'pull\srequest\s\d{1,}',
                                   r'pr\s\d{1,}']
            # clean_text_str = " ".join(str(x) for x in clean_text)
            for regular_expression in regular_expressions:
                founded_entities = [entity.group() for entity in
                                    re.finditer(regular_expression, text, flags=re.IGNORECASE)]
                counter = 0
                for entity in founded_entities:
                    # TODO: The following line needs to be improved but for now, it is working based on the above REs.
                    entity = entity[entity.rfind(' ')+1:]  # it is based on the above REs.
                    start_index = text[counter:].find(entity) + counter
                    end_index = start_index + len(entity) - 1
                    entities.append({'body': entity, 'start': start_index, 'end': end_index, 'dim': entity_name})
                    counter = end_index
            return entities

        issue_number_entities = []
        issue_number_entities.extend(re_patterns())
        return [dict(t) for t in {tuple(d.items()) for d in issue_number_entities}]  # removing duplicate entries

    @staticmethod
    def __issue_status_entity_detector(text, clean_text, spacy_tokens):
        entity_name = 'issue_status'

        def re_patterns():
            entities = []
            # OPTIMIZE: the following list could be compacted!
            regular_expressions = [r'open\spr[s]?[\w|\s](?!\d)', r'opened\spr[s]?[\w|\s](?!\d)',
                                   r'close\spr[s]?[\w|\s](?!\d)', r'closed\spr[s]?[\w|\s](?!\d)',
                                   r'open\sissue[s]?[\w|\s](?!\d)', r'opened\sissue[s]?[\w|\s](?!\d)',
                                   r'close\sissue[s]?[\w|\s](?!\d)', r'closed\sissue[s]?[\w|\s](?!\d)',
                                   r'open\spull\srequest[s]?[\w|\s](?!\d)', r'opened\spull\srequest[s]?[\w|\s](?!\d)',
                                   r'close\spull\srequest[s]?[\w|\s](?!\d)', r'closed\spull\srequest[s]?[\w|\s](?!\d)',
                                   r'open\spull-request[s]?[\w|\s](?!\d)', r'opened\spull-request[s]?[\w|\s](?!\d)',
                                   r'close\spull-request[s]?[\w|\s](?!\d)', r'closed\spull-request[s]?[\w|\s](?!\d)',

                                   r'open\spr[s]?(?!.)', r'opened\spr[s]?(?!.)',
                                   r'close\spr[s]?(?!.)', r'closed\spr[s]?(?!.)',
                                   r'open\sissue[s]?(?!.)', r'opened\sissue[s]?(?!.)',
                                   r'close\sissue[s]?(?!.)', r'closed\sissue[s]?(?!.)',
                                   r'open\spull\srequest[s]?(?!.)', r'opened\spull\srequest[s]?(?!.)',
                                   r'close\spull\srequest[s]?(?!.)', r'closed\spull\srequest[s]?(?!.)',
                                   r'open\spull-request[s]?(?!.)', r'opened\spull-request[s]?(?!.)',
                                   r'close\spull-request[s]?(?!.)', r'closed\spull-request[s]?(?!.)',
                                   ]
            # clean_text_str = " ".join(str(x) for x in clean_text)
            for regular_expression in regular_expressions:
                founded_entities = [entity.group() for entity in
                                    re.finditer(regular_expression, text, flags=re.IGNORECASE)]
                counter = 0
                for entity in founded_entities:
                    # TODO: The following line needs to be improved but for now, it is working based on the above REs.
                    entity = entity[:entity.find(' ')]  # it is based on the above REs.
                    start_index = text[counter:].find(entity) + counter
                    end_index = start_index + len(entity) - 1
                    entities.append({'body': entity, 'start': start_index, 'end': end_index, 'dim': entity_name})
                    counter = end_index
            return entities

        issue_status_entities = []
        issue_status_entities.extend(re_patterns())
        return [dict(t) for t in {tuple(d.items()) for d in issue_status_entities}]  # removing duplicate entries

    def __time_entity_using_duckling(self, text):
        entities = []
        if self.use_duckling and self.__check_if_duckling_server_up():
            data = {
                'locale': 'en_US',
                'text': text,
                'dims': "[\"time\"]"
            }
            address = 'http://' + self.duckling_address + ':' + str(self.duckling_port) + '/' + 'parse'
            time_entities = requests.post(address, data=data)
            entities = entities + time_entities.json()
            return entities
        else:
            return []  # The duckling server is off in this case, or, use_duckling is False. So, we send an empty list.

    def __re_entities(self, text):
        entities = []
        for regular_expression_name, regular_expression in self.regular_expressions.items():
            re_entities = [entity.group() for entity in re.finditer(regular_expression, text, flags=re.IGNORECASE)]
            counter = 0
            for re_entity in re_entities:
                start_index = text[counter:].find(re_entity) + counter
                end_index = start_index + len(re_entity) - 1
                entities.append({'body': re_entity, 'start': start_index, 'end': end_index,
                                 'dim': regular_expression_name})
                counter = end_index
        return entities

    def __check_if_duckling_server_up(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.duckling_address, self.duckling_port))
        if result == 0:
            return True  # Server is up.
        else:
            return False  # Server is off.


# I wrote the following code just to show the usage and a sample for this class
if __name__ == "__main__":
    from classes.preprocessing import Preprocessing
    import spacy

    nlp = spacy.load('en_core_web_sm')
    preprocessing = Preprocessing(nlp_object=nlp, remove_special_char=False, remove_number=False,
                                  convert_number_to_string=False, deselect_stop_words=['not'])
    entityDetector = EntityDetector()
    sentences = ['Please tell me who removed My.exe, read.txt, and picture.jpeg files and also issue 449 in last week?',
                 'Who are the assignees on pulL Request with Number 456?',
                 'For this repo, show me the the number of Open Issues.',
                 'who opened issue 6744?']
    for sentence in sentences:
        clean_txt = preprocessing.do_preprocess(sentence)
        tokens = preprocessing.tokenize_text(sentence)
        print(sentence, '\n', entityDetector.detect_entities(sentence, clean_txt, tokens))

