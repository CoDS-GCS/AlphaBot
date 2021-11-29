from snorkel.labeling import labeling_function
from classes.entity import EntityDetector
from classes.preprocessing import Preprocessing
from classes.language_processing import LanguageProcessing
import spacy
from snorkel.preprocess import preprocessor


class LabelingFunctions:
    _ABSTAIN = -1
    _MOST_RECENT_PRS = 0
    _NUMBER_OF_COMMITS_IN_BRANCH = 1
    _TOP_CONTRIBUTORS = 2
    _INTIAL_COMMIT = 3
    _MOST_RECENT_ISSUES = 4
    _LONGEST_OPEN_ISSUE = 5
    _LATEST_COMMIT = 6
    _CONTRIBUTIONS_BY_DEVELOPER = 7
    _ACTIVITY_REPORT = 8
    _NUMBER_OF_PRS = 9
    _PR_ASSIGNEES = 10
    _PR_CREATION_DATE = 11
    _PR_CLOSING_DATE = 12
    _PR_CLOSER = 13
    _NUMBER_OF_ISSUES = 14
    _NUMBER_OF_FORKS = 15
    _NUMBER_OF_COMMITS = 16
    _NUMBER_OF_BRANCHES = 17
    _LIST_RELEASES = 18
    _LIST_COLLABORATORS = 19
    _LIST_BRANCHES = 20
    _ISSUE_RELATED_COMMITS = 21
    _FILE_CREATOR = 22
    _LARGEST_FILES = 23
    _LONGEST_OPEN_PR = 24
    _ISSUE_ASSIGNEES = 25
    _COMMITS_IN_PR = 26
    _REPOSITORY_TOPICS = 27
    _REPOSITORY_OWNER = 28
    _REPOSITORY_LICENSE = 29
    _REPOSITORY_CREATION_DATE = 30
    _PR_CREATOR = 31
    _PR_CONTRIBUTORS = 32
    _NUMBER_OF_WATCHERS = 33
    _NUMBER_OF_SUBSCRIBERS = 34
    _NUMBER_OF_STARS = 35
    _NUMBER_OF_DOWNLOADS = 36
    _NUMBER_OF_COLLABORATORS = 37
    _LIST_LANGUAGES = 38
    _LATEST_RELEASE = 39
    _LATEST_COMMIT_IN_BRANCH = 40
    _LAST_DEVELOPER_TO_TOUCH_A_FILE = 41
    _ISSUE_CREATOR = 42
    _ISSUE_CREATION_DATE = 43
    _ISSUE_CONTRIBUTORS = 44
    _ISSUE_CLOSING_DATE = 45
    _ISSUE_CLOSER = 46
    _INTIAL_COMMIT_IN_BRANCH = 47
    _FILES_CHANGED_BY_PR = 48
    _DEVELOPERS_WITH_MOST_OPEN_ISSUES = 49
    _DEFAULT_BRANCH = 50
    _MAIN_PROGRAMMING_LANGUAGE = 51
    CARDINALITY = 52  # KEEP THIS UPDATED!
    nlp = spacy.load('en_core_web_md')
    preprocessing = Preprocessing(nlp_object=nlp, remove_special_char=False, remove_number=False,
                                  convert_number_to_string=False, deselect_stop_words=['not', 'last', 'first',
                                                                                       'when', 'who', 'most', 'top'])
    entity_detector = EntityDetector(regular_expressions={'file': r"[^\\ ]*\.(\w+)", 'number': r"[0-9]+"},
                                     use_duckling=True)
    language_processing = LanguageProcessing()

    def __init__(self):
        pass

    @staticmethod
    def has_proper_singular_noun(command):
        for token in command.tokens:
            if token.tag_ == 'NNP' and token.pos_ == 'PROPN' and token.text not in ['repo']:
                return True
        return False

    @staticmethod
    def not_a_verb(command, keyword):
        for token in command.tokens:
            if token.lemma_ == keyword and token.pos_ == 'VERB':
                return False
        return True

    @staticmethod
    def has_any(command, list_of_entities):  # has any of these (at least one of them) (empty list NOT allowed) (OR)
        return any(elem in [entity['dim'] for entity in command.entities] for elem in list_of_entities)
        pass

    @staticmethod
    def does_not_have(command, list_of_entities):  # does not have these entities (empty list allowed) (AND)
        return all(elem not in [entity['dim'] for entity in command.entities] for elem in list_of_entities)

    @staticmethod
    def has(command, list_of_entities):  # has these entities (empty list not allowed) (AND)
        return all(elem in [entity['dim'] for entity in command.entities] for elem in list_of_entities)

    @staticmethod
    def has_only(command, list_of_entities):  # has only (just these entities) (empty list allowed) (OR)
        # (if one or more entities are not in command.entities, it will return true also).
        return all(elem['dim'] in list_of_entities for elem in command.entities)

    @staticmethod
    def has_and_has_only(command, list_of_entities):
        if LabelingFunctions.has(command, list_of_entities) \
                and \
                LabelingFunctions.has_only(command, list_of_entities):
            return True
        else:
            return False

    @staticmethod
    def has_no_entity(command):
        if len(command.entities) > 0:
            return False
        else:
            return True

    @staticmethod
    def text_starts_with(subtext_to_find, command):
        text = command.text
        if text.lower().startswith(subtext_to_find.lower()):
            return True
        else:
            return False

    @staticmethod
    def found_in_text(subtext_to_find, command):
        text = command.text
        if text.lower().find(subtext_to_find.lower()) != -1:
            return True
        else:
            return False

    @staticmethod
    def check_not_in_clean_text(command, list_of_keywords):
        clean_text = command.clean_text
        values = clean_text.values()
        flag = True
        for keyword in list_of_keywords:
            if type(keyword) is str:
                if keyword in values:
                    flag = False
                    break
            else:
                raise Exception('invalid input type for check_not_in_clean_text() method')
        return flag

    @staticmethod
    def check_in_clean_text(command, list_of_keywords):
        clean_text = command.clean_text
        values = clean_text.values()
        flag = False
        for element in list_of_keywords:  # x or y or (z and k)
            if type(element) is str:  # single keyword: x, y
                if element in values:
                    flag = True  # simulating OR
            elif type(element) is list:  # list: (z, k)
                inner_flag = True
                for keyword in element:
                    if keyword not in values:  # simulating AND
                        inner_flag = False
                if inner_flag:  # simulating OR for the this list
                    flag = True
            else:
                raise Exception('invalid input type for check_in_clean_text() method')
        return flag

    @staticmethod
    def is_wh_question(command):
        if LabelingFunctions.language_processing.get_question_type(command.tokens) == 'wh':
            return True
        else:
            return False

    @staticmethod
    def is_polar_question(command):
        if LabelingFunctions.language_processing.get_question_type(command.tokens) == 'polar':
            return True
        else:
            return False

    @staticmethod
    def is_no_question(command):
        if LabelingFunctions.language_processing.get_question_type(command.tokens) == 'nq':
            return True
        else:
            return False

    @staticmethod
    @preprocessor(memoize=True)
    def preprocess_command(command):
        text = command.text
        command.tokens = LabelingFunctions.preprocessing.tokenize_text(text)
        command.clean_text = LabelingFunctions.preprocessing.do_preprocess(text)
        command.entities = LabelingFunctions.entity_detector.detect_entities(text, command.clean_text, command.tokens)
        command.question_type = LabelingFunctions.language_processing.get_question_type(command.tokens)
        return command

    # Condition 1: has no entities.
    # Condition 2: 'branch' in the clean text of command.
    # Condition 3: ('main' or 'base' or 'default') in the clean text of command.
    # Intent: DEFAULT_BRANCH
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def default_branch_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['main', 'base', 'default']):
            return LabelingFunctions._DEFAULT_BRANCH
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_and_has_only([issue_status]) (means: has issue_status and this is the only entity).
    # Condition 2: is a wh question.
    # Condition 3: text starts with who.
    # Intent: DEVELOPERS_WITH_MOST_OPEN_ISSUES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def developers_with_most_open_issues_1(command):
        if LabelingFunctions.has_and_has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.is_wh_question(command) \
                and \
                LabelingFunctions.text_starts_with('who', command):
            return LabelingFunctions._DEVELOPERS_WITH_MOST_OPEN_ISSUES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_and_has_only([issue_status]) (means: has issue_status and this is the only entity).
    # Condition 2: 'developer' in the clean text of command.
    # Intent: DEVELOPERS_WITH_MOST_OPEN_ISSUES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def developers_with_most_open_issues_2(command):
        if LabelingFunctions.has_and_has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['developer']):
            return LabelingFunctions._DEVELOPERS_WITH_MOST_OPEN_ISSUES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only([issue_number, number, time)] and (has(issue_number) or has(number)).
    # Condition 2: 'issue' NOT in the clean text of command.
    # Condition 3: ('change' or 'affect' or 'diff' or 'touch') in the clean text of command.
    # Condition 4: ('file') in the clean text of command.
    # Intent: FILES_CHANGED_BY_PR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def files_changed_by_pr_1(command):
        if LabelingFunctions.has_only(command, ['issue_number', 'number', 'time']) \
                and \
                (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['change', 'affect', 'diff', 'touch']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['file']):
            return LabelingFunctions._FILES_CHANGED_BY_PR
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only(number, time) entity: means that we may have the 'number' or 'time', but we have to not have
    #   other entities.
    # Condition 2: 'issue' NOT in the clean text of command.
    # Condition 3: 'commit' in the clean text of command.
    # Condition 4: ('1st' or 'first' or 'initial' or 'intial' or 'inital') in the clean text of command.
    # Condition 5: ('branch' or 'master') in the clean text of command.
    # Intent: INTIAL_COMMIT_IN_BRANCH
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def intial_commit_in_branch_1(command):
        if LabelingFunctions.has_only(command, ['number', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['1st', 'first', 'initial', 'intial', 'inital']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch', 'master']):
            return LabelingFunctions._INTIAL_COMMIT_IN_BRANCH
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has(issue_number) and has_only([issue_number, number]).
    # Condition 2: ('pr' or 'pull' or 'request', 'when', 'date', 'time') NOT in the clean text of command.
    # Condition 3: ('close' or 'merge' or 'resolve' or 'resolver' or 'closing') in the clean text of command.
    # Intent: ISSUE_CLOSER
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_closer_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request', 'when', 'date', 'time']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'resolve', 'resolver', 'closing']):
            return LabelingFunctions._ISSUE_CLOSER
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has(issue_number) and has_only([issue_number, number]).
    # Condition 2: ('close' or 'closing' or 'merge' or 'resolve') in the clean text of command.
    # Condition 3: ('pr' and 'pull' and 'request') NOT in the clean text of command.
    # Condition 4: text starts with when.
    # Intent: ISSUE_CLOSING_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_closing_date_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'resolve', 'closing']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request']) \
                and \
                LabelingFunctions.text_starts_with('when', command):
            return LabelingFunctions._ISSUE_CLOSING_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has(issue_number) and has_only([issue_number, number]).
    # Condition 2: ('close' or 'closing' or 'merge' or 'resolve') in the clean text of command.
    # Condition 3: ('date' or 'time') in the clean text of command.
    # Condition 4: ('pr' and 'pull' and 'request') NOT in the clean text of command.
    # Intent: ISSUE_CLOSING_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_closing_date_2(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'resolve', 'closing']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['date', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request']):
            return LabelingFunctions._ISSUE_CLOSING_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('contribute' or 'contributor' or 'comment' or 'commenter' or 'help' or 'work' or 'touch')
    #   in the clean text of command.
    # Condition 3: ('pr', 'pull', 'request', 'most', 'high', 'top') NOT in the clean text of command.
    # Intent: ISSUE_CONTRIBUTORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_contributors_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['contribute', 'contributor', 'comment', 'commenter',
                                                                'help', 'work', 'touch']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request', 'most', 'high', 'top']):
            return LabelingFunctions._ISSUE_CONTRIBUTORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('creation' or 'create' or 'open') in the clean text of command.
    # Condition 3: ('date' or 'time') in the clean text of command.
    # Condition 4: ('pr' and 'pull' and 'request') NOT in the clean text of command.
    # Intent: ISSUE_CREATION_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_creation_date_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['creation', 'create', 'open']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['date', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request']):
            return LabelingFunctions._ISSUE_CREATION_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('creation' or 'create' or 'open') in the clean text of command.
    # Condition 3: ('pr' and 'pull' and 'request') NOT in the clean text of command.
    # Condition 4: text starts with when.
    # Intent: ISSUE_CREATION_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_creation_date_2(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['creation', 'create', 'open']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request']) \
                and \
                LabelingFunctions.text_starts_with('when', command):
            return LabelingFunctions._ISSUE_CREATION_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('create' or 'creator' or 'open' or 'opne' or 'opener') in the clean text of command.
    # Condition 3: ('pr' and 'pull' and 'request', 'when', 'date', 'time') NOT in the clean text of command.
    # Intent: ISSUE_CREATOR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_creator_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['create', 'creator', 'open', 'opne', 'opener']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'request', 'when', 'date', 'time']):
            return LabelingFunctions._ISSUE_CREATOR
        else:
            return LabelingFunctions._ABSTAIN

    # TODO: the file entity detector should be better: "branch rel-1.3.0" will be detected as a file due to the format!
    # Condition 1: has(file) and has_only([file, number]).
    # Condition 2: ('touch' or 'edit' or 'change' or 'commit') in the clean text of command.
    # Condition 3: 'how many' not in the text.
    # Intent: LAST_DEVELOPER_TO_TOUCH_A_FILE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def last_developer_to_touch_a_file_1(command):
        if LabelingFunctions.has(command, ['file']) \
                and \
                LabelingFunctions.has_only(command, ['file', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['touch', 'edit', 'change', 'commit']) \
                and \
                not LabelingFunctions.found_in_text('how many', command):
            return LabelingFunctions._LAST_DEVELOPER_TO_TOUCH_A_FILE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only([number]).
    # Condition 2: 'commit' in the clean text of command.
    # Condition 3: ('last' or 'late' or 'recent') in the clean text of command.
    # Condition 4: ('branch' or 'master') in the clean text of command.
    # Condition 5: 'issue' NOT in the clean text of command.
    # Intent: LATEST_COMMIT_IN_BRANCH
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def latest_commit_in_branch_1(command):
        if LabelingFunctions.has_only(command, ['number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['last', 'late', 'recent']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch', 'master']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._LATEST_COMMIT_IN_BRANCH
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('late' or 'new' or 'last') in the clean text of command.
    # Condition 3: 'release' in the clean text of command.
    # Intent: LATEST_RELEASE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def latest_release_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['late', 'new', 'last']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['release']):
            return LabelingFunctions._LATEST_RELEASE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('language' or 'technology') in the clean text of command.
    # Condition 3: 'main', 'most', 'dominant' NOT in the clean text of command.
    # Intent: LIST_LANGUAGES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def list_languages_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['language', 'technology']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['main', 'most', 'dominant']):
            return LabelingFunctions._LIST_LANGUAGES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('language' or 'technology') in the clean text of command.
    # Condition 3: ('main' or 'most' or 'dominant') in the clean text of command.
    # Condition 4: 'languages' (Plural form) NOT in the text of command.
    # Intent: MAIN_PROGRAMMING_LANGUAGE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def main_programming_language_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['language', 'technology']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['main', 'most', 'dominant']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['languages']):
            return LabelingFunctions._MAIN_PROGRAMMING_LANGUAGE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('collaborator' or 'developer' or 'contributor' or 'contributer') in the clean text of command.
    # Condition 3: 'number' in the clean text of command.
    # Condition 4: is_no_question (is a nq sentence).
    # Intent: NUMBER_OF_COLLABORATORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_collaborators_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['collaborator', 'developer', 'contributor',
                                                                'contributer']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['number']) \
                and \
                LabelingFunctions.is_no_question(command):
            return LabelingFunctions._NUMBER_OF_COLLABORATORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('collaborator' or 'developer' or 'contributor' or 'contributer') in the clean text of command.
    # Condition 3: is_wh_question.
    # Condition 4: text starts with 'how many' but NOT 'how many commit'.
    # Condition 5: if the command has 'commit' keyword, it is not a verb.
    # Intent: NUMBER_OF_COLLABORATORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_collaborators_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['collaborator', 'developer', 'contributor',
                                                                'contributer']) \
                and \
                LabelingFunctions.is_wh_question(command) \
                and \
                LabelingFunctions.text_starts_with('how many', command) and \
                not LabelingFunctions.found_in_text('how many commit', command) \
                and \
                LabelingFunctions.not_a_verb(command, ['commit']):
            return LabelingFunctions._NUMBER_OF_COLLABORATORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: 'download' in the clean text of command.
    # Intent: NUMBER_OF_DOWNLOADS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_downloads_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['download']):
            return LabelingFunctions._NUMBER_OF_DOWNLOADS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: 'star' in the clean text of command.
    # Intent: NUMBER_OF_STARS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_stars_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['star']):
            return LabelingFunctions._NUMBER_OF_STARS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('subscribe' or 'subscriber') in the clean text of command.
    # Intent: NUMBER_OF_SUBSCRIBERS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_subscribers_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['subscribe', 'subscriber']):
            return LabelingFunctions._NUMBER_OF_SUBSCRIBERS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_no_entity (means no entity at all).
    # Condition 2: ('watcher' or 'watch') in the clean text of command.
    # Intent: NUMBER_OF_WATCHERS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_watchers_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['watcher', 'watch']):
            return LabelingFunctions._NUMBER_OF_WATCHERS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('contribute' or 'contributor' or 'comment', 'commenter', 'help') in the clean text of command.
    # Condition 3: ('issue', 'most', 'high', 'top')  NOT in the clean text of command.
    # Intent: PR_CONTRIBUTORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_contributors_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['contribute', 'contributor', 'comment', 'commenter',
                                                                'help']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'most', 'high', 'top']):
            return LabelingFunctions._PR_CONTRIBUTORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: (has(issue_number) or has(number)) and has_only([issue_number, number]).
    # Condition 2: ('create', 'creator' or 'start' or 'open', 'author', 'opener', 'opne') in the clean text of command.
    # Condition 3: ('issue', 'when', 'date', 'time')  NOT in the clean text of command.
    # Intent: PR_CREATOR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_creator_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['create', 'creator', 'start', 'open', 'author',
                                                                'opener', 'opne']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'when', 'date', 'time']):
            return LabelingFunctions._PR_CREATOR
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: No entities.
    # Condition 2: ('create' or 'creation' or 'start') in the clean text of command.
    # Condition 3: ('developer', 'file') NOT in the clean text of command.
    # Intent: REPOSITORY_CREATION_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def repository_creation_date_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['create', 'creation', 'start']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['developer', 'file', 'release']):
            return LabelingFunctions._REPOSITORY_CREATION_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: No entities.
    # Condition 2: ('license' or 'licence') in the clean text of command.
    # Intent: REPOSITORY_LICENSE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def repository_license_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['license', 'licence']):
            return LabelingFunctions._REPOSITORY_LICENSE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: No entities.
    # Condition 2: 'owner' or 'own' or 'ownership' in the clean text of command.
    # Condition 3: 'topic' not in the clean text.
    # Intent: REPOSITORY_OWNER
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def repository_owner_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['owner', 'own', 'ownership']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['topic']):
            return LabelingFunctions._REPOSITORY_OWNER
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: No entities.
    # Condition 2: 'belong' in the clean text of command.
    # Condition 3: ('repo' or 'repository') in the clean text of command.
    # Condition 4: 'topic' not in the clean text.
    # Intent: REPOSITORY_OWNER
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def repository_owner_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['belong']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['repo', 'repository']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['topic']):
            return LabelingFunctions._REPOSITORY_OWNER
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: No entities.
    # Condition 2: 'topic' in the clean text of command.
    # Intent: REPOSITORY_TOPICS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def repository_topics_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['topic']):
            return LabelingFunctions._REPOSITORY_TOPICS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: does_not_have 'number' and 'issue_number' entities.
    # Condition 2: 'issue' in the clean text of command.
    # Condition 3: 'new' or 'late' or 'recent' in the clean text of command.
    # Intent: MOST_RECENT_ISSUES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def most_recent_issues_1(command):
        if LabelingFunctions.does_not_have(command, ['number', 'issue_number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['issue']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['new', 'late', 'recent']):
            return LabelingFunctions._MOST_RECENT_ISSUES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: does_not_have 'issue_status', 'issue_number', and 'filename' entities.
    # Condition 2: 'commit' in the clean text of command.
    # Condition 3: '1st' or 'first' or 'initial' or 'intial' in the clean text of command.
    # Condition 4: 'issue', 'branch' NOT in the clean text of command.
    # Intent: INTIAL_COMMIT
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def intial_commit_1(command):
        if LabelingFunctions.does_not_have(command, ['issue_status', 'issue_number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['1st', 'first', 'initial', 'intial', 'inital']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'branch']):
            return LabelingFunctions._INTIAL_COMMIT
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: does_not_have ['time', 'issue_status', 'issue_number', 'filename'] entities.
    # Condition 2: 'commit' in the clean text of command.
    # Condition 3: 'last' or 'recent' or 'late' in the clean text of command.
    # Condition 4: 'issue', 'branch', and 'master' NOT in the clean text of command.
    # Intent: LATEST_COMMIT
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def latest_commit_1(command):
        if LabelingFunctions.does_not_have(command, ['issue_status', 'issue_number', 'file', 'time']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['last', 'recent', 'late']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'branch', 'master']):
            return LabelingFunctions._LATEST_COMMIT
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has no entity.
    # Condition 2: 'contribution' or 'contribute' or 'contributor' or 'commit' in the clean text of command.
    # Condition 3: 'issue', 'branch' NOT in the clean text of command.
    # Condition 4: (WH question and starts with 'how' or 'what') or (polar question)
    # Condition 5: has_proper_singular_noun == True OR ('by' in the text).
    # Intent: CONTRIBUTIONS_BY_DEVELOPER
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def contributions_by_developer_1(command):
        condition_wh = (LabelingFunctions.is_wh_question(command)
                        and
                        (LabelingFunctions.text_starts_with('how', command)
                         or
                         LabelingFunctions.text_starts_with('what', command)))
        condition_polar = LabelingFunctions.is_polar_question(command)
        condition_4 = condition_wh or condition_polar
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['contribution', 'contribute', 'contributor',
                                                                'commit']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'branch']) \
                and \
                condition_4 \
                and \
                (LabelingFunctions.has_proper_singular_noun(command) or LabelingFunctions.found_in_text('by', command)):
            return LabelingFunctions._CONTRIBUTIONS_BY_DEVELOPER
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has at least one time entity: has(['time') and no other entities except 'time' or 'number'.
    # Condition 2: 'report' or 'happen' or 'event' pr 'recent' or 'activity' in the clean text of command.
    # Condition 3: Does not have 'issue', 'bug' keywords in the clean text.
    # Intent: ACTIVITY_REPORT
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def activity_report_1(command):
        if LabelingFunctions.has(command, ['time']) and LabelingFunctions.has_only(command, ['time', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['report', 'happen', 'event', 'recent', 'activity']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'bug']):
            return LabelingFunctions._ACTIVITY_REPORT
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: have no entities at all.
    # Condition 2: 'report' or 'event' in the clean text of command.
    # Condition 3: Does not have 'issue', 'bug' keywords in the clean text.
    # Intent: ACTIVITY_REPORT
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def activity_report_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['report', 'event']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'bug']):
            return LabelingFunctions._ACTIVITY_REPORT
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_and_has_only(['issue_status') means: has at least one issue_status entity and no other entities
    #   except issue_status.
    # Condition 2: 'number' or 'report' or 'count' in the clean text of command OR has 'how many' in the text.
    # Condition 3: does not have 'issue' in the clean text.
    # Intent: NUMBER_OF_PRS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_prs_1(command):
        if LabelingFunctions.has_and_has_only(command, ['issue_status']) \
                and \
                (LabelingFunctions.check_in_clean_text(command, ['number', 'report', 'count']) or
                 LabelingFunctions.found_in_text('how many', command)) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._NUMBER_OF_PRS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: no entities at all.
    # Condition 2: 'number' or 'report' or 'count' in the clean text of command or 'how many' exists in text.
    # Condition 3: 'prs' or 'pull' in the clean text of command.
    # Condition 4: does not have 'issue' in the clean text.
    # Intent: NUMBER_OF_PRS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_prs_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                (LabelingFunctions.check_in_clean_text(command, ['number', 'report', 'count']) or
                 LabelingFunctions.found_in_text('how many', command)) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['prs', 'pull']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._NUMBER_OF_PRS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has at least one issue_number entity, and no other entities except issue_number, number, or filename.
    # Condition 2: 'assign' or 'assignee' or 'developer' or 'fix' exists in clean text of command.
    # Condition 3: 'issue', 'merge', 'close', 'start', 'create', 'contribute', 'contributor', 'comment', 'commenter',
    #   'help' NOT in the clean text of command.
    # Intent: PR_ASSIGNEES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_assignees_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['assign', 'assignee', 'developer', 'fix']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'merge', 'close', 'start', 'create',
                                                                    'contribute', 'contributor', 'comment', 'commenter',
                                                                    'help']):
            return LabelingFunctions._PR_ASSIGNEES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has at least one issue_number entity, and no other entities except issue_number, number, or filename.
    # Condition 2: 'creation' or 'create' or 'open' or 'opened' exists in clean text of command.
    # Condition 3: 'issue' not in clean text of command.
    # Condition 4: Is wh question and 'when' exists in command.
    # Intent: PR_CREATION_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_creation_date_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['creation', 'create', 'open', 'opened']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']) \
                and \
                LabelingFunctions.is_wh_question(command) and LabelingFunctions.found_in_text('when', command):
            return LabelingFunctions._PR_CREATION_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has at least one issue_number entity, and no other entities except issue_number, number, or filename.
    # Condition 2: 'creation' or 'create' or 'open' or 'opened' exists in clean text of command.
    # Condition 3: 'date' or 'time' exists in clean text of command.
    # Condition 4: 'issue' not in clean text of command.
    # Intent: PR_CREATION_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_creation_date_2(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['creation', 'create', 'open', 'opened']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['date', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._PR_CREATION_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has at least one issue_number entity, and no other entities except issue_number, number, or filename.
    # Condition 2: 'close' or 'merge' or 'closing' or 'resolve' exists in clean text of command.
    # Condition 3: 'date' or 'time' exists in clean text of command.
    # Condition 4: 'issue' NOT in clean text of command.
    # Intent: PR_CLOSING_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_closing_date_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'closing', 'resolve']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['date', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._PR_CLOSING_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has at least one issue_number entity, and no other entities except issue_number, number, or filename.
    # Condition 2: 'close' or 'merge' or 'closing' or 'resolve' exists in clean text of command.
    # Condition 3: Is wh question and 'when' exists in command.
    # Condition 4: 'issue' NOT in clean text of command.
    # Intent: PR_CLOSING_DATE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_closing_date_2(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number', 'filename']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'closing', 'resolve']) \
                and \
                LabelingFunctions.is_wh_question(command) and LabelingFunctions.found_in_text('when', command) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']):
            return LabelingFunctions._PR_CLOSING_DATE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Is wh question and 'who' exists in text.
    # Condition 2: 'active' exists in the clean text of command.
    # Condition 3: 'developer' exists in the clean text of command.
    # Intent: TOP_CONTRIBUTORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def top_contributors_1(command):
        if LabelingFunctions.is_wh_question(command) and LabelingFunctions.found_in_text('who', command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['active']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['developer']):
            return LabelingFunctions._TOP_CONTRIBUTORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Can have only number and time entities.
    # Condition 2: Has the word 'top', or 'most' in the clean text.
    # Condition 3: Has the word 'contribute', 'contributor', or 'contributer' in the clean text.
    # Intent: TOP_CONTRIBUTORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def top_contributors_2(command):
        if LabelingFunctions.has_only(command, ['number', 'time']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['top', 'most', 'high']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['contribute', 'contributor', 'contributer',
                                                                'contribution', 'who']):
            return LabelingFunctions._TOP_CONTRIBUTORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Is wh question and 'when' exists in text.
    # Condition 2: 'recent' in clean text of command.
    # Condition 3: ('pr' or 'pull') exist in clean text of command.
    # Intent: MOST_RECENT_PRS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def most_recent_prs_1(command):
        if LabelingFunctions.is_wh_question(command) and LabelingFunctions.found_in_text('when', command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['recent']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['pr', 'pull']):
            return LabelingFunctions._MOST_RECENT_PRS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: does_not_have 'number' and 'issue_number' entities.
    # Condition 2: 'pr', 'prs', 'pull', or 'pull-request' in the clean text of command.
    # Condition 3: 'issue' not in the clean text of command.
    # Condition 4: 'new' or 'late' or 'recent' in the clean text of command.
    # Intent: MOST_RECENT_PRS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def most_recent_prs_2(command):
        if LabelingFunctions.does_not_have(command, ['number', 'issue_number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['pr', 'pull', 'pull-request', 'prs']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['new', 'late', 'recent']):
            return LabelingFunctions._MOST_RECENT_PRS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Is wh question and 'how many' exists in text.
    # Condition 2: 'commit' exists in clean text of command.
    # Condition 3: 'branch' exists in clean text of command.
    # Intent: NUMBER_OF_COMMITS_IN_BRANCH
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_commits_in_branch_1(command):
        if LabelingFunctions.is_wh_question(command) and LabelingFunctions.found_in_text('how many', command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']):
            return LabelingFunctions._NUMBER_OF_COMMITS_IN_BRANCH
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Can only have number entity (has_only).
    # Condition 2: 'commit' exists in clean text of command and not a verb.
    # Condition 3: 'commit' is NOT a verb.
    # Condition 4: 'branch' exists in clean text of command.
    # Condition 5: 'last', 1st', 'first', 'initial', 'intial', 'inital' NOT in the clean text of command.
    # Intent: NUMBER_OF_COMMITS_IN_BRANCH
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_commits_in_branch_2(command):
        if LabelingFunctions.has_only(command, ['number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit']) \
                and \
                LabelingFunctions.not_a_verb(command, 'commit') \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['last', 'late', '1st', 'first', 'initial', 'intial',
                                                                    'inital']):
            return LabelingFunctions._NUMBER_OF_COMMITS_IN_BRANCH
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has 'issue_number' entity.
    # Condition 2: Has 'pr', or 'pull-request', or 'pull' in the clean text.
    # Condition 3: Has 'close' or 'merge' or 'merger' in the clean text.
    # Condition 4: Does NOT have 'issue', 'when', 'date', 'time' in the clean text.
    # Intent: PR_CLOSER
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def pr_closer_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['pr', 'pull-request', 'pull']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['close', 'merge', 'merger']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'when', 'date', 'time']):
            return LabelingFunctions._PR_CLOSER
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only 'issue_status' entity.
    # Condition 2: Has the word 'issue' or 'bug' in the clean text.
    # Condition 3: Has the word 'number', 'count', 'report' in the clean text.
    # Condition 4: Does not have the words 'pr', 'prs', 'pull', 'pull-request', 'commit', 'assign' in the clean text.
    # Intent: NUMBER_OF_ISSUES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_issues_1(command):
        if LabelingFunctions.has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['issue', 'bug']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['number', 'count', 'report']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'prs', 'pull', 'pull-request', 'commit',
                                                                    'assign']):
            return LabelingFunctions._NUMBER_OF_ISSUES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only 'issue_status' entity.
    # Condition 2: Does not have the words 'pr', 'prs', 'pull', 'pull-request', 'star', 'fork', 'watch', 'branch',
    #   'watcher', 'download', 'subscriber', 'commit', 'collaborator', 'collaborater', 'developer', 'contributor',
    #   'contributer' in the clean text.
    # Condition 3: 'how many' exists in text.
    # Intent: NUMBER_OF_ISSUES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_issues_2(command):
        if LabelingFunctions.has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'prs', 'pull-request', 'star', 'fork',
                                                                    'watch', 'branch', 'watcher', 'download',
                                                                    'subscriber', 'commit', 'collaborator',
                                                                    'collaborater', 'developer', 'contributor',
                                                                    'contributer', 'contribution']) \
                and \
                LabelingFunctions.found_in_text('how many', command):
            return LabelingFunctions._NUMBER_OF_ISSUES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity.
    # Condition 2: Has the word 'fork' in the clean text.
    # Condition 3: Has the word 'number', 'count' in the clean text.
    # Intent: NUMBER_OF_FORKS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_forks_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['fork']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['number', 'count']):
            return LabelingFunctions._NUMBER_OF_FORKS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity.
    # Condition 2: Has the word 'fork' in the clean text.
    # Condition 3: 'how many' is in the text.
    # Intent: NUMBER_OF_FORKS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_forks_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['fork']) \
                and \
                LabelingFunctions.found_in_text('how many', command):
            return LabelingFunctions._NUMBER_OF_FORKS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'commit' or 'changes' in the clean text.
    # Condition 3: Has the word 'number', 'count' in the clean text.
    # Condition 4: Not has_proper_singular_noun: Does not have proper_singular_noun.
    # Condition 5: Does not have 'branch' in the clean text (Preventing it from detecting number_of_commits_in_branch).
    # Intent: NUMBER_OF_COMMITS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_commits_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit', 'change']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['number', 'count']) \
                and \
                not LabelingFunctions.has_proper_singular_noun(command) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['branch', 'high']):
            return LabelingFunctions._NUMBER_OF_COMMITS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'commit' or 'changes' in the clean text.
    # Condition 3: 'how many' is in the text.
    # Condition 4: Not has_proper_singular_noun: Does not have proper_singular_noun.
    # Condition 5: Does not have 'branch' in the clean text (Preventing it from detecting number_of_commits_in_branch).
    #   Also does not have 'developer' in the clean text (Preventing it from detecting NUMBER_OF_COLLABORATORS).
    # Intent: NUMBER_OF_COMMITS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_commits_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit', 'change']) \
                and \
                LabelingFunctions.found_in_text('how many', command) \
                and \
                not LabelingFunctions.has_proper_singular_noun(command) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['branch', 'developer']):
            return LabelingFunctions._NUMBER_OF_COMMITS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity.
    # Condition 2: Has the word 'branch' in the clean text.
    # Condition 3: Has the word 'number', 'count', 'available' in the clean text.
    # Condition 4: Does not have word 'issue', 'pr', 'pull' or 'pull-request', 'commit' in the clean text.
    # Intent: NUMBER_OF_BRANCHES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_branches_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['number', 'count', 'available']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'pull-request', 'commit']):
            return LabelingFunctions._NUMBER_OF_BRANCHES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity.
    # Condition 2: Has the word 'branch' in the clean text.
    # Condition 3: 'how many' is in the text.
    # Condition 4: Does not have word 'issue', 'pr', 'pull' or 'pull-request', 'commit' in the clean text.
    # Intent: NUMBER_OF_BRANCHES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def number_of_branches_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']) \
                and \
                LabelingFunctions.found_in_text('how many', command) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'pull-request', 'commit']):
            return LabelingFunctions._NUMBER_OF_BRANCHES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'release' in the clean text.
    # Condition 3: Does not have 'issue', 'pr', 'pull', 'star', 'fork', 'watch', 'branch' in the clean text. Also,
    #   'late', 'new', 'last' to prevent this LF detect the LATEST_RELEASE related commands.
    # Intent: LIST_RELEASES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def list_releases_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['release']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'star', 'fork', 'watch',
                                                                    'branch', 'last', 'late', 'new']):
            return LabelingFunctions._LIST_RELEASES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'collaborator', 'contributor' in the clean text
    # Condition 3: Does not have 'issue', 'pr', 'pull', 'star', 'fork', 'watch', 'branch', 'release', 'commit', 'most',
    #   'high', 'top', 'number' in the clean text.
    # Condition 4: 'how many' NOT in the text.
    # Intent: LIST_COLLABORATORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def list_collaborators_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['collaborator', 'contributor', 'worker']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'star', 'fork', 'watch',
                                                                    'branch', 'release', 'commit', 'most', 'high',
                                                                    'top', 'number']) \
                and \
                not LabelingFunctions.found_in_text('how many', command):
            return LabelingFunctions._LIST_COLLABORATORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'people', or 'developer', or 'who' in the clean text.
    # Condition 3: Has the word 'work', or 'contribute', or 'collaborate', or 'commit', or 'add' in the clean text.
    # Condition 4: Does not have 'issue', 'pr', 'pull', 'star', 'fork', 'watch', 'branch', 'release', 'most', 'high',
    #   'top', '1st', 'first', 'initial', 'intial', 'inital' in the clean text.
    # Condition 5: 'how many' NOT in the text.
    # Intent: LIST_COLLABORATORS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def list_collaborators_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['people', 'developer', 'who']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['work', 'contribute', 'collaborate', 'commit', 'add',
                                                                'help']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'star', 'fork', 'watch',
                                                                    'branch', 'release', 'most', 'high', 'top',
                                                                    'last', '1st', 'first', 'initial', 'intial',
                                                                    'inital']) \
                and \
                not LabelingFunctions.found_in_text('how many', command):
            return LabelingFunctions._LIST_COLLABORATORS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Can only have (has_only) 'time' entity.
    # Condition 2: Has the word 'branch' in the clean text.
    # Condition 3: 'how many' does NOT exist in the text.
    # Condition 4: Does not have 'issue', 'pr', 'pull', 'star', 'fork', 'watch', 'release', 'count', 'number', 'commit'
    #   in the clean text. Also, does not have 'main', 'base', 'default' keywords to separate it from DEFAULT_BRANCH
    #   intent.
    # Intent: LIST_BRANCHES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def list_branches_1(command):
        if LabelingFunctions.has_only(command, ['time']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['branch']) \
                and \
                not LabelingFunctions.found_in_text('how many', command) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['issue', 'pr', 'pull', 'star', 'fork', 'watch',
                                                                    'release', 'count', 'number', 'commit', 'default',
                                                                    'main', 'base']):
            return LabelingFunctions._LIST_BRANCHES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has ('issue_number' or 'number) entity and has_only 'issue number' and 'number' entities.
    # Condition 2: Has the word 'commit' or 'change' in the clean text.
    # Condition 3: Does not have 'star', 'fork', 'watch', 'release', 'count', 'number', 'file', 'pr', 'pull',
    #   'pull-request', 'branch' in the clean text.
    # Intent: ISSUE_RELATED_COMMITS
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_related_commits_1(command):
        if (LabelingFunctions.has(command, ['issue_number']) or LabelingFunctions.has(command, ['number'])) \
                and \
                LabelingFunctions.has_only(command, ['issue_number', 'number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit', 'change']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['star', 'fork', 'watch', 'release', 'count',
                                                                    'number', 'file', 'pr', 'pull', 'pull-request',
                                                                    'branch']):
            return LabelingFunctions._ISSUE_RELATED_COMMITS
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has 'file' entity.
    # Condition 2: Has the word 'create' or 'creator', or 'add', or 'start' in the clean text.
    # Condition 3: Does not have 'star', 'fork', 'watch', 'release', 'count', 'number' in the clean text.
    # Intent: FILE_CREATOR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def file_creator_1(command):
        if LabelingFunctions.has(command, ['file']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['create', 'creator', 'add', 'start']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['star', 'fork', 'watch', 'release', 'count',
                                                                    'number']):
            return LabelingFunctions._FILE_CREATOR
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity.
    # Condition 2: Has the word 'create' or 'creator', or 'add', or 'start' in the clean text.
    # Condition 3: Has the word 'file' in the clean text.
    # Condition 4: Does not have 'star', 'fork', 'watch', 'release', 'count', 'number' in the clean text.
    # Intent: FILE_CREATOR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def file_creator_2(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['create', 'creator', 'add', 'start']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['file']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['star', 'fork', 'watch', 'release', 'count',
                                                                    'number']):
            return LabelingFunctions._FILE_CREATOR
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has no entity
    # Condition 2: Has the word 'large', or 'long', or 'big', or 'huge', or 'line', or 'loc' in the clean text.
    # Condition 3: Has the word 'file' in the clean text.
    # Intent: LARGEST_FILES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def largest_files_1(command):
        if LabelingFunctions.has_no_entity(command) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['large', 'long', 'big', 'huge', 'line', 'loc']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['file']):
            return LabelingFunctions._LARGEST_FILES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only issue_status entity.
    # Condition 2: Has the word 'long' in the clean text.
    # Condition 3: Has the word 'pr' or 'pull' or 'pull-request' in the clean text.
    # Intent: LONGEST_OPEN_PR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def longest_open_pr_1(command):
        if LabelingFunctions.has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['long']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['pr', 'pull', 'pull-request']):
            return LabelingFunctions._LONGEST_OPEN_PR
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: has_only issue_status entity.
    # Condition 2: Has the word 'long' or 'old' or 'time' in the clean text.
    # Condition 3: Does not have the words 'pr', 'pull', and 'pull-request' in the clean text.
    # Condition 4: Has the word 'issue' in the clean text.
    # Intent: LONGEST_OPEN_ISSUE
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def longest_open_issue_1(command):
        if LabelingFunctions.has_only(command, ['issue_status']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['long', 'old', 'time']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'pull-request']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['issue']):
            return LabelingFunctions._LONGEST_OPEN_ISSUE
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has 'issue_number' entity.
    # Condition 2: Has the word 'work', or 'assignee', or 'assigne', or 'fix' in the clean text.
    # Condition 3: Does not have the word 'pr', 'pull', 'pull-request' in the clean text.
    # Intent: ISSUE_ASSIGNEES
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def issue_assignees_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['work', 'assignee', 'assigne', 'assign', 'fix']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['pr', 'pull', 'pull-request']):
            return LabelingFunctions._ISSUE_ASSIGNEES
        else:
            return LabelingFunctions._ABSTAIN

    # Condition 1: Has 'issue_number' entity.
    # Condition 2: Has the word 'commit' or 'change' in the clean text.
    # Condition 3: Has the word 'pr' or 'pull' or 'pull-request' in the clean text.
    # Condition 4: Does not have 'file' in the clean text. (Preventing it to detect FILES_CHANGED_BY_PR).
    # Intent: COMMITS_IN_PR
    @staticmethod
    @labeling_function(pre=[preprocess_command.__func__])
    def commits_in_pr_1(command):
        if LabelingFunctions.has(command, ['issue_number']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['commit', 'change']) \
                and \
                LabelingFunctions.check_in_clean_text(command, ['pr', 'pull', 'pull-request']) \
                and \
                LabelingFunctions.check_not_in_clean_text(command, ['file']):
            return LabelingFunctions._COMMITS_IN_PR
        else:
            return LabelingFunctions._ABSTAIN


if __name__ == "__main__":
    labeling_function = LabelingFunctions()
    functions = {
        labeling_function.most_recent_prs_1: True,
        labeling_function.top_contributors_1: True,
        labeling_function.number_of_commits_in_branch_1: True,
        labeling_function.pr_closing_date_1: True,
        labeling_function.pr_closing_date_2: True,
        labeling_function.pr_creation_date_1: True,
        labeling_function.pr_creation_date_2: True,
        labeling_function.pr_assignees_1: True,
        labeling_function.number_of_prs_1: True,
        labeling_function.number_of_prs_2: True,
        labeling_function.activity_report_1: True,
        labeling_function.activity_report_2: True,
        labeling_function.contributions_by_developer_1: True,
        labeling_function.latest_commit_1: True,
        labeling_function.default_branch_1: True,
        labeling_function.developers_with_most_open_issues_1: True,
        labeling_function.developers_with_most_open_issues_2: True,
        labeling_function.files_changed_by_pr_1: True,
        labeling_function.intial_commit_in_branch_1: True,
        labeling_function.issue_closer_1: True,
        labeling_function.issue_closing_date_1: True,
        labeling_function.issue_closing_date_2: True,
        labeling_function.issue_contributors_1: True,
        labeling_function.issue_creation_date_1: True,
        labeling_function.issue_creation_date_2: True,
        labeling_function.issue_creator_1: True,
        labeling_function.last_developer_to_touch_a_file_1: True,
        labeling_function.latest_commit_in_branch_1: True,
        labeling_function.latest_release_1: True,
        labeling_function.list_languages_1: True,
        labeling_function.number_of_collaborators_2: True,
        labeling_function.number_of_collaborators_1: True,
        labeling_function.number_of_downloads_1: True,
        labeling_function.number_of_stars_1: True,
        labeling_function.number_of_subscribers_1: True,
        labeling_function.number_of_watchers_1: True,
        labeling_function.pr_contributors_1: True,
        labeling_function.pr_creator_1: True,
        labeling_function.repository_creation_date_1: True,
        labeling_function.repository_license_1: True,
        labeling_function.repository_owner_1: True,
        labeling_function.repository_owner_2: True,
        labeling_function.repository_topics_1: True,
        labeling_function.commits_in_pr_1: True,
        labeling_function.issue_assignees_1: True,
        labeling_function.longest_open_pr_1: True,
        labeling_function.largest_files_1: True,
        labeling_function.file_creator_1: True,
        labeling_function.issue_related_commits_1: True,
        labeling_function.list_branches_1: True,
        labeling_function.list_collaborators_2: True,
        labeling_function.list_collaborators_1: True,
        labeling_function.list_releases_1: True,
        labeling_function.number_of_branches_1: True,
        labeling_function.number_of_branches_2: True,
        labeling_function.number_of_commits_1: True,
        labeling_function.number_of_commits_2: True,
        labeling_function.number_of_forks_1: True,
        labeling_function.number_of_forks_2: True,
        labeling_function.number_of_issues_1: True,
        labeling_function.number_of_issues_2: True,
        labeling_function.pr_closer_1: True,
        labeling_function.most_recent_issues_1: True,
        labeling_function.longest_open_issue_1: True,
        labeling_function.most_recent_prs_2: True,
        labeling_function.intial_commit_1: True,
        labeling_function.top_contributors_2: True,
        labeling_function.main_programming_language_1: True,
        labeling_function.file_creator_2: True,
        labeling_function.number_of_commits_in_branch_2: True
    }
    print('labeling_functions')
