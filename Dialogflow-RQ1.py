import datetime
import inquirer
import os
import glob
import pytz
from classes.data_tools import DataTools
from classes.engineer import Engineer
from classes.labeling_functions import LabelingFunctions
import json
from inquirer.themes import GreenPassion
from google.cloud import dialogflow
import asyncio
import yaml
import re
import time
import random
from google.api_core.exceptions import InvalidArgument
import pandas as pd
from csv import reader


def calculate_f1(path):
    with open(path, 'r') as csv_file:
        csv_reader = reader(csv_file)
        first_list_of_rows = list(csv_reader)
    dict_of_intents = {}
    for row in first_list_of_rows:
        if row[1] not in dict_of_intents.keys():
            dict_of_intents[row[1]] = {'TP': 0, 'TF': 0, 'FP': 0, 'FN': 0, 'support': 0, 'f1': 0.0}
        if row[2] not in dict_of_intents.keys():
            dict_of_intents[row[2]] = {'TP': 0, 'TF': 0, 'FP': 0, 'FN': 0, 'support': 0, 'f1': 0.0}
        dict_of_intents[row[1]]['support'] += 1
        if row[1] == row[2]:
            dict_of_intents[row[1]]['TP'] += 1
        else:
            dict_of_intents[row[1]]['FN'] += 1
            dict_of_intents[row[2]]['FP'] += 1
    total = 0
    for key, values in dict_of_intents.items():
        dict_of_intents[key]['f1'] = values['TP']/(values['TP'] + ((values['FP'] + values['FN'])/2))
        total += values['support']
    sum_all = 0.0
    for key, value in dict_of_intents.items():
        sum_all += value['f1'] * value['support']
    return (sum_all/total) * 100


async def train_agent():
    await dialogflow.TrainAgentRequest()


def get_intent_ids(project_id, display_name):

    intents_client = dialogflow.IntentsClient()

    parent = dialogflow.AgentsClient.agent_path(project_id)
    intents = intents_client.list_intents(request={"parent": parent})
    intent_names = [
        intent.name for intent in intents if intent.display_name == display_name
    ]

    intent_ids = [intent_name.split("/")[-1] for intent_name in intent_names]

    return intent_ids


def delete_intent(project_id, intent_id):
    """Delete intent with the given intent type and intent value."""

    intents_client = dialogflow.IntentsClient()

    intent_path = intents_client.intent_path(project_id, intent_id)

    intents_client.delete_intent(request={"name": intent_path})


def create_intent(project_id, display_name, training_phrases_parts):
    """Create an intent of the given intent type."""

    intents_client = dialogflow.IntentsClient()

    parent = dialogflow.AgentsClient.agent_path(project_id)
    training_phrases = []
    for training_phrases_part in training_phrases_parts:
        part = dialogflow.Intent.TrainingPhrase.Part(text=training_phrases_part)
        # Here we create a new training phrase for each provided part.
        training_phrase = dialogflow.Intent.TrainingPhrase(parts=[part])
        training_phrases.append(training_phrase)

    intent = dialogflow.Intent(display_name=display_name, training_phrases=training_phrases)

    response = intents_client.create_intent(request={"parent": parent, "intent": intent})

    print("Intent created: {}".format(response))


def prepare_data(training_data_path):
    with open(training_data_path, 'r') as f:
        doc = yaml.safe_load(f)

    # Extract the training examples from nlu.yml and store them in the training_data dictionary
    # The key is the intent name, and the value is a list of examples for that intent
    training_data = {}
    for record in doc["nlu"]:
        if 'intent' in record.keys():
            # remove the tagged entity that is used by Rasa (e.g., [filename])
            examples = re.sub(r'\([^)]*\)', '', record['examples'])

            # Text cleaning, remove [, ], and new line. Then split based on -
            examples = examples.replace('\n', '').replace('[', '').replace(']', '').split('- ')
            examples = list(filter(None, examples))

            training_data[record['intent']] = examples
    return training_data


def prepare_data_test(testing_data_path):
    with open(testing_data_path, 'r') as f:
        doc = yaml.safe_load(f)

    # Extract the training examples from nlu.yml and store them in the training_data dictionary
    # The key is the intent name, and the value is a list of examples for that intent
    testing_data = {}
    for record in doc["nlu"]:
        if 'intent' in record.keys():
            # remove the tagged entity that is used by Rasa (e.g., [filename])
            examples = re.sub(r'\([^)]*\)', '', record['examples'])

            # Text cleaning, remove [, ], and new line. Then split based on -
            examples = examples.replace('\n', '').replace('[', '').replace(']', '').split('- ')
            examples = list(filter(None, examples))
            for example in examples:
                testing_data[example] = [record['intent']]
    return testing_data


def continue_to_run_question(question):
    qu = [
        inquirer.Confirm('continue',
                         message=question, default=True),
    ]

    ans = inquirer.prompt(qu, theme=GreenPassion())
    if not ans['continue']:
        print('bye!')
        exit(0)
    else:
        return True


if __name__ == '__main__':

    display_name = "commit_creator"
    parent = dialogflow.AgentsClient.agent_path("chatbot-ws-arqe")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'keys/private_arqe.json'
    DIALOGFLOW_PROJECT_ID = 'chatbot-ws-arqe'
    SESSION_ID = 'me'
    DIALOGFLOW_LANGUAGE_CODE = 'en'
    intents_client = dialogflow.IntentsClient()

    current_datetime = datetime.datetime.now(pytz.timezone('America/Montreal'))
    timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
    #  Getting the fraction for train-test and the output directory: ##############################
    total = 0
    while abs(total - 1.00) > 1e-5:
        questions = [
            inquirer.Text('training-fraction', message='Please enter the training-fraction for Google Dialogflow',
                          default=0.4),
            inquirer.Text('testing-fraction', message='Please enter the testing-fraction', default=0.3),
            inquirer.Text('validation-fraction', message='Please enter the validation-fraction for Weak Supervision',
                          default=0.3),
            inquirer.Text('default-directory', message='Please enter the default directory for the outputs',
                          default=timestamp),
        ]
        answers = inquirer.prompt(questions, theme=GreenPassion())
        training_fraction = float(answers['training-fraction'])
        testing_fraction = float(answers['testing-fraction'])
        validation_fraction = float(answers['validation-fraction'])
        total = training_fraction + testing_fraction + validation_fraction
        if abs(total - 1.00) > 1e-5:
            print('training_fraction + testing_fraction + validation_fraction is not equal to 1 (or 100 percent)!')
            print('Please try again:')

    #  Creating the project directory: ##############################
    project_directory = "/vagrant/Output/" + str(answers['default-directory']) + '/'
    command = "mkdir " + project_directory
    os.system(command)

    #  Choosing the proper dataset to use: ##############################
    questions = [
        inquirer.List('dataset',
                      message="Which dataset do you want to use?",
                      choices=[
                          ('Dataset WITH general intents', 'w'),
                          ('Dataset WITHOUT general intents', 'wo')
                      ],
                      carousel=True,
                      default='wo'
                      ),
    ]
    # answers = inquirer.prompt(questions, theme=GreenPassion())
    command = "cp -r /vagrant/Dataset/Paper/ " + project_directory + 'data'
    # if answers['dataset'] == 'wo':
    #     command = "cp -r /vagrant/Dataset/Paper/ " + project_directory + 'data'
    # else:
    #     command = "cp -r /vagrant/Dataset/Original/ " + project_directory + 'data'
    os.system(command)

    questions = [
        inquirer.List('ask-question-in-each-step',
                      message="Should I ask the question for continue each single step?",
                      choices=[
                          ('No just finish the evaluation!', 'N'),
                          ('Yes please. I want to track every step you are doing!', 'Y')
                      ],
                      carousel=True,
                      default='N'
                      ),
    ]
    answers = inquirer.prompt(questions, theme=GreenPassion())
    ask_question = False
    if answers['ask-question-in-each-step'] == 'Y':
        ask_question = True

    testing_fraction = testing_fraction / (1 - training_fraction)
    validation_fraction = 1 - testing_fraction
    command = "bash Bash/Rasa/rasa_data_split_with_validation.sh " + \
              " -f " + str(training_fraction) + \
              " -t " + str(testing_fraction) + \
              " -p " + project_directory[:len(project_directory)-1]
    print('Splitting the Google NLU data to training, testing, and validation sets:')
    os.system(command)

    command = "bash Bash/Rasa/make_data_for_train.sh " + \
              "-t " + timestamp
    print('Making data ready for training the Google NLU:')
    os.system(command)

    if ask_question:
        continue_to_run_question("Test-Training-Validation split is finished. Should I continue?")
    # Training the first Google Model: ##############################

    print('Training the Google NLU model:')
    trainingData = prepare_data(project_directory + 'train/nlu.yml')
    print(trainingData)
    # Delete the intent
    i = 0
    for key in trainingData.keys():
        i += 1
        intent_id = get_intent_ids(DIALOGFLOW_PROJECT_ID, key)
        print(intent_id)
        if intent_id:
            print('\n' + 'deleting ' + str(i) + ' : ' + str(intent_id) + '\n')
            delete_intent(DIALOGFLOW_PROJECT_ID, intent_id[0])
            time.sleep(random.randint(1, 3))
    # Create the intent
    for key in trainingData.keys():
        create_intent(DIALOGFLOW_PROJECT_ID,
                      key,
                      trainingData[key])
        time.sleep(random.randint(1, 3))
    # Train the Dialogflow
    asyncio.wait(train_agent)
    print('\n\nTraining Dialogflow...Please Wait...\n\n')
    time.sleep(600)
    if ask_question:
        continue_to_run_question("Training Google model is finished. Should I continue?")
    # Testing the first Google model ##############################

    print('Testing the Google model:')
    print('Testing the test file on the trained model:')
    testing_path = project_directory + 'testing_validation/test_data.yml'
    testingData = prepare_data_test(testing_path)
    command = "mkdir " + project_directory + 'first_test_results'
    os.system(command)
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    for example, intent in testingData.items():
        text_input = dialogflow.TextInput(text=example, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)
        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise

        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        intent.append(response.query_result.intent.display_name)
        testingData[example] = intent
        time.sleep(random.randint(1, 2))
    pd.DataFrame.from_dict(data=testingData, orient='index').to_csv(project_directory + 'first_test_results/' +
                                                                    'results.csv', header=False)

    if ask_question:
        continue_to_run_question("Testing the first Google model is finished. Should I continue?")

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
    command = "mkdir " + project_directory + 'engineer'
    os.system(command)
    dataTools = DataTools(project_directory + 'engineer/', fixed_directory=True)
    list_of_commands = dataTools.yml_to_input_commands(project_directory + 'testing_validation/training_data.yml')
    pandas_dataframe = dataTools.list_of_commands_to_pandas_dataframe(list_of_commands)
    dataTools.pandas_dataframe_to_csv(pandas_dataframe)
    engineer = Engineer(pandas_dataframe, functions, labeling_function.CARDINALITY, dataTools)
    engineer.produce_labeling_matrix()
    engineer.print_labeling_matrix()
    engineer.print_labeling_functions_summary()
    engineer.predict_by_majority_vote_model()
    dict_of_intents = {0: 'MOST_RECENT_PRS', 1: 'NUMBER_OF_COMMITS_IN_BRANCH', 2: 'TOP_CONTRIBUTORS',
                       3: 'INTIAL_COMMIT', 4: 'MOST_RECENT_ISSUES', 5: 'LONGEST_OPEN_ISSUE',
                       6: 'LATEST_COMMIT', 7: 'CONTRIBUTIONS_BY_DEVELOPER', 8: 'ACTIVITY_REPORT',
                       9: 'NUMBER_OF_PRS', 10: 'PR_ASSIGNEES', 11: 'PR_CREATION_DATE',
                       12: 'PR_CLOSING_DATE', 13: 'PR_CLOSER', 14: 'NUMBER_OF_ISSUES', 15: 'NUMBER_OF_FORKS',
                       16: 'NUMBER_OF_COMMITS', 17: 'NUMBER_OF_BRANCHES', 18: 'LIST_RELEASES', 19: 'LIST_COLLABORATORS',
                       20: 'LIST_BRANCHES', 21: 'ISSUE_RELATED_COMMITS', 22: 'FILE_CREATOR', 23: 'LARGEST_FILES',
                       24: 'LONGEST_OPEN_PR', 25: 'ISSUE_ASSIGNEES', 26: 'COMMITS_IN_PR',
                       27: 'REPOSITORY_TOPICS', 28: 'REPOSITORY_OWNER', 29: 'REPOSITORY_LICENSE',
                       30: 'REPOSITORY_CREATION_DATE', 31: 'PR_CREATOR', 32: 'PR_CONTRIBUTORS',
                       33: 'NUMBER_OF_WATCHERS', 34: 'NUMBER_OF_SUBSCRIBERS', 35: 'NUMBER_OF_STARS',
                       36: 'NUMBER_OF_DOWNLOADS', 37: 'NUMBER_OF_COLLABORATORS', 38: 'LIST_LANGUAGES',
                       39: 'LATEST_RELEASE', 40: 'LATEST_COMMIT_IN_BRANCH', 41: 'LAST_DEVELOPER_TO_TOUCH_A_FILE',
                       42: 'ISSUE_CREATOR', 43: 'ISSUE_CREATION_DATE', 44: 'ISSUE_CONTRIBUTORS',
                       45: 'ISSUE_CLOSING_DATE', 46: 'ISSUE_CLOSER', 47: 'INTIAL_COMMIT_IN_BRANCH',
                       48: 'FILES_CHANGED_BY_PR', 49: 'DEVELOPERS_WITH_MOST_OPEN_ISSUES', 50: 'DEFAULT_BRANCH',
                       51: 'MAIN_PROGRAMMING_LANGUAGE'}
    dataTools.save_predicted_commands_with_intents(dict_of_intents, model='majority')
    nlu_file_path = project_directory + 'train/nlu.yml'
    dataTools.add_predictions_to_nlu(nlu_file_path, dict_of_intents, model='majority')

    if ask_question:
        continue_to_run_question("Engineer finished his job. Should I continue?")
    # Making data for the second Google model from the predictions of the Label model: ##############################

    command = "bash Bash/Rasa/make_data_for_train_from_new_nlu.sh " + \
              " -t " + timestamp + \
              " -f " + 'new_nlu.yml'
    print('Creating the new data directory for re-training:')
    os.system(command)

    if ask_question:
        continue_to_run_question("The re-train directory (with its files) is created. Should I continue?")

    # Training the second Google model: ##############################
    print('deleting previous trained model:')
    trainingData = prepare_data(project_directory + 're-train/nlu.yml')
    print(trainingData)
    # Delete the intent
    i = 0
    for key in trainingData.keys():
        i += 1
        intent_id = get_intent_ids(DIALOGFLOW_PROJECT_ID, key)
        print(intent_id)
        if intent_id:
            print('\n' + 'deleting ' + str(i) + ' : ' + str(intent_id) + '\n')
            delete_intent(DIALOGFLOW_PROJECT_ID, intent_id[0])
            time.sleep(random.randint(1, 3))
    print('re-training Google NLU again from scratch:')
    # Create the intent
    for key in trainingData.keys():
        create_intent(DIALOGFLOW_PROJECT_ID,
                      key,
                      trainingData[key])
        time.sleep(random.randint(1, 3))
    # Train the Dialogflow
    asyncio.wait(train_agent)
    print('\n\nTraining Dialogflow...Please Wait...\n\n')
    time.sleep(600)

    if ask_question:
        continue_to_run_question("Re-training Google model is finished. Should I continue?")

    # Testing the second Google model: ##############################
    print('Testing the re-trained Google model:')
    command = "mkdir " + project_directory + 'second_test_results'
    os.system(command)
    print('Testing the test file on the re-trained model:')
    testing_path = project_directory + 'testing_validation/test_data.yml'
    testingData = prepare_data_test(testing_path)

    for example, intent in testingData.items():
        text_input = dialogflow.TextInput(text=example, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)
        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise

        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        intent.append(response.query_result.intent.display_name)
        testingData[example] = intent
        time.sleep(random.randint(1, 2))
    pd.DataFrame.from_dict(data=testingData, orient='index').to_csv(project_directory + 'second_test_results/' +
                                                                    'results.csv', header=False)

    if ask_question:
        continue_to_run_question("Testing the second Google model is finished. Should I continue?")
    print('\nPrinting the results:')
    first_f1 = calculate_f1(project_directory + 'first_test_results/' + 'results.csv')
    second_f1 = calculate_f1(project_directory + 'second_test_results/' + 'results.csv')
    print('\nBaseline F1-score: ', first_f1)
    print('\nAlphaBot F1-score: ', second_f1)
    print('\nSee you soon!')
