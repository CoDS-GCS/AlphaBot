import datetime
import os
import pytz
from classes.data_tools import DataTools
from classes.engineer import Engineer
from classes.labeling_functions import LabelingFunctions
import json
import glob
import inquirer
from inquirer.themes import GreenPassion
from snorkel.labeling import labeling_function
import ast
import random
from google.cloud import dialogflow
import asyncio
import yaml
import re
import time
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import csv
import sys
from csv import reader
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm  # Collect all the font names available to matplotlib


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


def calculate_results(path):
    with open(path, 'r') as csv_file:
        csv_reader = reader(csv_file)
        first_list_of_rows = list(csv_reader)

    first_total = len(first_list_of_rows)
    first_correct = 0
    for row in first_list_of_rows:
        if row[1] == row[2]:
            first_correct += 1
    first_accuracy = (first_correct / first_total) * 100
    return [first_correct, first_total, first_accuracy]


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


def shuffle_dictionary(input_dictionary):
    l = list(input_dictionary.items())
    random.shuffle(l)
    return dict(l)


@labeling_function()
def dummy_method(x):
    return -1


if __name__ == '__main__':
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.weight"] = "bold"
    # Generate 2 colors from the 'tab10' colormap
    colors = cm.get_cmap('tab10', 2)
    # font
    font_names = [f.name for f in fm.fontManager.ttflist]
    current_datetime = datetime.datetime.now(pytz.timezone('America/Montreal'))
    timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
    if len(sys.argv) - 1:
        argument_list = sys.argv
        answer_model = int(argument_list[1])
        answer_experiment = int(argument_list[2])
        answer_iteration = int(argument_list[3])
        answer_directory = timestamp
        answer_home_directory = argument_list[4]
        answer_pause = int(argument_list[5])
        answer_train_pause = int(argument_list[6])
        answer_bot_name = argument_list[7]
        answer_bot_key = argument_list[8]
    else:
        questions = [
            inquirer.List('dataset',
                          message="Which baseline model do you want to use for this experiment?",
                          choices=[
                              ('10-45-45 splits.', 10),
                              ('30-35-35 splits.', 30),
                              ('50-25-25 splits.', 50),
                              ('70-15-15 splits.', 70),
                              ('90-5-5 splits.', 90)
                          ],
                          carousel=True,
                          default=10
                          ),
            inquirer.Text('experiment', message='Please enter experiment (random shuffle of LFs)', default=1),
            inquirer.Text('iteration', message='Please enter iteration (more iteration, more time needed)', default=1),
            inquirer.Text('default-directory', message='Please enter the default directory for the outputs',
                          default=timestamp),
        ]
        answers = inquirer.prompt(questions, theme=GreenPassion())
        answer_model = int(answers['dataset'])
        answer_experiment = int(answers['experiment'])
        answer_iteration = int(answers['iteration'])
        answer_directory = answers['default-directory']
        answer_home_directory = '/vagrant/'
        answer_pause = 3
        answer_train_pause = 500

    display_name = "commit_creator"
    parent = dialogflow.AgentsClient.agent_path('chatbot-ws-arqe')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'keys/private_arqe.json'
    DIALOGFLOW_PROJECT_ID = 'chatbot-ws-arqe'
    SESSION_ID = 'me'
    DIALOGFLOW_LANGUAGE_CODE = 'en'
    intents_client = dialogflow.IntentsClient()

    chatbot_directory = answer_home_directory
    project_directory = chatbot_directory + "Output/" + answer_directory + '/'
    command = "mkdir " + project_directory
    os.system(command)

    labeling_function_class = LabelingFunctions()
    functions_helper = {
        labeling_function_class.most_recent_prs_1: 'most_recent_prs_1',
        labeling_function_class.top_contributors_1: 'top_contributors_1',
        labeling_function_class.number_of_commits_in_branch_1: 'number_of_commits_in_branch_1',
        labeling_function_class.pr_closing_date_1: 'pr_closing_date_1',
        labeling_function_class.pr_closing_date_2: 'pr_closing_date_2',
        labeling_function_class.pr_creation_date_1: 'pr_creation_date_1',
        labeling_function_class.pr_creation_date_2: 'pr_creation_date_2',
        labeling_function_class.pr_assignees_1: 'pr_assignees_1',
        labeling_function_class.number_of_prs_1: 'number_of_prs_1',
        labeling_function_class.number_of_prs_2: 'number_of_prs_2',
        labeling_function_class.activity_report_1: 'activity_report_1',
        labeling_function_class.activity_report_2: 'activity_report_2',
        labeling_function_class.contributions_by_developer_1: 'contributions_by_developer_1',
        labeling_function_class.latest_commit_1: 'latest_commit_1',
        labeling_function_class.default_branch_1: 'default_branch_1',
        labeling_function_class.developers_with_most_open_issues_1: 'developers_with_most_open_issues_1',
        labeling_function_class.developers_with_most_open_issues_2: 'developers_with_most_open_issues_2',
        labeling_function_class.files_changed_by_pr_1: 'files_changed_by_pr_1',
        labeling_function_class.intial_commit_in_branch_1: 'intial_commit_in_branch_1',
        labeling_function_class.issue_closer_1: 'issue_closer_1',
        labeling_function_class.issue_closing_date_1: 'issue_closing_date_1',
        labeling_function_class.issue_closing_date_2: 'issue_closing_date_2',
        labeling_function_class.issue_contributors_1: 'issue_contributors_1',
        labeling_function_class.issue_creation_date_1: 'issue_creation_date_1',
        labeling_function_class.issue_creation_date_2: 'issue_creation_date_2',
        labeling_function_class.issue_creator_1: 'issue_creator_1',
        labeling_function_class.last_developer_to_touch_a_file_1: 'last_developer_to_touch_a_file_1',
        labeling_function_class.latest_commit_in_branch_1: 'latest_commit_in_branch_1',
        labeling_function_class.latest_release_1: 'latest_release_1',
        labeling_function_class.list_languages_1: 'list_languages_1',
        labeling_function_class.number_of_collaborators_2: 'number_of_collaborators_2',
        labeling_function_class.number_of_collaborators_1: 'number_of_collaborators_1',
        labeling_function_class.number_of_downloads_1: 'number_of_downloads_1',
        labeling_function_class.number_of_stars_1: 'number_of_stars_1',
        labeling_function_class.number_of_subscribers_1: 'number_of_subscribers_1',
        labeling_function_class.number_of_watchers_1: 'number_of_watchers_1',
        labeling_function_class.pr_contributors_1: 'pr_contributors_1',
        labeling_function_class.pr_creator_1: 'pr_creator_1',
        labeling_function_class.repository_creation_date_1: 'repository_creation_date_1',
        labeling_function_class.repository_license_1: 'repository_license_1',
        labeling_function_class.repository_owner_1: 'repository_owner_1',
        labeling_function_class.repository_owner_2: 'repository_owner_2',
        labeling_function_class.repository_topics_1: 'repository_topics_1',
        labeling_function_class.commits_in_pr_1: 'commits_in_pr_1',
        labeling_function_class.issue_assignees_1: 'issue_assignees_1',
        labeling_function_class.longest_open_pr_1: 'longest_open_pr_1',
        labeling_function_class.largest_files_1: 'largest_files_1',
        labeling_function_class.file_creator_1: 'file_creator_1',
        labeling_function_class.issue_related_commits_1: 'issue_related_commits_1',
        labeling_function_class.list_branches_1: 'list_branches_1',
        labeling_function_class.list_collaborators_2: 'list_collaborators_2',
        labeling_function_class.list_collaborators_1: 'list_collaborators_1',
        labeling_function_class.list_releases_1: 'list_releases_1',
        labeling_function_class.number_of_branches_1: 'number_of_branches_1',
        labeling_function_class.number_of_branches_2: 'number_of_branches_2',
        labeling_function_class.number_of_commits_1: 'number_of_commits_1',
        labeling_function_class.number_of_commits_2: 'number_of_commits_2',
        labeling_function_class.number_of_forks_1: 'number_of_forks_1',
        labeling_function_class.number_of_forks_2: 'number_of_forks_2',
        labeling_function_class.number_of_issues_1: 'number_of_issues_1',
        labeling_function_class.number_of_issues_2: 'number_of_issues_2',
        labeling_function_class.pr_closer_1: 'pr_closer_1',
        labeling_function_class.most_recent_issues_1: 'most_recent_issues_1',
        labeling_function_class.longest_open_issue_1: 'longest_open_issue_1',
        labeling_function_class.most_recent_prs_2: 'most_recent_prs_2',
        labeling_function_class.intial_commit_1: 'intial_commit_1',
        labeling_function_class.top_contributors_2: 'top_contributors_2',
        labeling_function_class.main_programming_language_1: 'main_programming_language_1',
        labeling_function_class.file_creator_2: 'file_creator_2',
        labeling_function_class.number_of_commits_in_branch_2: 'number_of_commits_in_branch_2'
    }
    dict_of_intents = {
        0: 'MOST_RECENT_PRS', 1: 'NUMBER_OF_COMMITS_IN_BRANCH', 2: 'TOP_CONTRIBUTORS',
        3: 'INTIAL_COMMIT', 4: 'MOST_RECENT_ISSUES', 5: 'LONGEST_OPEN_ISSUE',
        6: 'LATEST_COMMIT', 7: 'CONTRIBUTIONS_BY_DEVELOPER', 8: 'ACTIVITY_REPORT',
        9: 'NUMBER_OF_PRS', 10: 'PR_ASSIGNEES', 11: 'PR_CREATION_DATE',
        12: 'PR_CLOSING_DATE', 13: 'PR_CLOSER', 14: 'NUMBER_OF_ISSUES', 15: 'NUMBER_OF_FORKS',
        16: 'NUMBER_OF_COMMITS', 17: 'NUMBER_OF_BRANCHES', 18: 'LIST_RELEASES',
        19: 'LIST_COLLABORATORS',
        20: 'LIST_BRANCHES', 21: 'ISSUE_RELATED_COMMITS', 22: 'FILE_CREATOR',
        23: 'LARGEST_FILES',
        24: 'LONGEST_OPEN_PR', 25: 'ISSUE_ASSIGNEES', 26: 'COMMITS_IN_PR',
        27: 'REPOSITORY_TOPICS', 28: 'REPOSITORY_OWNER', 29: 'REPOSITORY_LICENSE',
        30: 'REPOSITORY_CREATION_DATE', 31: 'PR_CREATOR', 32: 'PR_CONTRIBUTORS',
        33: 'NUMBER_OF_WATCHERS', 34: 'NUMBER_OF_SUBSCRIBERS', 35: 'NUMBER_OF_STARS',
        36: 'NUMBER_OF_DOWNLOADS', 37: 'NUMBER_OF_COLLABORATORS', 38: 'LIST_LANGUAGES',
        39: 'LATEST_RELEASE', 40: 'LATEST_COMMIT_IN_BRANCH',
        41: 'LAST_DEVELOPER_TO_TOUCH_A_FILE',
        42: 'ISSUE_CREATOR', 43: 'ISSUE_CREATION_DATE', 44: 'ISSUE_CONTRIBUTORS',
        45: 'ISSUE_CLOSING_DATE', 46: 'ISSUE_CLOSER', 47: 'INTIAL_COMMIT_IN_BRANCH',
        48: 'FILES_CHANGED_BY_PR', 49: 'DEVELOPERS_WITH_MOST_OPEN_ISSUES', 50: 'DEFAULT_BRANCH',
        51: 'MAIN_PROGRAMMING_LANGUAGE'
    }
    experiments = [answer_experiment]
    iterations = answer_iteration
    path = project_directory
    for experiment in experiments:
        functions = shuffle_dictionary(functions_helper)
        temporary_dict = {}
        i = 1
        for key, value in functions.items():
            temporary_dict[i] = value
            i += 1
        with open(project_directory + "shuffle-functions-" + str(experiment) + ".txt", 'w') as file:
            file.write(json.dumps(temporary_dict))  # use `json.loads` to do the reverse
        file = open(
            project_directory + "shuffle-functions-" + str(experiment) + ".txt", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        for number, func in dictionary.items():
            lf = list(functions_helper.keys())[list(functions_helper.values()).index(func)]
            functions[lf] = func
        experiment_directory = path + 'e' + str(experiment) + '/'
        command = "mkdir " + experiment_directory
        os.system(command)
        temporary_dict = {}
        i = 1
        for key, value in functions.items():
            temporary_dict[i] = value
            i += 1
        with open(experiment_directory + 'shuffle-functions.txt', 'w') as file:
            file.write(json.dumps(temporary_dict))
        for iteration in range(1, iterations + 1):
            iteration_directory = experiment_directory + 'i' + str(iteration) + '/'
            command = "mkdir " + iteration_directory
            os.system(command)
            for model in [answer_model]:
                model_directory = iteration_directory + str(model) + '/'
                command = "mkdir " + model_directory
                os.system(command)
                dict_of_function = {dummy_method: True}
                counter = 1
                for function, function_name in functions.items():
                    dict_of_function[function] = True
                    step_directory = model_directory + str(counter) + '/'
                    counter += 1
                    command = "mkdir " + step_directory
                    os.system(command)
                    path_of_engineer = step_directory + 'engineer/'
                    command = "mkdir " + path_of_engineer
                    os.system(command)

                    dataTools = DataTools(path_of_engineer, fixed_directory=True)
                    list_of_commands = dataTools.yml_to_input_commands(
                        chatbot_directory + 'Dialogflow-Baseline-Results/' + str(model) +
                        '/testing_validation/training_data.yml')
                    pandas_dataframe = dataTools.list_of_commands_to_pandas_dataframe(list_of_commands)
                    dataTools.pandas_dataframe_to_csv(pandas_dataframe)
                    engineer = Engineer(pandas_dataframe, dict_of_function, cardinality=52,
                                        data_tools_instance=dataTools)
                    engineer.produce_labeling_matrix()
                    engineer.predict_by_majority_vote_model()
                    dataTools.save_predicted_commands_with_intents(dict_of_intents, model='majority')
                    nlu_file_path = chatbot_directory + 'Dialogflow-Baseline-Results/' + str(model) + '/train/nlu.yml'
                    dataTools.add_predictions_to_nlu(nlu_file_path, dict_of_intents, model='majority')

                    command = "bash Bash/Dialogflow/make_data_for_train_from_new_nlu.sh " + \
                              " -t " + step_directory + \
                              " -f " + 'new_nlu.yml'
                    print('Creating the new data directory for re-training:')
                    os.system(command)

                    # Train the google NLU
                    print('Training the Google NLU model:')
                    trainingData = prepare_data(step_directory + 're-train/nlu.yml')
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
                            time.sleep(random.randint(answer_pause, answer_pause + 1))
                    # Create the intent
                    for key in trainingData.keys():
                        create_intent(DIALOGFLOW_PROJECT_ID,
                                      key,
                                      trainingData[key])
                        time.sleep(random.randint(answer_pause, answer_pause + 1))
                    # Train the Dialogflow
                    asyncio.wait(train_agent)
                    print('\n\nTraining Dialogflow...Please Wait...\n\n')
                    time.sleep(answer_train_pause)

                    # Testing the google NLU
                    print('Testing the Google model:')
                    print('Testing the test file on the trained model:')
                    testing_path = chatbot_directory + 'Dialogflow-Baseline-Results/' + str(model) + \
                        '/testing_validation/test_data.yml'
                    testingData = prepare_data_test(testing_path)
                    command = "mkdir " + step_directory + 'test_results'
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
                        time.sleep(random.randint(answer_pause, answer_pause + 1))
                    pd.DataFrame.from_dict(data=testingData, orient='index').to_csv(
                        step_directory + 'test_results/' +
                        'results.csv', header=False)
    exp_list = [name for name in os.listdir(project_directory) if os.path.isdir(os.path.join(project_directory, name))]
    for exp in exp_list:
        exp_directory = project_directory + exp + '/'
        iteration_list = [name for name in os.listdir(exp_directory) if
                          os.path.isdir(os.path.join(exp_directory, name))]
        for iteration in iteration_list:
            iteration_directory = exp_directory + iteration + '/'
            model_list = [name for name in os.listdir(iteration_directory) if
                          os.path.isdir(os.path.join(iteration_directory, name))]
            for model in model_list:
                model_directory = iteration_directory + model + '/'
                before_path = 'Dialogflow-Baseline-Results/' + model + '/' + \
                              'test_results/results.csv'
                f1_before = calculate_f1(before_path)
                errors_before = 0
                step_list = [name for name in os.listdir(model_directory) if
                             os.path.isdir(os.path.join(model_directory, name))]
                results = []
                for step in step_list:
                    step_directory = model_directory + step + '/'
                    after_path = step_directory + 'test_results/results.csv'
                    if os.path.isfile(after_path):
                        f1_after = calculate_f1(after_path)
                        errors_after = 0
                        number_of_predictions = len(open(step_directory + 'engineer/predicted.csv').readlines()) - 1
                        results.append([exp, iteration, model, step, "%0.4f" % f1_before,
                                        "%0.4f" % f1_after, "%0.4f" % (f1_after - f1_before),
                                        number_of_predictions, errors_before, errors_after,
                                        errors_after - errors_before])
                fields = ['experiment', 'iteration', 'model', 'step_lf', 'f1_before', 'f1_after', 'difference',
                          'number_of_predictions', 'intent_errors_before', 'intent_errors_after', 'errors_difference']
                results = sorted(results, key=lambda x: int(x[3]))  # step: 3
                with open(project_directory + 'Dialogflow-rq2-' + exp + '-' + iteration + '-' + model + '.csv',
                          'w') as f:
                    write = csv.writer(f)
                    write.writerow(fields)
                    write.writerows(results)
                data_1 = pd.read_csv(
                    project_directory + 'Dialogflow-rq2-' + exp + '-' + iteration + '-' + model + '.csv')
                data_plot = data_1['difference']
                # Create figure object and store it in a variable called 'fig'
                fig = plt.figure(figsize=(5, 5))
                # Add axes object to our figure that takes up entire figure
                ax = fig.add_axes([0, 0, 1, 1])
                # Edit the major and minor ticks of the x and y axes
                ax.xaxis.set_tick_params(which='major', size=7, width=2, direction='out')
                ax.xaxis.set_tick_params(which='minor', size=4, width=1, direction='out')
                ax.yaxis.set_tick_params(which='major', size=7, width=2, direction='out')
                ax.yaxis.set_tick_params(which='minor', size=4, width=1, direction='out')
                # Edit the major and minor tick locations of x and y axes
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
                ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
                ax.plot(data_1['step_lf'].to_numpy(), data_plot.to_numpy(), 'g--', linewidth=2.5, color='orange',
                        label='Dialogflow-rq2-' + exp + '-' + iteration + '-' + model)
                # Add the x and y-axis labels
                ax.set_xlabel('Number of applied LFs', labelpad=10, fontweight='bold', fontsize=18.5)
                ax.set_ylabel('Improvement in F1-score (%)', labelpad=10, fontweight='bold', fontsize=18.5)
                # Set the axis limits
                ax.set_xlim(0, 70)
                ax.set_ylim(-10, 60)
                # Add legend to plot
                ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=18.5)
                # Save figure
                plt.savefig(project_directory + 'Dialogflow-rq2-' + exp + '-' + iteration + '-' + model + '.pdf',
                            dpi=300,
                            transparent=False, bbox_inches='tight')
                plt.savefig(project_directory + 'Dialogflow-rq2-' + exp + '-' + iteration + '-' + model + '.png',
                            dpi=300,
                            transparent=False, bbox_inches='tight')
                # Show figure
                # plt.show()
