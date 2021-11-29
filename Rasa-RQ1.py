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

    current_datetime = datetime.datetime.now(pytz.timezone('America/Montreal'))
    timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
    #  Getting the fraction for train-test and the output directory: ##############################
    total = 0
    while abs(total - 1.00) > 1e-5:
        questions = [
            inquirer.Text('training-fraction', message='Please enter the training-fraction for Rasa', default=0.4),
            inquirer.Text('testing-fraction', message='Please enter the testing-fraction for Rasa', default=0.3),
            inquirer.Text('validation-fraction', message='Please enter the validation-fraction for Rasa', default=0.3),
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
        inquirer.List('training-type',
                      message="How do you want to re-train the Rasa after applying the LFs?",
                      choices=[
                          ('Incremental training', 'i'),
                          ('Train again from scratch', 's')
                      ],
                      carousel=True,
                      default='s'
                      ),
    ]
    answers = inquirer.prompt(questions, theme=GreenPassion())
    training_type = answers['training-type']

    questions = [
        inquirer.List('ask-question-in-each-step',
                      message="Should I ask the question for continue each single step>",
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
    print('Splitting the Rasa nlu data to training, testing, and validation sets:')
    os.system(command)

    command = "bash Bash/Rasa/make_data_for_train.sh " + \
              "-t " + timestamp
    print('Making data ready for training the Rasa:')
    os.system(command)

    if ask_question:
        continue_to_run_question("Test-Training-Validation split is finished. Should I continue?")
    # Training the first Rasa Model: ##############################

    print('Training the Rasa model:')
    command = "bash Bash/Rasa/rasa_train_from_scratch.sh " + \
              " -d " + project_directory + 'train' + \
              " -o " + project_directory + 'models'
    os.system(command)

    if ask_question:
        continue_to_run_question("Training Rasa model is finished. Should I continue?")
    # Testing the first Rasa model ##############################

    list_of_files = glob.glob(project_directory + 'models/*.tar.gz')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Testing the Rasa model:')
    command = "mkdir " + project_directory + 'first_test_results'
    os.system(command)
    command = "bash Bash/Rasa/rasa_test_model.sh " + \
              " -t " + project_directory + 'testing_validation/test_data.yml' + \
              " -o " + project_directory + 'first_test_results' + \
              " -m " + latest_file
    print('Testing the test file on the trained model:')
    os.system(command)

    if ask_question:
        continue_to_run_question("Testing the first Rasa model is finished. Should I continue?")

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
    # Making data for the second Rasa model from the predictions of the Label model: ##############################

    command = "bash Bash/Rasa/make_data_for_train_from_new_nlu.sh " + \
              " -t " + timestamp + \
              " -f " + 'new_nlu.yml'
    print('Creating the new data directory for re-training:')
    os.system(command)

    if ask_question:
        continue_to_run_question("The re-train directory (with its files) is created. Should I continue?")

    # Training the second Rasa model (choosing between 2 options): ##############################
    if training_type == 'i':
        command = "bash Bash/Rasa/rasa_incremental_train.sh" + \
                  " -m " + latest_file + \
                  " -d " + project_directory + 're-train' + \
                  " -o " + project_directory + 'models'
        print('Incremental Rasa training:')
        os.system(command)
    elif training_type == 's':

        command = "bash Bash/Rasa/rasa_train_from_scratch.sh" + \
                  " -d " + project_directory + 're-train' + \
                  " -o " + project_directory + 'models'
        print('re-training Rasa again from scratch:')
        os.system(command)

    if ask_question:
        continue_to_run_question("Re-training Rasa model is finished. Should I continue?")
    # Testing the second Rasa model: ##############################

    list_of_files = glob.glob(project_directory + 'models/*.tar.gz')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Testing the re-trained Rasa model:')
    command = "mkdir " + project_directory + 'second_test_results'
    os.system(command)
    command = "bash Bash/Rasa/rasa_test_model.sh " + \
              " -t " + project_directory + 'testing_validation/test_data.yml' + \
              " -o " + project_directory + 'second_test_results' + \
              " -m " + latest_file
    print('Testing the test file on the re-trained model:')
    os.system(command)

    if ask_question:
        continue_to_run_question("Testing the second Rasa model is finished. Should I continue?")
    # Comparing the two previous tested models: ##############################
    with open(project_directory + 'first_test_results/intent_report.json') as json_file:
        data = json.load(json_file)
    json_file.close()
    first_f1 = data['weighted avg']['f1-score']
    with open(project_directory + 'second_test_results/intent_report.json') as json_file:
        data = json.load(json_file)
    json_file.close()
    second_f1 = data['weighted avg']['f1-score']
    print('\nBaseline F1-score: ', first_f1)
    print('\nAlphaBot F1-score: ', second_f1)
    print('\nSee you soon!')
