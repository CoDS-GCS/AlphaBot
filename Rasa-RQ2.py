import datetime
import os
import pytz
from classes.data_tools import DataTools
from classes.engineer import Engineer
from classes.labeling_functions import LabelingFunctions
import csv
import json
import glob
import inquirer
from inquirer.themes import GreenPassion
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm  # Collect all the font names available to matplotlib
import pandas as pd


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
    chatbot_directory = "/vagrant/"
    question_1 = [
        inquirer.Confirm('models',
                         message="Do you want to train the 9 different Rasa models from scratch (very time-consuming) "
                                 + "or used the baseline ones in the repository?",
                         default=False
                         ),
    ]
    answer_1 = inquirer.prompt(question_1, theme=GreenPassion())

    question_2 = [
        inquirer.Checkbox('evaluations',
                          message="Which models do you choose for the evaluations? (Choosing by pressing the SPACE)",
                          choices=[('10%-45%-45% Train-Test-Validate', 10), ('20%-40%-40% Train-Test-Validate', 20),
                                   ('30%-35%-35% Train-Test-Validate', 30), ('40%-30%-30% Train-Test-Validate', 40),
                                   ('50%-25%-25% Train-Test-Validate', 50), ('60%-20%-20% Train-Test-Validate', 60),
                                   ('70%-15%-15% Train-Test-Validate', 70), ('80%-10%-10% Train-Test-Validate', 80),
                                   ('90%-5%-5% Train-Test-Validate', 90)],
                          ),
    ]
    answer_2 = inquirer.prompt(question_2)

    question_3 = [
        inquirer.Text('experiments', message='Please enter the number of experiments you want to have: ', default=1)
    ]
    answer_3 = inquirer.prompt(question_3, theme=GreenPassion())

    question_4 = [
        inquirer.Text('iterations', message='Please enter the number of iterations for each of the experiments: ',
                      default=1)
    ]
    answer_4 = inquirer.prompt(question_4, theme=GreenPassion())

    current_datetime = datetime.datetime.now(pytz.timezone('America/Montreal'))
    timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
    project_directory = "/vagrant/Output/" + timestamp + '/'
    command = "mkdir " + project_directory
    os.system(command)

    rasa_models_directory = ''
    if not answer_1['models']:
        rasa_models_directory = chatbot_directory + "Rasa-Baseline-Results/"
        pass
    else:
        rasa_models_directory = project_directory + 'models/'
        command = "mkdir " + rasa_models_directory
        os.system(command)

        for training_fraction in answer_2['evaluations']:
            model_directory = rasa_models_directory + str(training_fraction) + '/'
            command = "mkdir " + model_directory
            os.system(command)

            command = "cp -r /vagrant/Dataset/Paper/ " + model_directory + 'data'
            os.system(command)

            testing_fraction = (100 - training_fraction) / 2
            testing_fraction = testing_fraction / (100 - training_fraction)
            validation_fraction = 1 - testing_fraction
            command = "bash Bash/Rasa/rasa_data_split_with_validation.sh " + \
                      " -f " + str(training_fraction / 100) + \
                      " -t " + str(testing_fraction) + \
                      " -p " + model_directory[:len(model_directory) - 1]
            print(f'Splitting the Rasa nlu data to training {training_fraction}%, testing, and validation sets:')
            os.system(command)

            path = model_directory[:len(model_directory) - 1]
            command = "bash Bash/Rasa/make_data_for_train_rq2.sh " + \
                      "-p " + path
            print('Making data ready for training the Rasa:')
            os.system(command)

            print('Training the Rasa model:')
            command = "bash Bash/Rasa/rasa_train_from_scratch.sh " + \
                      " -d " + model_directory + 'train' + \
                      " -o " + model_directory + 'models'
            os.system(command)

            list_of_files = glob.glob(model_directory + 'models/*.tar.gz')
            latest_file = max(list_of_files, key=os.path.getctime)
            print('Testing the Rasa model:')
            command = "mkdir " + model_directory + 'test_results'
            os.system(command)

            command = "bash Bash/Rasa/rasa_test_model.sh " + \
                      " -t " + model_directory + 'testing_validation/test_data.yml' + \
                      " -o " + model_directory + 'test_results' + \
                      " -m " + latest_file
            print('Testing the test file on the trained model:')
            os.system(command)

    labeling_function_class = LabelingFunctions()
    functions = {
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
    experiments = int(answer_3['experiments'])
    iterations = int(answer_4['iterations'])
    path = project_directory

    for experiment in range(1, experiments + 1):
        functions = shuffle_dictionary(functions)
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
            for model in answer_2['evaluations']:
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
                        rasa_models_directory + str(model) +
                        '/testing_validation/training_data.yml')
                    pandas_dataframe = dataTools.list_of_commands_to_pandas_dataframe(list_of_commands)
                    dataTools.pandas_dataframe_to_csv(pandas_dataframe)
                    engineer = Engineer(pandas_dataframe, dict_of_function, cardinality=52,
                                        data_tools_instance=dataTools)
                    engineer.produce_labeling_matrix()
                    engineer.predict_by_majority_vote_model()
                    dataTools.save_predicted_commands_with_intents(dict_of_intents, model='majority')
                    nlu_file_path = rasa_models_directory + str(model) + \
                        '/train/nlu.yml'
                    dataTools.add_predictions_to_nlu(nlu_file_path, dict_of_intents, model='majority')

                    command = "bash Bash/Rasa/make_data_for_train_from_new_nlu_rq2.sh " + \
                              " -p " + step_directory + \
                              " -f " + 'new_nlu.yml'
                    print('Creating the new data directory for re-training:')
                    os.system(command)

                    command = "mkdir " + step_directory + 'model/'
                    os.system(command)
                    command = "bash Bash/Rasa/rasa_train_from_scratch.sh" + \
                              " -d " + step_directory + 're-train' + \
                              " -o " + step_directory + 'model'
                    print('re-training Rasa again from scratch:')
                    os.system(command)

                    list_of_files = glob.glob(step_directory + 'model/*.tar.gz')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print('Testing the re-trained Rasa model:')
                    command = "mkdir " + step_directory + 'test_results'
                    os.system(command)
                    command = "bash Bash/Rasa/rasa_test_model.sh " + \
                              " -t " + rasa_models_directory + str(model) + \
                              '/testing_validation/test_data.yml' + \
                              " -o " + step_directory + 'test_results' + \
                              " -m " + latest_file
                    print('Testing the test file on the re-trained model:')
                    os.system(command)

                    with open(step_directory + 'test_results/intent_report.json') as json_file:
                        data = json.load(json_file)
                    json_file.close()
                    accuracy = data['accuracy']
                    print('\naccuracy of the model: ', accuracy)

                    command = "rm -rf " + step_directory + 'model/*'
                    os.system(command)
                    print('\nThe current experiment is finished!')

    exp_list = [name for name in os.listdir(project_directory) if os.path.isdir(os.path.join(project_directory, name))]
    for exp in exp_list:
        if exp == 'models':
            continue
        exp_directory = project_directory + exp + '/'
        iteration_list = [name for name in os.listdir(exp_directory) if
                          os.path.isdir(os.path.join(exp_directory, name))]
        for iteration in iteration_list:
            iteration_directory = exp_directory + iteration + '/'
            model_list = [name for name in os.listdir(iteration_directory) if
                          os.path.isdir(os.path.join(iteration_directory, name))]
            for model in model_list:
                model_directory = iteration_directory + model + '/'
                before_path = rasa_models_directory + model + '/' + 'test_results/intent_report.json'
                with open(before_path) as json_file:
                    data = json.load(json_file)
                json_file.close()
                f1_before = data['weighted avg']['f1-score']

                errors_path = rasa_models_directory + model + '/' + 'test_results/intent_errors.json'
                with open(errors_path) as json_file:
                    data = json.load(json_file)
                json_file.close()
                errors_before = len(data)
                step_list = [name for name in os.listdir(model_directory) if
                             os.path.isdir(os.path.join(model_directory, name))]
                results = []
                for step in step_list:
                    step_directory = model_directory + step + '/'
                    after_path = step_directory + 'test_results/intent_report.json'
                    if os.path.isfile(after_path):
                        with open(after_path) as json_file:
                            data = json.load(json_file)
                        json_file.close()
                        f1_after = data['weighted avg']['f1-score']
                        number_of_predictions = len(open(step_directory + 'engineer/predicted.csv').readlines()) - 1
                        errors_path = step_directory + 'test_results/intent_errors.json'
                        with open(errors_path) as json_file:
                            data = json.load(json_file)
                        json_file.close()
                        errors_after = len(data)
                        results.append([exp, iteration, model, step, "%0.4f" % (f1_before * 100),
                                        "%0.4f" % (f1_after * 100),
                                        "%0.4f" % ((f1_after - f1_before) * 100),
                                        number_of_predictions, errors_before, errors_after,
                                        errors_after - errors_before])
                fields = ['experiment', 'iteration', 'model', 'step_lf', 'f1_before', 'f1_after', 'difference',
                          'number_of_predictions', 'intent_errors_before', 'intent_errors_after', 'errors_difference']
                results = sorted(results, key=lambda x: int(x[3]))  # step: 3
                # print(results)
                with open(project_directory + 'rasa-rq2-' + exp + '-' + iteration + '-' + model + '.csv', 'w') as f:
                    write = csv.writer(f)
                    write.writerow(fields)
                    write.writerows(results)
                data_1 = pd.read_csv(project_directory + 'rasa-rq2-' + exp + '-' + iteration + '-' + model + '.csv')
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
                        label='rasa-rq2-' + exp + '-' + iteration + '-' + model)
                # Add the x and y-axis labels
                ax.set_xlabel('Number of applied LFs', labelpad=10, fontweight='bold', fontsize=18.5)
                ax.set_ylabel('Improvement in F1-score (%)', labelpad=10, fontweight='bold', fontsize=18.5)
                # Set the axis limits
                ax.set_xlim(0, 70)
                ax.set_ylim(-10, 60)
                # Add legend to plot
                ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=18.5)
                # Save figure
                plt.savefig(project_directory + 'rasa-rq2-' + exp + '-' + iteration + '-' + model + '.pdf', dpi=300,
                            transparent=False, bbox_inches='tight')
                plt.savefig(project_directory + 'rasa-rq2-' + exp + '-' + iteration + '-' + model + '.png', dpi=300,
                            transparent=False, bbox_inches='tight')
                # Show figure
                # plt.show()
