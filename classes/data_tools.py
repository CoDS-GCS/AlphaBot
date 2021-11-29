import os
import pytz
import datetime
import yaml
import csv
import pandas as pd
import pickle
from numpy import asarray
from numpy import savetxt
import re
from classes.entity import EntityDetector


class DataTools:

    def __init__(self, output_directory=None, fixed_directory=False, time_zone='America/Montreal'):
        self.output_timestamp = self.__create_timestamp(time_zone)
        self.project_directory = None
        if output_directory is not None and not fixed_directory:
            self.__create_project_directory(output_directory)
        elif output_directory is not None and fixed_directory:
            self.project_directory = output_directory
        self.entityDetector = EntityDetector()

    @staticmethod
    def __create_timestamp(time_zone):
        current_datetime = datetime.datetime.now(pytz.timezone(time_zone))
        return current_datetime.strftime("%Y%m%d-%H%M%S")

    def __create_project_directory(self, output_directory):
        try:
            project_directory = output_directory + self.output_timestamp + '/'
            os.mkdir(project_directory)
            self.project_directory = project_directory
        except OSError as error:
            print('directory exists!')
            exit(0)

    def create_new_nlu_for_finetune(self, dict_of_intents):
        list_of_commands_path = self.project_directory + 'list_of_commands.csv'
        list_of_commands = []
        with open(list_of_commands_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # This skips the first row of the CSV file.
            next(csv_reader)
            for row in csv_reader:
                list_of_commands.append(row)
        csv_file.close()
        label_model_predictions_path = self.project_directory + 'label_model_predictions.csv'
        list_of_predictions = []
        with open(label_model_predictions_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                list_of_predictions.append(row)
        csv_file.close()
        cmd_and_pred_dict = {list_of_commands[i][0]: int(list_of_predictions[i][0]) for i in
                             range(len(list_of_commands))}
        s2 = '  '
        s4 = '    '
        with open(self.project_directory + 'finetune_nlu.yml', 'w') as nlu_file:
            nlu_file.write('version: "2.0"\n')
            nlu_file.write('nlu:\n')
            nlu_file.write('- regex: filename\n'+s2+'examples: |\n' + s4 + r'- [ ^\\] * \.(\w+)' + '\n')
            nlu_file.write('- regex: number\n'+s2+'examples: |\n' + s4 + r'- [0 - 9] +' + '\n')
        nlu_file.close()
        for command, prediction in cmd_and_pred_dict.items():
            if prediction != -1:
                with open(self.project_directory + 'finetune_nlu.yml') as f:
                    nlu_content = f.readlines()
                f.close()
                intent_str = dict_of_intents[prediction].lower()
                try:
                    index = nlu_content.index('- intent: ' + intent_str + '\n')
                except ValueError:
                    with open(self.project_directory + 'finetune_nlu.yml', 'a') as nlu_file:
                        nlu_file.write('- intent: ' + intent_str + '\n'+s2+'examples: |\n')
                        nlu_file.write(s4 + '- ' + self.__add_entities_to_command(command) + '\n')
                    nlu_file.close()
                    with open(self.project_directory + 'finetune_nlu.yml') as f:
                        nlu_content = f.readlines()
                    f.close()
                    index = nlu_content.index('- intent: ' + intent_str + '\n')
                first_or_default = next((i + index + 1 for i, x in enumerate(nlu_content[index + 1:])
                                         if x.startswith('- intent')), None)
                if first_or_default:
                    line = nlu_content[first_or_default - 1]
                    white_space = re.match(r"\s*", line).group()
                    b = nlu_content[:]
                    b[first_or_default:first_or_default] = [white_space + '- ' + self.__add_entities_to_command(command)
                                                            + '\n']
                    nlu_content = b
                else:
                    line = nlu_content[index - 1]
                    white_space = re.match(r"\s*", line).group()
                    nlu_content.append(white_space + '- ' + self.__add_entities_to_command(command) + '\n')

        return True

    def add_predictions_to_nlu(self, nlu_file_path, dict_of_intents, model='majority'):
        list_of_commands_path = self.project_directory + 'list_of_commands.csv'
        list_of_commands = []
        with open(list_of_commands_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # This skips the first row of the CSV file.
            next(csv_reader)
            for row in csv_reader:
                list_of_commands.append(row)
        csv_file.close()
        if model == 'majority':
            label_model_predictions_path = self.project_directory + 'majority_vote_predictions.csv'
        else:
            label_model_predictions_path = self.project_directory + 'label_model_predictions.csv'
        list_of_predictions = []
        with open(label_model_predictions_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                list_of_predictions.append(row)
        csv_file.close()
        cmd_and_pred_dict = {list_of_commands[i][0]: int(list_of_predictions[i][0]) for i in
                             range(len(list_of_commands))}
        with open(nlu_file_path) as f:
            nlu_content = f.readlines()
        for command, prediction in cmd_and_pred_dict.items():  # this is not necessary due to the splitting
            try:
                index = nlu_content.index('- ' + command + '\n')
                nlu_content.pop(index)
            except ValueError:
                continue
        for command, prediction in cmd_and_pred_dict.items():
            if prediction != -1:
                intent_str = dict_of_intents[prediction].lower()
                try:
                    index = nlu_content.index('- intent: ' + intent_str + '\n')
                except ValueError:
                    print('error in NLU file! NLU file needs modification!')
                    return False
                first_or_default = next((i + index + 1 for i, x in enumerate(nlu_content[index + 1:])
                                         if x.startswith('- intent')), None)
                if first_or_default:
                    line = nlu_content[first_or_default - 1]
                    white_space = re.match(r"\s*", line).group()
                    b = nlu_content[:]
                    b[first_or_default:first_or_default] = [white_space + '- ' + self.__add_entities_to_command(command)
                                                            + '\n']
                    nlu_content = b
                else:
                    line = nlu_content[index - 1]
                    white_space = re.match(r"\s*", line).group()
                    nlu_content.append(white_space + '- ' + self.__add_entities_to_command(command) + '\n')

        new_nlu_file = open(self.project_directory + 'new_nlu.yml', 'w')
        new_nlu_file.writelines(nlu_content)
        new_nlu_file.close()
        return True

    def __add_entities_to_command(self, command_text):
        entities_list = ['filename', 'issue_status', 'issue_number']
        entities = self.entityDetector.detect_entities(command_text)
        entities = sorted(entities, key=lambda x: x['start'])
        counter = 0
        for entity in entities:
            if entity['dim'] in entities_list:
                replacement = '[' + entity['body'] + ']' + '(' + entity['dim'] + ')'
                command_text = command_text[0:entity['start'] + counter] + replacement + command_text[
                                                                                         entity['end'] + 1 +
                                                                                         counter:]
                counter += 4 + len(entity['dim'])
        return command_text

    def save_predicted_commands_with_intents(self, dict_of_intents, model='majority'):
        list_of_commands_path = self.project_directory + 'list_of_commands.csv'
        list_of_commands = []
        with open(list_of_commands_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # This skips the first row of the CSV file.
            next(csv_reader)
            for row in csv_reader:
                list_of_commands.append(row)
        csv_file.close()
        if model == 'majority':
            label_model_predictions_path = self.project_directory + 'majority_vote_predictions.csv'
        else:
            label_model_predictions_path = self.project_directory + 'label_model_predictions.csv'
        list_of_predictions = []
        with open(label_model_predictions_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                list_of_predictions.append(row)
        csv_file.close()
        cmd_and_pred_dict = {list_of_commands[i][0]: int(list_of_predictions[i][0]) for i in
                             range(len(list_of_commands))}
        with open(self.project_directory + 'predicted.csv', 'w') as csv_file:
            csv_file.write('command,intent\n')
            for key in cmd_and_pred_dict.keys():
                if cmd_and_pred_dict[key] != -1:
                    csv_file.write("%s,%s\n" % (key, dict_of_intents[cmd_and_pred_dict[key]].lower()))
        csv_file.close()
        pass

    def user_input_to_input_command(self, user_input):
        list_of_commands = [[self.__remove_entities_from_command(user_input)]]
        self.list_of_commands_to_csv(list_of_commands)
        return list_of_commands

    def yml_to_input_commands(self, yml_test_data):
        with open(yml_test_data) as yml_file:
            data = yaml.load(yml_file, Loader=yaml.FullLoader)
            contents = data['nlu']
        list_of_commands = []
        list_of_commands_with_intents = []
        for content in contents:
            if 'intent' in content:
                intent = content['intent']
                commands = content['examples'].split("\n")[:-1]
                commands = [command[2:] for command in commands]
                for command in commands:
                    list_of_commands.append([self.__remove_entities_from_command(command)])
                    list_of_commands_with_intents.append([self.__remove_entities_from_command(command), intent])
        self.list_of_commands_to_csv(list_of_commands)
        self.list_of_commands_to_csv(list_of_commands_with_intents, file_name='list_of_commands_with_intents.csv')
        return list_of_commands

    @staticmethod
    def __remove_entities_from_command(command_text):
        entities = ['filename', 'issue_status', 'issue_number']
        regular_expressions = []
        for entity in entities:
            regular_expressions.append(r'\[[^\[]*\]\(' + entity + r'\)',)
        for regular_expression in regular_expressions:
            founded_entities = [entity.group() for entity in
                                re.finditer(regular_expression, command_text, flags=re.IGNORECASE)]
            for entity in founded_entities:
                command_text = command_text.replace(entity, entity[1:entity.find(']')], 1)
        return command_text

    def list_of_predictions_to_csv(self, list_of_predictions, file_name='predictions.csv'):
        file_path = self.project_directory + file_name
        with open(file_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(list_of_predictions)
        csv_file.close()

    def list_of_commands_to_csv(self, list_of_commands, file_name='list_of_commands.csv'):
        columns = ['command']
        file_path = self.project_directory + file_name
        with open(file_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columns)
            csv_writer.writerows(list_of_commands)
        csv_file.close()

    @staticmethod
    def list_of_commands_to_pandas_dataframe(list_of_commands):
        dataframe = pd.DataFrame(list_of_commands, columns=['text'])
        return dataframe

    def pandas_dataframe_to_csv(self, dataframe, file_name='pandas_dataframe.csv', index=False, header=True):
        file_path = self.project_directory + file_name
        dataframe.to_csv(file_path, index=index, header=header)

    def csv_to_pandas_dataframe(self, file_name, encoding='ISO-8859-1'):
        file_path = self.project_directory + file_name
        pandas_dataframe = pd.read_csv(file_path, encoding=encoding)
        return pandas_dataframe

    @staticmethod
    def convert_spacy_tokens_to_dataframe(tokens):
        spacy_pos_tagged = [(token, token.tag_, token.pos_, token.lemma_) for token in tokens]
        pandas_dataframe_of_tokens = pd.DataFrame(spacy_pos_tagged,
                                                  columns=['Word', 'POS tag', 'Tag type', 'Lemma'])
        return pandas_dataframe_of_tokens

    @staticmethod
    def save_to_pickle_file(object_to_save, file_path):
        pickle.dump(object_to_save, open(file_path, "wb"))

    @staticmethod
    def save_matrix_to_csv(matrix_to_save, file_path):
        data = asarray(matrix_to_save)
        savetxt(file_path, data, delimiter=',')

    @staticmethod
    def print_pickle_file(pickle_file_to_read):
        content = pickle.load(open(pickle_file_to_read, "rb"))
        DataTools.print_to_terminal(content, 'Pickle file')

    @staticmethod
    def load_pickle_file(pickle_file_to_load):
        content = pickle.load(open(pickle_file_to_load, "rb"))
        return content

    @staticmethod
    def print_to_terminal(content, representative):
        print(representative + ' :', end='\n')
        print(content, end='\n')
