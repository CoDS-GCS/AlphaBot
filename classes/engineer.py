from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
import pandas as pd
import warnings
from classes.data_tools import DataTools


class Engineer:

    def __init__(self, dataframe, label_functions, cardinality, data_tools_instance, show_warnings=False):
        if not show_warnings:
            warnings.filterwarnings('ignore', '.*W008.*', category=UserWarning)
            warnings.simplefilter(action='ignore', category=FutureWarning)
        self.dataframe = dataframe
        self.labeling_functions = self.__pick_active_labeling_functions(label_functions)
        self.data_tools = data_tools_instance
        self.cardinality = cardinality
        self.labeling_matrix = None
        self.majority_vote_model = MajorityLabelVoter(cardinality)
        self.label_model = LabelModel(cardinality, verbose=False)

    def produce_labeling_matrix(self):
        self.__pandas_lf_applier()

    def train_label_model(self, labeling_matrix):
        self.label_model.fit(L_train=labeling_matrix)
        self.data_tools.save_to_pickle_file(self.label_model, self.data_tools.project_directory + 'label_model.p')

    def load_label_model(self, pickle_file_path):
        self.label_model = self.data_tools.load_pickle_file(pickle_file_path)

    def predict_by_label_model(self):
        predictions = []
        for i in range(0, len(self.labeling_matrix)):
            results_from_functions_on_command = self.labeling_matrix[i:i + 1, ]
            label_model_prediction = self.label_model.predict(results_from_functions_on_command)
            predictions.append([label_model_prediction[0]])
        self.data_tools.list_of_predictions_to_csv(predictions, 'label_model_predictions.csv')

    def predict_by_majority_vote_model(self):
        predictions = []
        for i in range(0, len(self.labeling_matrix)):
            results_from_functions_on_command = self.labeling_matrix[i:i + 1, ]
            majority_prediction = self.majority_vote_model.predict(results_from_functions_on_command)
            predictions.append([majority_prediction[0]])
        self.data_tools.list_of_predictions_to_csv(predictions, 'majority_vote_predictions.csv')

    @staticmethod
    def __pick_active_labeling_functions(label_functions):
        list_of_active_functions = []
        for function, activation in label_functions.items():
            if activation:
                list_of_active_functions.append(function)
        return list_of_active_functions

    def __pandas_lf_applier(self):
        pandas_lf_applier = PandasLFApplier(lfs=self.labeling_functions)
        self.labeling_matrix = pandas_lf_applier.apply(df=self.dataframe)
        self.data_tools.save_matrix_to_csv(self.labeling_matrix, self.data_tools.project_directory
                                           + 'labeling_matrix.csv')
        self.data_tools.save_to_pickle_file(self.labeling_matrix, self.data_tools.project_directory
                                            + 'labeling_matrix.p')
        self.__labeling_functions_summary(self.labeling_matrix)

    def __labeling_functions_summary(self, labeling_matrix):
        summary = LFAnalysis(L=labeling_matrix, lfs=self.labeling_functions).lf_summary()
        self.data_tools.save_to_pickle_file(summary, self.data_tools.project_directory +
                                            'labeling_functions_summary.p')

    def print_labeling_functions_summary(self):
        file_path = self.data_tools.project_directory + 'labeling_functions_summary.p'
        self.data_tools.print_pickle_file(file_path)

    def print_labeling_matrix(self):
        file_path = self.data_tools.project_directory + 'labeling_matrix.p'
        self.data_tools.print_pickle_file(file_path)


# I wrote the following code just to show the usage and a sample for this class
if __name__ == "__main__":
    from labeling_functions import LabelingFunctions
    input_data = {'text': ['when was the most recent PR?', 'how many commits are in the branch rel-1.3.0?',
                           'who are the active developers?'],
                  'intent': ['MOST_RECENT_PRS', 'NUMBER_OF_PRS', 'TOP_CONTRIBUTORS'],
                  }
    labeling_functions = LabelingFunctions()
    functions = {
        labeling_functions.most_recent_prs_1: True,
        labeling_functions.top_contributors_1: True,
        labeling_functions.number_of_commits_in_branch_1: True
    }
    df = pd.DataFrame(input_data, columns=['text', 'intent'])
    output_directory = '/vagrant/outputs/'
    dataTools = DataTools(output_directory)
    engineer = Engineer(df, functions, labeling_functions.CARDINALITY, dataTools)
    engineer.produce_labeling_matrix()
