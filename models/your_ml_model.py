from MyChatbotData import MyChatbotData
from machine_learning import MachineLearningModel
from text_processing import preprocess

class MLClassifier:
    def __init__(self, data):
        self.data = data
        self.ml_model = MachineLearningModel()

    def train(self):
        X_train = self.data.df['phrase']
        y_train = self.data.df['intent']
        self.ml_model.train(X_train, y_train)

    def predict(self, query):
        preprocessed_query = preprocess(query)
        return self.ml_model.predict([preprocessed_query])[0]