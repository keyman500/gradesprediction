import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

class gradespredicter:
    def __init__(self):
        self.data = pd.read_csv("student-mat.csv", sep=";")
        self.data = self.data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
        self.predict = "G3"
    
    def train(self):
        X = np.array(self.data.drop([self.predict], 1))
        y = np.array(self.data[self.predict])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print("accuracy: ",acc)
        return x_test,y_test ,linear

    def prediction(self,data):
        x_test,y_test,linear = self.train()
        predict = linear.predict(data)
        for x in range(len(predict)):
            print("prediction: ",predict[x]," data: ",data[x])
   
    def test(self):
        x_test,y_test,linear = self.train()
        predictions = linear.predict(x_test)
        for x in range(len(predictions)):
            print(predictions[x], x_test[x], y_test[x])

g = gradespredicter()
g.test()


