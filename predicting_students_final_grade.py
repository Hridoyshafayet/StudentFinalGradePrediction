import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as py_plot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3

# print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

print("CO: \n", linear.coef_)
print("CO: \n", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

style.use("dark_background")

# finding the co-relations between different attributes using a scatter plot

p = 'G1'
final_grade = 'G3'
py_plot.scatter(data[p], data[final_grade])
py_plot.xlabel(p)
py_plot.ylabel("Final Grade")
py_plot.show()



