import re
import numpy as np
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from flask import Flask
import csv
import warnings
from flask import jsonify
from flask import request
from sklearn.metrics import mean_squared_error
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from IPython.display import Image
import pydotplus
import pandas as pd
from sklearn import tree
from io import StringIO
from sklearn.tree import export_graphviz
import pydot

from sklearn.tree import export_graphviz
import pydot
import pydotplus


warnings.filterwarnings("ignore", category=DeprecationWarning)





training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)




# to start python application with apis
app = Flask(__name__) 

df = pd.read_csv('Data/pre_processed.csv')
df = df.drop_duplicates()
y = df["prognosis"]
X = df.drop('prognosis',axis=1)
X_train,X_test , y_train , y_test= train_test_split( X , y ,test_size=0.3 , random_state = 42  )
rnd_forest = RandomForestClassifier() # the algorithm used to train the model

rnd_forest.fit(X_train.values ,y_train.values ) # training the model using the algorithm

accuracy2 = cross_val_score(rnd_forest,X_train.values ,y_train.values ,cv=5).mean() # training accuracy 
accuracy3 = cross_val_score(rnd_forest,X_test.values ,y_test.values ,cv=5).mean() # training accuracy 
y_pred = rnd_forest.predict(X_test.values)
errors = abs(accuracy2 - accuracy3)
accuracy= accuracy_score(y_test.values,y_pred) # testing accuracy


# Import tools needed for visualization dot -Tpng tree.dot -o tree.png

# Pull out one tree from the forest
treeq = rnd_forest.estimators_[0]

out_file = tree.export_graphviz(
    treeq,
    feature_names   = X.columns,
    class_names     = rnd_forest.predict(X_test),
    filled          = True,
    rounded         = True
)
graph = pydotplus.graph_from_dot_data(out_file)
# Image(graph.create_png())
#graph.write_png('tree.png') 
@app.route('/s')# first trial
def hello_world():
    json_file = {}
    json_file['query'] = 'hello_world'
    return jsonify(json_file)


#########   API2   ############
@app.route('/DiseaseApi' , methods =['POST'])
def Disease():
    data = request.get_json()
    symptoms_exp = data['disease']
    print(symptoms_exp)#contains all the symptopms the user chose
    print("symptopms that the patient said yes to" + "\n")
   
    # print (scores)
    print("Testing accuracy \n")
    print(accuracy)
    print("Testing  accuracy 2 \n")
    print(errors)
    # print("third")
    print(accuracy2)
    print("Testing  accuracy 3 \n")
    print(accuracy3)
   
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    print(input_vector)# just zeroes 
    for item in symptoms_exp: # symptopms coming from the user in array
      print(item)
      print(symptoms_dict[item])
      input_vector[[symptoms_dict[item]]] = 1 # will see these these symptopms in the all syptopms direcory and turn them into ones in they are in symptopms exp array
      
    print(input_vector)#zeroes and ones array after the users answer 
    print(rnd_forest.predict([input_vector]))
    print("inside second prediction")
    json_file = {}
    json_file['query'] = rnd_forest.predict([input_vector]) # putting the result in json file to send it back to the front end 
    return jsonify({'result' : ((rnd_forest.predict([input_vector])).tolist())})
    # final prediction is this





description_list = dict()
#########   API3  ############
@app.route('/DescriptionApi' , methods =['POST'])
def getDiseaseDescription():
    global description_list
    data = request.get_json()
    disease = data['disease']
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

    return jsonify({'result' : description_list[disease]})

severityDictionary=dict()
precautionDictionary=dict()


#########   API4 ############
@app.route('/PrecautionApi' , methods =['POST'])
def getprecautionDict():
    data = request.get_json()
    disease = data['disease']
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)
            
    # precution_list=precautionDictionary[disease]
    return jsonify({'result' : precautionDictionary[disease]})

#########   API5   ############
@app.route('/SeverityApi' , methods =['POST'])
def getSeverityDict():
    global severityDictionary
    data = request.get_json()
    exp = data['expression']
    days = data['days']
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
    
    symptoms_dict = {}

    for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
    def calc_condition(exp,days):
      sum=0
      for item in exp:
         sum=sum+severityDictionary[item]
      if((sum*days)/(len(exp)+1)>13):
        return "You should take the consultation from doctor. "
      else:
        return "It might not be that bad but you should take precautions."

    severity = calc_condition(exp,days)  #takes the array of symptopms and days to see it the disease is serious 
    return jsonify({'result' : severity})

    
if __name__ == '__main__':
       app.run('0.0.0.0',port=5001)


################### APIs Ends Here ########################################






