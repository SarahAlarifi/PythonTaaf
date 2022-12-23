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
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import tensorflow as tf
# import tflite as tflite




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


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# # testx    = testing[cols]
# # testy    = testing['prognosis']  
# # testy    = le.transform(testy)


# clf1  = DecisionTreeClassifier()
# clf = clf1.fit(x_train,y_train)



# print(clf.score(x_train,y_train))
# # print ("cross result========")
# scores = cross_val_score(clf, x_test, y_test, cv=3)
# # print (scores)
# print (scores.mean())


# df = pd.read_csv('Data/Training.csv')
# X = df.iloc[:, :-1]
# y = df['prognosis']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# rf_clf = DecisionTreeClassifier()
# rf_clf.fit(X_train, y_train)
# scores = cross_val_score(rf_clf, X_test, y_test, cv=3)

# # model=SVC()
# # model.fit(x_train,y_train)
# # print("for svm: ")
# # print(model.score(x_test,y_test))

# importances = rf_clf.feature_importances_
# indices = np.argsort(importances)[::-1]
# features = cols



# to start python application with apis
app = Flask(__name__) 

df = pd.read_csv('Data/Training.csv')
X = df.iloc[:, :-1]
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_clf = DecisionTreeClassifier()
rf_clf.fit(X_train, y_train)
scores = cross_val_score(rf_clf, X_test, y_test, cv=10)

# model=SVC()
# model.fit(x_train,y_train)
# print("for svm: ")
# print(model.score(x_test,y_test))

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


#########   API starts here  ############
@app.route('/s')# first trial
def hello_world():
    json_file = {}
    json_file['query'] = 'hello_world'
    return jsonify(json_file)

#########   API1   ############
@app.route('/SymptopmsApi' , methods =['POST'])# api to get the symptomp based on the first symptopm chosen from the model
def get_Symptopms():
    tree = rf_clf
    feature_names = cols
    data = request.get_json()
    disease_input = data['disease']
    # disease_input = "headache"
    days = 2 #soso try to edit this to get the number of days the user has been sick
    tree_ = tree.tree_
    print(tree_.feature)#contains numbers or indexes of the features 
    print("hereee")
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print(feature_name)# contains fatures names , some are undefined 

    chk_dis=",".join(feature_names).split(",")
    print(chk_dis)#all symptomps in the dataset 
    symptoms_present = []


    num_days= days
       
    symptoms_given =[]
    def recurse(node, depth):#this method is nested inside the API
     
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
           
            threshold = tree_.threshold[node]
          

            if name == disease_input:
                val = 1
            else:
                val = 0

            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                print(symptoms_present)
                print("the symptopm present after the elas if the value larger than threshold ")
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            print(red_cols)
            x=0
            
            for index , item in enumerate(red_cols):# this loop has been added by saly so the first symptomp chosen from the human body will be a part of the array that will go inside the model for predicting the disese
                if(item == disease_input):
                   print("here" + item)
                   x = index # index of the first symp
                   print(x)



            print("here is red_cols")
            global symptoms_given
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print(x)
            # symptoms_given.append([pd.Index([x])])
            symptoms_given.insert(0, red_cols[x])
            symptoms_given.append([pd.Index([red_cols[x]])])
            print(symptoms_given.append([pd.Index([red_cols[x]])]))#for test
            symptoms_given = symptoms_given.insert(0, red_cols[x])
            print(symptoms_given)#this contains all the symptopms that the bot is going to ask the user about 
            print(reduced_data.loc[present_disease].values[0].nonzero())#for test
            #the symptopm that he is going to ask about
            # print(present_disease) # the disese !!
        return symptoms_given
    

    symptoms_given = recurse(0, 1)
    print("check")
    print(symptoms_given)
    return json.dumps({'result' : symptoms_given.tolist()})
    # return json.dump(dumped)

# df = pd.read_csv('Data/Training.csv')
# X = df.iloc[:, :-1]
# y = df['prognosis']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
# rf_clf = DecisionTreeClassifier()
# rf_clf.fit(X_train, y_train)
# scores = cross_val_score(rf_clf, X_test, y_test, cv=3)
#########   API2   ############
@app.route('/DiseaseApi' , methods =['POST'])
def Disease():
    data = request.get_json()
    symptoms_exp = data['disease']
    print(symptoms_exp)#contains all the symptopms the user chose
    print("symptopms that the patient said yes to")
   
    # print (scores)
    print (scores.mean())
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    print(input_vector)
    for item in symptoms_exp:
      print(item)
      print(symptoms_dict[item])
      input_vector[[symptoms_dict[item]]] = 1
      
    print(input_vector)#zeroes and ones array after the users answer 
    print(rf_clf.predict([input_vector]))
    print("inside second prediction")
    json_file = {}
    json_file['query'] = rf_clf.predict([input_vector])
    return jsonify({'result' : ((rf_clf.predict([input_vector])).tolist())})
    # final prediction is this


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))



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
    app.run(debug=True)


################### APIs Ends Here ########################################






def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    print(exp)
    print("here is the expression")
    print(days)
    print("here is the days")
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)




def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)
    # json_file = {}
    # json_file['query'] = "Hello, " + name
    # return jsonify(json_file)

# if __name__ == '__main__':
#     app.run()

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]

#will taks symptomp array as zeroes and ones and return the disease

def sec_predict(symptoms_exp):
    print(symptoms_exp)
    print("here is what i want")
    print("symptopms that the patient said yes to")
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    scores = cross_val_score(rf_clf, X_test, y_test, cv=3)
    # print (scores)
    print (scores.mean())
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    print(input_vector)
    for item in symptoms_exp:
      print(item)
      print(symptoms_dict[item])
      input_vector[[symptoms_dict[item]]] = 1
      
    print(input_vector)#zeroes and ones array after the users answer 
    print(rf_clf.predict([input_vector]))
    print("inside second prediction")
    return rf_clf.predict([input_vector]) # final prediction is this


# def print_disease(node):
#     node = node[0]
#     val  = node.nonzero() 
#     disease = le.inverse_transform(val[0])
#     return list(map(lambda x:x.strip(),list(disease)))

############################################################################################

#based on the first disease it will produce the set of symptomps
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    print(tree_.feature)#contains numbers or indexes of the features 
    print("hereee")
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print(feature_name)# contains fatures names , some are undefined 

    chk_dis=",".join(feature_names).split(",")
    print(chk_dis)#all symptomps 
    symptoms_present = []


    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = input("")# this should be coming from flutter
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        
        if conf==1: # if one then the entered symptopm is a valid symptom 
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")

    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            print(name)#symptopm name
            print("here is the name") 
            threshold = tree_.threshold[node]
            print(threshold) # from 0 to 1 
            print("here is the threshold")

            if name == disease_input:
                val = 1
            else:
                val = 0

            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                print(symptoms_present)
                print("the symptopm present after the elas if the value larger than threshold ")
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            print(red_cols)
            x=0
            symptoms_given =[]
            for index , item in enumerate(red_cols):
                if(item == disease_input):
                   print("here" + item)
                   x = index 
                   symptoms_given.append(x)
                   print(x)



            print("here is red_cols")
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print(x)
            # symptoms_given.append([pd.Index([x])])
            symptoms_given.insert(0, red_cols[x])
            symptoms_given.append([pd.Index([red_cols[x]])])
            print(symptoms_given.append([pd.Index([red_cols[x]])]))
            symptoms_given = symptoms_given.insert(0, red_cols[x])
            print(symptoms_given)
            print(reduced_data.loc[present_disease].values[0].nonzero())#the symptopm that he is going to ask about
            print(present_disease) # the disese !!
           
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms) #only if yes .. these are the symptopms that are anwered yes

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
               

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])
                
            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)
######################################################################################
# tf.keras.models.save_model(clf,'model.pbtxt')
# convertor = tf.lite.TFLiteConverter.from_keras_model(model=clf)
# modelconvertor = convertor.convert()
# open("model.tflite","web").write(modelconvertor)
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
# get_Symptopms()
tree_to_code(rf_clf,cols)

print("----------------------------------------------------------------------------------------")

