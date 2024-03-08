import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Diabetes.csv',encoding='latin1')
print(data.head())
data['Output'] = data['Glucose'].apply(lambda x: 'diabetes' if x > 125 else 'prediabetes' if x > 99 and x <= 125 else 'normal' if x > 70  else 'diabetes')
print(data.head())
data['Output'] = data['Output'].replace(['normal','prediabetes', 'diabetes'],[0,1,2])
data.head()
data.to_csv('diabetes2.csv',index=False, header=True)
data= data[data['Insulin'] == 0]
data.describe()
data = data[['Glucose','Output']]
data.head()
data= data[data['Glucose'] != 0]
data.describe()
data.to_csv('diabetestest2.csv',index=False, header=True)
data = pd.read_csv('diabetes2.csv',encoding='latin1')
data.head()
data = data[['Glucose','Insulin','Output']]
data.head()
data= data[data['Insulin'] != 0]
data= data[data['Glucose'] != 0]
data.to_csv('diabetesNonzero.csv', index=False)
data.head()
data.describe()
from sklearn.model_selection import train_test_split
splitRatio = 0.2
train , test = train_test_split(data,test_size = splitRatio,random_state = 123)
X_train = train[[x for x in train.columns if x not in ["Insulin"]]]
y_train = train[["Insulin"]]
X_test  = test[[x for x in test.columns if x not in ["Insulin"]]]
y_test  = test[["Insulin"]]
from sklearn.model_selection import train_test_split
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
models =  LinearRegression()
models.fit(X_train,y_train)
prediction = models.predict(X_test)
new_df = pd.read_csv('diabetestest2.csv',encoding='latin1')
prediction = models.predict(new_df)
prediction = prediction.astype(int)
out=pd.DataFrame(prediction, columns=['Insulin'])
out.to_csv('diabetespredresult.csv',index=False, header=True)
df1 = pd.read_csv('diabetestest2.csv',encoding='latin1')
df2 = pd.read_csv('diabetespredresult.csv',encoding='latin1')
data = pd.concat([df1, df2], axis = 1)
data = data[['Glucose','Insulin','Output']]
df3 = pd.read_csv('diabetesNonzero.csv',encoding='latin1')
data.head()
frames = [data, df3]
data = pd.concat(frames)
data.to_csv('diabetesmerge.csv', index=False)
data = pd.read_csv('diabetesmerge.csv',encoding='latin1')
from sklearn.model_selection import train_test_split
splitRatio = 0.2
train , test = train_test_split(data,test_size = splitRatio,random_state = 123)
X_train = train[[x for x in train.columns if x not in ["Output"]]]
y_train = train[["Output"]]
X_test  = test[[x for x in test.columns if x not in ["Output"]]]
y_test  = test[["Output"]]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
accuracy_score(y_test,prediction)
import tkinter as tk
import pandas as pd
def saveinfo():
    Valor=int(E1.get())
    Valor1=int(E2.get())
    list=[[Valor,Valor1]]
    df=pd.DataFrame(list)
    df.to_csv("newdf.csv")
    top.destroy()
top=tk.Tk()
top.title("First input for glucose and insulin data")
L1=tk.Label(top,text="Enter Glucose value")
L1.place(x=10,y=10)
E1=tk.Entry(top,bd=5)
E1.place(x=300,y=10)
L2=tk.Label(top,text="Enter Insulin value")
L2.place(x=10,y=50)
E2=tk.Entry(top,bd=5)
E2.place(x=300,y=50)
b=tk.Button(top,text="Submit",command=saveinfo)
b.place(x=10,y=90)
top.geometry('500x500')
top.mainloop()
new_df=pd.read_csv('newdf.csv',encoding='latin1',names=["Glucose","Insulin"])
new_df=new_df[new_df["Glucose"]!=0]
prediction = model.predict(new_df)
model1=LogisticRegression()
model1.fit(X_train,y_train)
prediction1 = model1.predict(new_df)

model2=GaussianNB()
model2.fit(X_train,y_train)
prediction2 = model2.predict(new_df)

new = model.predict_proba(new_df)[:]
new1 = model1.predict_proba(new_df)[:]
new2 = model2.predict_proba(new_df)[:]
msg = ''
if prediction == 0:
	msg = 'Normal'
elif prediction == 1:
	msg = 'Prediabetic'
elif prediction == 2:
    	msg = 'Diabetic'

Proba = int(((new[:,2])) * 100)
value="your  K-neighbour Diabetic Status is  "+ msg +". You have "+format(Proba)+" % chances of being diabetic."
if prediction1 == 0:
	msg = 'Normal'
elif prediction1 == 1:
	msg = 'Prediabetic'
elif prediction1 == 2:
    	msg = 'Diabetic'

Proba1 = int(((new1[:,2])) * 100)
value7="your Logistics Diabetic Status is  "+ msg +". You have "+format(Proba1)+" % chances of being diabetic."
if prediction2 == 0:
	msg = 'Normal'
elif prediction2 == 1:
	msg = 'Prediabetic'
elif prediction2 == 2:
    	msg = 'Diabetic'

Proba2 = int(((new2[:,2])) * 100)
value8="your Naive Bayes Diabetic Status is  "+ msg +". You have "+format(Proba2)+" % chances of being diabetic."
from sklearn.model_selection import train_test_split
splitRatio = 0.2
train , test = train_test_split(data,test_size = splitRatio,random_state = 123)
X_train = train[[x for x in train.columns if x not in ["Output"]]]
y_train = train[["Output"]]
X_test  = test[[x for x in test.columns if x not in ["Output"]]]
y_test  = test[["Output"]]
from sklearn.model_selection import train_test_split
model = KNeighborsClassifier()
model.fit(X_train,y_train)
model1=LogisticRegression()
model1.fit(X_train,y_train)
model2=GaussianNB()
model2.fit(X_train,y_train)
prediction = model.predict(X_test)
res=tk.Tk()
res.title("OUTPUT")
var=tk.StringVar()
var1=tk.StringVar()
var2=tk.StringVar()
var3=tk.StringVar()
var4=tk.StringVar()
var5=tk.StringVar()
var6=tk.StringVar()
var7=tk.StringVar()
var8=tk.StringVar()
value1="K-Neighbor Train accuracy is : "+ str(model.score(X_train, y_train))
value2="K-Neighbor Test accuracy is : "+ str(model.score(X_test, y_test))
value3="Logistic Regression Train accuracy is : "+ str(model1.score(X_train, y_train))
value4="Logistic Regression Test accuracy is : "+ str(model1.score(X_test, y_test))
value5="Gaussian NB Train accuracy is : "+ str(model2.score(X_train, y_train))
value6="Gaussian NB Test accuracy is : "+ str(model2.score(X_test, y_test))
lab=tk.Label(res,textvariable=var1)
lab.place(x=10,y=10)
lab.config(font=("Courier", 14))
lab1=tk.Label(res,textvariable=var2)
lab1.place(x=10,y=70)
lab1.config(font=("Courier", 14))
lab2=tk.Label(res,textvariable=var3)
lab2.place(x=10,y=140)
lab2.config(font=("Courier", 14))
lab3=tk.Label(res,textvariable=var4)
lab3.place(x=10,y=210)
lab3.config(font=("Courier", 14))
lab4=tk.Label(res,textvariable=var5)
lab4.place(x=10,y=280)
lab4.config(font=("Courier", 14))
lab5=tk.Label(res,textvariable=var6)
lab5.place(x=10,y=350)
lab5.config(font=("Courier", 14))
lab6=tk.Label(res,textvariable=var)
lab6.place(x=10,y=420)
lab6.config(font=("Forte", 14))
lab7=tk.Label(res,textvariable=var7)
lab7.place(x=10,y=480)
lab7.config(font=("Forte", 14))
lab8=tk.Label(res,textvariable=var8)
lab8.place(x=10,y=520)
lab8.config(font=("Forte", 14))
var.set(value)
var1.set(value1)
var2.set(value2)
var3.set(value3)
var4.set(value4)
var5.set(value5)
var6.set(value6)
var7.set(value7)
var8.set(value8)
res.geometry('1000x500')
res.mainloop()
list1=[model.score(X_test, y_test),model1.score(X_test, y_test),model2.score(X_test, y_test)]
plot=pd.DataFrame(list1,columns=["TestValues"])
sns.set_style('whitegrid')
sns.distplot(plot["TestValues"],kde=False,color='darkred',bins=30)
plt.show()

list3=[(Proba),(Proba1),(Proba2)]
plot3=pd.DataFrame(list3,columns=["TestValues"])
sns.set_style('whitegrid')
sns.distplot(plot3["TestValues"],kde=False,color='darkred',bins=30)
plt.show()
import os
os.remove("newdf.csv")
	
