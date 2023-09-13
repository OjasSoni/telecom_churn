print("System Started")
from unittest import result
import numpy as np # Multi-dimensional array object
import pandas as pd # Data Manipulation
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sns # Data Visualization
import plotly.express as px # Interactive Data Visualizations
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot # Offline version of the Plotly modules.
import cufflinks as cf # Works as a connector between the pandas library and plotly
cf.go_offline() 
init_notebook_mode(connected=True) # To connect Jupyter notebook with JavaScript
from jupyterthemes import jtplot # Jupyter theme
jtplot.style(theme= 'monokai', context= 'notebook', ticks= True, grid= False)
from sklearn.model_selection import train_test_split
import pickle
print("Dependencies loaded successfully")


# load model

filename = "models/gnb.pickle"

model1 = pickle.load(open(filename, "rb"))


filename = "models/knn.pickle"

model2 = pickle.load(open(filename, "rb"))



filename = "models/lr.pickle"

model3 = pickle.load(open(filename, "rb"))


filename = "models/rf.pickle"

model4 = pickle.load(open(filename, "rb"))


filename = "models/svc.pickle"

model5 = pickle.load(open(filename, "rb"))




# webapp starts here:


import streamlit as st

# var1 = st.number_input('Enter State')
var1=25.9984
# var2 = st.number_input('Enter Account Length')
var2=100.25
# var3 = st.number_input('Enter Area Code')
# var4 = st.number_input('Enter Phone Number')
var5 = st.number_input('Enter International Plan')
var6 = st.number_input('Enter Voice Mail Plan')
# var7 = st.number_input('Enter number_vmail_messages')
var7=7.75
# var8 = st.number_input('Enter total_day_minutes')
var8=180
# var9 = st.number_input('Enter total_day_calls')
var9=100
var10 = st.number_input('Enter total_day_charge')
# var11 = st.number_input('Enter total_eve_minutes')
var11=200.6
# var12 = st.number_input('Enter total_eve_calls')
var12=100.191
# var13 = st.number_input('Enter total_eve_charge')
var13=17.05
# var14 = st.number_input('Enter total_night_minutes')
var14=200.3
# var15 = st.number_input('Enter total_night_calls')
var15=99.9
# var16 = st.number_input('Enter total_night_charge')
var16=9.01
# var17 = st.number_input('Enter total_intl_minutes')
var17=10.26
# var18 = st.number_input('Enter total_intl_calls')
var18=4.43
# var19 = st.number_input('Enter total_intl_charge')
var19=2.77
var20 = st.number_input('Enter number_customer_service_calls')

list1=[]
list1.append(var1)
list1.append(var2)
# list1.append(var3)
# list1.append(var4)
list1.append(var5)
list1.append(var6)
list1.append(var7)
list1.append(var8)
list1.append(var9)
list1.append(var10)
list1.append(var11)
list1.append(var12)
list1.append(var13)
list1.append(var14)
list1.append(var15)
list1.append(var16)
list1.append(var17)
list1.append(var18)
list1.append(var19)
list1.append(var20)

list2=[]
list2.append(list1)


result=[]
result.append(model1.predict(list2))

result.append(model2.predict(list2))

result.append(model3.predict(list2))

result.append(model4.predict(list2))

result.append(model5.predict(list2))

ans=max(result, key=result.count)
# result=model_rf.predict(list2)



st.write('The current number is ', ans)


