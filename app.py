import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

DIABET01 = ['No Diabetes','Pre-Diabetes or Diabetes']
NY01 = ['No','Yes']
SEX = ['Female','Male']
AGE5YR = ['Age 18-24','Age 25-29','Age 30-34','Age 35-39','Age 40-44','Age 45-49','Age 50-54','Age 55-59','Age 60-64','Age 65-69','Age 70-74','Age 75-79','Age 80 or older']
EDUCA = ['Never attended school or only kindergarten','Grades 1 through 8 (Elementary)','Grades 9 through 11 (Some high school)','Grade 12 or GED (High school graduate)','College 1 year to 3 years (Some college or technical school)','College 4 years or more (College graduate)']
INCOME2 = ['Less than $10,000','Less than $15,000','Less than $20,000','Less than $25,000','Less than $35,000','Less than $50,000','Less than $70,000','Less than $75,000 or more']
GENHEALTH = ['Excelent','Very Good','Good','Fair','Poor']

def modelling():
     df = pd.read_csv('data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
     df.drop(['AnyHealthcare', 'NoDocbcCost'], axis = 1, inplace=True)
     df_temp = df.copy()
     df_class = df_temp.pop('Diabetes_binary')
     df_normalized = MinMaxScaler().fit_transform(df_temp)
     x_train,x_test,y_train,y_test = train_test_split(df_normalized,df_class,test_size=0.2,random_state=0)
     clf = AdaBoostClassifier(random_state=0, learning_rate=0.9500000000000002, n_estimators=85)
     return clf.fit(x_train,y_train), list(df_temp.columns)
def predict(data):
     model,col = modelling()
     return model.predict(pd.DataFrame([data],columns=col))

#Apps
st.title(':mag_right: Diabetes Check')
st.subheader('Fill in the questionnaire below to check diabetes')

with st.form('form'):
     highbp = NY01.index(st.radio('Do you have high blood pressure?',NY01))
     highchol = NY01.index(st.radio('Do you have high cholestrol?',NY01))
     cholcheck = NY01.index(st.radio('Have you checked your cholesterol in past 5 years?',NY01))
     weight = st.number_input('What\'s your weight in kilograms?',min_value=0)
     height = st.number_input('What\'s your height in centimeters?',min_value=0)
     smoker = NY01.index(st.radio('Have you smoked at least 100 cigarettes in your entire life? (5 packs = 100 cigarettes)',NY01))
     stroke = NY01.index(st.radio('Have you had a stroke?',NY01))
     heartAttack = NY01.index(st.radio('Do you have coronary heart disease or myocardial infarction?',NY01))
     physAct = NY01.index(st.radio('Did you do any physical activity in past 30 days?',NY01))
     fruits = NY01.index(st.radio('Do you consume fruit 1 or more a day?',NY01))
     veggies = NY01.index(st.radio('Do you consume vegetable 1 or more a day?',NY01))
     hvyAlcohol = NY01.index(st.radio('Are you a heavy drinkers? (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)',NY01))
     genHlth = GENHEALTH.index(st.select_slider('How do you think about your general health?',GENHEALTH)) + 1
     mentlHlth = st.slider('Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?',0,30,help='Scale 1-30 days')
     physHlth = st.slider('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?',0,30,help='Scale 1-30 days')
     diffWalk = NY01.index(st.radio('Do you have serious difficulty walking or climbing stairs?',NY01))
     sex = SEX.index(st.radio('What is your gender?',SEX))
     age = AGE5YR.index(st.selectbox('How old are you?',AGE5YR))
     education = EDUCA.index(st.selectbox('What\'s your education level?',EDUCA)) + 1
     income = INCOME2.index(st.selectbox('How much your income in a year?',INCOME2)) + 1
     
     submitted = st.form_submit_button("Submit")

if submitted:
     with st.spinner('Your data is being processed please wait...'):
          bmi = round(weight/(height/100)**2)
          data = (highbp,highchol,cholcheck,bmi,smoker,stroke,heartAttack,physAct,fruits,veggies,hvyAlcohol,genHlth,mentlHlth,physHlth,diffWalk,sex,age,education,income)
          ans = predict(data)
     if ans == 0:
          st.success('We have detected that you do not have diabetes')
     elif ans == 1:
          st.warning('We have detected that you have pre-diabates or diabetes, please meet the doctor immediately')
