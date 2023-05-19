from flask import Flask,request,render_template,session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from datetime import datetime
import json
import numpy as np
import pickle
import string
import nltk
nltk.data.path.append('"E:\\nltk_data\packages\corpora\stopwords.zip"')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as s
import re
nltk.download('wordnet')

wn=WordNetLemmatizer()
stop_words=s.words('english')

def clean_text(text):
  remov_pun=''.join([c.lower() for c in text if c not in string.punctuation])
  tokens=re.split('\W+',remov_pun)
  text=[ wn.lemmatize(word) for word in tokens if word not in stop_words]
  return text






with open('config.json','r') as c:
    params=json.load(c)["params"]




app = Flask(__name__)

cv=pickle.load(open('count.pkl','rb'))
knn = pickle.load(open('KNN.pkl', 'rb'))
dc = pickle.load(open('decision.pkl', 'rb'))
mn = pickle.load(open('multinomialnb.pkl', 'rb'))
passi = pickle.load(open('passiveaggresive.pkl', 'rb'))
rp = pickle.load(open('randomforestclassifier.pkl', 'rb'))
knr = pickle.load(open('KNR.pkl', 'rb'))
lr=pickle.load(open('linearRegression.pkl','rb'))
lor=pickle.load(open('logisticRegression.pkl','rb'))
svm=pickle.load(open('SVM.pkl','rb'))
svm1=pickle.load(open('SVM1.pkl','rb'))
svm2=pickle.load(open('SVM2.pkl','rb'))
svmC=pickle.load(open('SVMc.pkl','rb'))



@app.route('/')
def first():
    return render_template('first.html')


def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text

def output(n):
    if n==0:
        return "Neutral"
    elif n==1:
        return "Positive" 
    else:
        return 'Negative'  
     
def manual(news):
    l=[]
    testing_news={'text':[news]}
    new=pd.DataFrame(testing_news)
    new['text']=new['text'].apply(wordopt)
    new_x=new['text']
    newxv=cv.transform(new_x)
    pred=knn.predict(newxv)
    p=dc.predict(newxv)
    pr=mn.predict(newxv)
    pre=passi.predict(newxv)
    logi=lor.predict(newxv)
    rf=rp.predict(newxv)
    predi=knr.predict(newxv)
    sv=svm.predict(newxv)
    sm=svm1.predict(newxv)
    svmm=svm2.predict(newxv)
    svvm=svmC.predict(newxv)
    print("knn_prediction: {}\n dc_prediction:{}\n rf_prediction:{} \npass_prediction:{} \nlogis_prediction:{}\n knc_prediction:{}\n mn_prediction:{}\nsvmc_prediction:{}\nsvm_prediction:{}\nsvm2_prediction:{}\nsvm1_prediction:{}".format(output(pred),output(p),output(rf),output(pre),output(logi),output(predi),output(pr),output(svvm),output(svmm),output(sv),output(sm)))
    l.append("knn_prediction: {}".format(output(pred)))
    l.append('dc_prediction:{}'.format(output(p)))
    l.append('rf_prediction:{}'.format(output(rf)))
    l.append('pass_prediction:{}'.format(output(pre)))
    l.append('logis_prediction:{}'.format(output(logi)))
    l.append('knc_prediction:{}'.format(output(predi)))
    l.append('mn_prediction:{}'.format(output(pr)))
    l.append('svm1_prediction:{}'.format(output(sm)))
    l.append('svm2_prediction:{}'.format(output(sv)))
    l.append('svm_prediction:{}'.format(output(svmm)))
    l.append('svmc_prediction:{}'.format(output(svvm)))
    return l
    


@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form.get('message')
		abc=manual(message)
         
	return render_template('layout.html',prediction=abc)
    
    
@app.route('/ogin')
def home():
    return render_template('signup page.html')


@app.route('/login',methods=['GET','POST'])
def login():
    username=request.form.get('uname')
    email=request.form.get('uemail')
    password=request.form.get('pass')
    if (username==params['admin_user'] and  password==params['admin_password'] and email==params['admin_email']):
        return render_template('index.html',name=username)
    else:
        return f'you enter wrong password'

app.run(debug=True)
     

