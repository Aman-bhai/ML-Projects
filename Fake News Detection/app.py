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


pickled_model = pickle.load(open('fake_news_model.pkl', 'rb'))
dc = pickle.load(open('data\descisiontree.pkl', 'rb'))
mn = pickle.load(open('data\multinomialnb.pkl', 'rb'))
passi = pickle.load(open('data\passive.pkl', 'rb'))
log = pickle.load(open('data\log.pkl', 'rb'))
rp = pickle.load(open('data\\random_forest.pkl', 'rb'))
kn = pickle.load(open('data\KNCt.pkl', 'rb'))


cv=pickle.load(open('cvtransform.pkl','rb'))

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
        return "Fake"
    elif n==1:
        return "Not Fake"   
     
def manual(news):
    l=[]
    testing_news={'text':[news]}
    new=pd.DataFrame(testing_news)
    new['text']=new['text'].apply(wordopt)
    new_x=new['text']
    newxv=cv.transform(new_x)
    pred=pickled_model.predict(newxv)
    p=dc.predict(newxv)
    pr=mn.predict(newxv)
    pre=passi.predict(newxv)
    logi=log.predict(newxv)
    rf=rp.predict(newxv)
    predi=kn.predict(newxv)
    print("svm_prediction: {}\n dc_prediction:{}\n rf_prediction:{} \npass_prediction:{} \nlogis_prediction:{}\n knc_prediction:{}\n mn_prediction:{}".format(output(pred[0]),output(p[0]),output(rf[0]),output(pre[0]),output(logi[0]),output(predi[0]),output(pr[0])))
    l.append("svm_prediction: {}".format(output(pred[0])))
    l.append('dc_prediction:{}'.format(output(p[0])))
    l.append('rf_prediction:{}'.format(output(rf[0])))
    l.append('pass_prediction:{}'.format(output(pre[0])))
    l.append('logis_prediction:{}'.format(output(logi[0])))
    l.append('knc_prediction:{}'.format(output(predi[0])))
    l.append('mn_prediction:{}'.format(output(pr[0])))
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
     

