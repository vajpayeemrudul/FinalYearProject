import nltk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from flask import send_file, redirect, url_for


from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

SYMPTOMS_GLOBAL = []
COLLECTED = False
initial_ques = [
    "Please tell me your name.",
    "What is your age?",
    "Recorded blood sugar levels? Enter 'N' if not recorded yet.",
    "Recorded blood pressure levels? Enter 'N' if not recorded yet.",
]
initial_ques_res = {item:[False, ""] for item  in initial_ques}
# initial_ques_ans = [False]*4



def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.05
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    print("tag: " ,tag) # tag is a string
    # print(ints)
    # print()
    list_of_intents = intents_json['intents']
    result = None
    checked_resp = ""
    for i in list_of_intents:
        # if 
        # if(i['tag']== tag):
        #     check = i.get("check", None)
        #     if check:
        #         if check.lower() in tag.lower(): # assuming tag is a string
        #             print("this is in checked reponse")
        #             checked_resp = random.choice(i['responses'])
        #     else:
        #         result = random.choice(i['responses'])
        #     print(result, tag)
        #     print()
        if (i['tag']== tag):
            result = random.choice(i['responses'])
    if checked_resp:
        return checked_resp
    if result is None:
        return "I did not understand your question."
    return result



def chatbot_response(msg):
    # COLLECTED = False
    if "COLLECTED" not in locals().keys():
        COLLECTED = False
    
    # if "initial_ques" not in locals().keys():
    #     initial_ques = [
    #         "Please tell me your name.",
    #         "What is your age?",
    #         "Recorded blood sugar levels? Enter 'N' if not recorded yet.",
    #         "Recorded blood pressure levels? Enter 'N' if not recorded yet.",
    #     ]
    #     initial_ques_ans = [False]*4
    
    # if not COLLECTED:


    if "prevent this" in msg:
        return "Please specify the name of the disease"

    tags_not_symptoms = ["greeting", "goodbye"]
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    tag = ints[0]['intent']

    if (tag in tags_not_symptoms or "prevention" in tag) and not COLLECTED:
        return res
    
    
    sym_complete = ["N", "no"]
    if msg.lower() in [item.lower() for item in sym_complete]:
        COLLECTED=True
        msg_concat = ", ".join(SYMPTOMS_GLOBAL)
        ints = predict_class(msg_concat, model)
        res = getResponse(ints, intents)
    else:
        if len(SYMPTOMS_GLOBAL)==0 and not COLLECTED:
            res = "How are you feeling? Please tell us your symptoms."
            SYMPTOMS_GLOBAL.append(msg)
        else:
            # SYMPTOMS_GLOBAL.append(msg)
            if not COLLECTED:
                SYMPTOMS_GLOBAL.append(msg)
                res = "Any other symptoms? Enter No/N if you have entered all symptoms. "
            else:
                return res

    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    for index, item in enumerate(initial_ques_res.items()):
        # print(initial_ques_res) 
        k,v = item
        if v[1] =="" and v[0] is False:
            initial_ques_res[k] = [True, ""]
            return k
        elif v[1] =="" and v[0] is True:
            initial_ques_res[k] = [True, userText]
        if index==3 and initial_ques_res[k][0] == True and initial_ques_res[k][0] != "":
            print(initial_ques_res)
            return chatbot_response(userText)
    
    print(initial_ques_res)
    # if init_check!= True:
    #     return chatbot_response(userText)
    
    return chatbot_response(userText)

# @app.route("/post")
@app.route('/download', methods=['GET', 'POST']) 
def download_report():
    print("1111111111111111111111111111111111111111111111111111111111111111111")
    data = initial_ques_res
    data = {k:[v[1]] for k,v in data.items()}
    df = pd.DataFrame(data)
    df = df.T
    fig, ax =plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    table = pd.plotting.table(ax, df)
    pp = PdfPages("report.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    return send_file('./report.pdf', as_attachment=True)
    # return redirect(url_for('download_report'))



if __name__ == "__main__":
    app.debug = True
    app.run()