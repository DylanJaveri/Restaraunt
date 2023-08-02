import random
import json
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

model = tf.keras.models.load_model('chatbot_model.h5')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=["GET", "POST"])
def get_bot_response():
    message = request.args.get('msg')
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    return res

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_up_message(message):
    sentence_words = nltk.word_tokenize(message)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(message):
    sentence_words = clean_up_message(message)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if _name_ == '__main__':
    app.run(debug=True)