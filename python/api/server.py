from flask import Flask, request, render_template
from flask_cors import  CORS
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import *
import os, pickle, re, keras, sklearn, string
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
import io
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from vncorenlp import VnCoreNLP

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Xử lý tạo các hàm con khi mới vào host
@app.route('/')
def clean_doc(text, word_segment = True):
  text = text.lower()
  #xóa dấu câu
  for punc in string.punctuation:
    text = text.replace(punc,' ')
  #xóa khoảng trắng thừa
  text = re.sub('\s+',' ', text)
  text = text.strip()
  #tách từ
  if word_segment == True:
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])
  else:
    pass
  return text
def predict_class(text1, MAX_LEN = 40):
    test_encoded1 = t.texts_to_sequences([clean_doc(text1)])
    test_pad1 = sequence.pad_sequences(test_encoded1, maxlen=MAX_LEN)
    question1_class = np.argmax(model.predict(test_pad1))
    return question1_class
    
# flask routing
# Nhấn nút thì dữ liệu sẽ được truyền lên theo phương thức POST
@app.route('/api/get_answer', methods=["POST", "GET"])
def get_answer():
    try:
        question = request.form.get("question")
    except:
        question = request.args.get('question')
    print(question)
    ans_class = predict_class(question)
    print(ans_class)
    answer = ANSWERS[ans_class] 
    print(answer)      
    return  {
            "code" : 0,
            "answer" : answer
            }   

if __name__ == '__main__':
    
    path_vncorenlp = "../resource/vncorenlp/VnCoreNLP-1.1.1.jar"
    rdrsegmenter = VnCoreNLP(path_vncorenlp, annotators="wseg", max_heap_size='-Xmx500m') 
    df = pd.read_excel('../data/data.xlsx')
    path_ans = '../data/Answer.xlsx'
    ans_df = pd.read_excel(path_ans)
    QUESTIONS = df['Question'].astype(str)
    QUESTIONS = QUESTIONS.apply(clean_doc).tolist()
    ANSWERS = ans_df['Answer'].tolist()


    t = Tokenizer(oov_token='<UNK>')
    # fit the tokenizer on the documents
    t.fit_on_texts(QUESTIONS)
    t.word_index['<PAD>'] = 0
    path_model = '..\model\model_chatbot.h5'
    model = keras.models.load_model(path_model)
    # print(predict_class('tôi đi học'))
    app.run(host = '0.0.0.0', 
    port = 9999, 
    debug=False)
