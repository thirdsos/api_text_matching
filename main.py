from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json
from docx import Document
from TextCategory import TextPreprocessor, TextCategory, Tfidf
# from TextCategory import TextCategoryInstance, TextPreprocessor, TextCategory
from io import BytesIO
import numpy as np
import base64
import pickle

app = Sanic(__name__)
CORS(app)

# define model here
with open('idf.pkl', 'rb') as pkl:
    idf = pickle.load(pkl)
TextCategoryInstance = TextCategory(idf)

@app.route('/')
async def test(request):
    return json({'hello': 'world'})

@app.route('/txt', methods=['POST'])
async def txt_reader(request):
    json_body = request.json
    file = json_body['file']
    file = base64.b64decode(file)
    file = BytesIO(file).read().decode('utf-8')
    return json({'response': file})

@app.route('/doc', methods=['POST'])
@app.route('/docx', methods=['POST'])
async def docx_reader(request):
    json_body = request.json
    file = json_body['file']
    file = base64.b64decode(file)
    doc = Document(BytesIO(file))
    
    paragraphs = [p.text for p in doc.paragraphs]
    # res = {}
    # for i, paragraph in enumerate(paragraphs):
    #     res[f'p{i}'] = paragraph
    # corpus = Tfidf.word_preprocessing(paragraphs)
    p = TextPreprocessor()
    corpus = p.word_preprocessing(paragraphs)
    lda, candidate, topic = TextCategoryInstance.get_topics(corpus, 15)
    topics_unique = np.hstack([t.split(', ') for t in lda])
    topics_unique = np.unique(topics_unique)
    res = {'response': lda, 'unique': topics_unique.tolist(), 'candidate': candidate.tolist(), 'topic': topic, 'paragraphs': '\t'+ ('\n'.join(paragraphs))}
    return json(res)


host = '0.0.0.0' # local network
port = 8000
if __name__ == '__main__':
    try:
        app.run(host=host, port=port, auto_reload=False, workers=10)
    except KeyboardInterrupt:
        exit(1)
        # cmd command for kill all python process
        # "taskkill /f /im "python.exe" /t"