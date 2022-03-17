import requests
import base64
from docx import Document
from io import BytesIO
with open('test-docx.docx', 'rb') as f:
    file = f.read()
    data = {'file': base64.b64encode(file).decode()}
url = 'http://127.0.0.1:8000/'
res = requests.post(url, json=data)
print(res.json())