# import time
# time.clock = time.time
import shutil
import os
from flask import *
from config import upload_dir
from chroma_handler import process_pdf
from chatbot_as_function import chatbot

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("uploader.html")


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist("file")
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        os.makedirs(upload_dir)
        for f in files:
            f.save(os.sep.join([upload_dir, f.filename]))
        for f in files:
            process_pdf(os.sep.join([upload_dir, f.filename]))
        # return "<h5>File upload successful!</h5>"
        return render_template("chat.html")
    
@app.route('/query', methods=['GET'])
def chat():
    if request.method == 'GET':
        query = request.args['question']
        return chatbot(query=query)


if __name__ == '__main__':
    app.run()
