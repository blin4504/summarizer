from flask import Flask, request, render_template
from summarizer import summarize

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('./index.html', title='Welcome', username='Brian')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    return summarize(text)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001, debug=True)
