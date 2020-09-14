from flask import render_template, url_for, request, jsonify
from app import app

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/query')
def query():
    return render_template('main.html')
