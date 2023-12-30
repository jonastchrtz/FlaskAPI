from flask import Flask, jsonify, request, Response
import requests

from recommend import find_most_similar_user

app = Flask(__name__)
firebase_url = "https://socialactivity-app-default-rtdb.europe-west1.firebasedatabase.app/"

@app.route('/recommend/<username>')
def show_post(username):
    most_similar_user = find_most_similar_user(username)
    return jsonify(most_similar_user)