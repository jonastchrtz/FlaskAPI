from flask import Flask, jsonify, request, Response
import requests

app = Flask(__name__)
firebase_url = "https://socialactivity-app-default-rtdb.europe-west1.firebasedatabase.app/"

@app.route('/recommend/<int:user_id>')
def show_post(user_id):
    return f'Post {user_id}'

@app.route('/')
def get_all_users():
    return requests.get(firebase_url + "users.json").json()