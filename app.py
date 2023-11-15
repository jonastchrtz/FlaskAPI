from flask import Flask, jsonify, request, Response
from User import User
from recommender_system import recommend_users
from firebase_admin import credentials, initialize_app, firestore

app = Flask(__name__)

cred = credentials.Certificate("firebase_admin.json")
firebase_app = initialize_app(cred)
db = firestore.client()


@app.route('/recommend/<int:user_id>')
def show_post(user_id):
    return f'Post {user_id}'

@app.route('/')
def test():
    return jsonify(recommend_users())

@app.route('/addUser', methods=['POST'])
def add_user():
    db.collection("users").add(request.json)
    return Response(status=200)