
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore


def send_data(data):
    cred = credentials.Certificate("firebase_admin.json")
    firebase_admin.initialize_app(cred)
    #firebase_admin.initialize_app(cred, { "databaseURL": "https://dynamic-return-342300-default-rtdb.firebaseio.com/"})
    db = firestore.client()
    db.collection('test').add(data)
    print('sent data')

