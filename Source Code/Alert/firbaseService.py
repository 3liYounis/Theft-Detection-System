import os
from firebase_admin import credentials, db
import firebase_admin

current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, "serviceAccountKey.json")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'SECRET',
})

alerts_db = db.reference('ALERTS')
