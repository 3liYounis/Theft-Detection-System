import json
from Alert.firbaseService import alerts_db


def alert(theft_link, face, time):
    if face:
        theftalert = {
            "theif_id": face.get('id', None) or None,
            "theif_name": face.get('name', None) or None,
            "theif_phone": face.get('phone', None) or None,
            "theif_age": face.get('age', None) or None,
            "theif_image": face.get('image', None) or None,
            "Recording": theft_link,
            "Date": time
        }
    else:
        theftalert = {
            "theif_id": None,
            "theif_name": None,
            "theif_phone": None,
            "theif_age": None,
            "theif_image": None,
            "Recording": theft_link,
            "Date": time
        }
    alerts_db.push(theftalert)
    path = "./../Data/Alert.json"
    with open(path, "w") as json_file:
        json.dump(theftalert, json_file, indent=4)
