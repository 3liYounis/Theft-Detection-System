import json


def alert(theft_link, item, face, time):
    theftalert = {
        "theft link": theft_link,
        "item": item,
        "face ID": face,
        "theft time": time
    }
    path = r"./Data/Alert.json"
    with open(path, "w") as json_file:
        json.dump(theftalert, json_file, indent=4)
