import json


def alert(theft_link, face, time):
    theftalert = {
        "Theft Link": theft_link,
        "Face Info": face,
        "Theft Time": time
    }
    path = "./../Data/Alert.json"
    with open(path, "w") as json_file:
        json.dump(theftalert, json_file, indent=4)
