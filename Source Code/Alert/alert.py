import json
from datetime import datetime  
def alert(theft_link, item, face, time):    
        theftalert={
            "theft link":theft_link,
            "item":item,
            "face ID":face,
            "theft time": time
        }
        path=r"./Data/Alert.json"
        with open(path,"w") as json_file:json.dump(theftalert,json_file,indent=4)
pass
        
alert( 
        theft_link=r"D:\FinalProject\Final-BS.C-Project-2025\Data\Stream\stream.mp4",
        item=[
                {"Item_Name": "banana","Item Price":" 11$ " },
                {"Item_Name": "apple","Item Price":" 10$ "  }
        ],
        face=[r"D:\FinalProject\Final-BS.C-Project-2025\Data\Faces\Ali.png" , r"D:\FinalProject\Final-BS.C-Project-2025\Data\Faces\Mahdi.png" , r"D:\FinalProject\Final-BS.C-Project-2025\Data\Faces\Yazan.png"], 
        time=datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
)
#for the video, i think to upload the video to the youtube then put  the link of the video on the json then the owner of the store can open it when he resve the alert.