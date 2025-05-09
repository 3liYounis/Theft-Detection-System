import os
import json
import cv2 as cv
from numpy import dot
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def match_face_to_database(face_embedding, portrait_database, threshold=0.6):
    best_match = None
    best_score = -1
    for name, portrait_emb in portrait_database.items():
        score = cosine_similarity(face_embedding, portrait_emb)
        if score > best_score:
            best_score = score
            best_match = name
    if best_score >= threshold:
        return best_match, best_score
    return None, None

def build_portrait_database(portrait_folder, app):
    database = {}
    for filename in os.listdir(portrait_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(portrait_folder, filename)
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = app.get(image)
        if faces:
            embedding = faces[0].embedding
            database[filename] = embedding
    return database

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) 
faces_folder = "./Data/Faces"
portrait_database = build_portrait_database(faces_folder, app)

def recognize_face(theft_recording=None):
    if theft_recording:
        cap = cv.VideoCapture(theft_recording)
    else:
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {theft_recording}")
    
    results = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = app.get(rgb_frame)
        for face in faces:
            name, score = match_face_to_database(face.embedding, portrait_database)
            info = {
                'match': name if name else "Unknown",
                'score': float(score) if score else 0.0
            }
            results.append(info)
            if not theft_recording:
                bbox = face.bbox.astype(int)
                label = "Unknown"
                if name:
                    percentage = score * 100
                    label = f"{name} ({percentage:.2f}%)"

                cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 2)
                cv.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv.imshow("Live Face Recognition", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        return max(results, key=lambda x:x['score'])
    cap.release()
    return max(results, key=lambda x:x['score'])

def get_face_info(face_image):
    json_file_path = "./Data/Face.json"
    with open(json_file_path, 'r') as file:
        faces = json.load(file)
    for face in faces:
        if face.get("image") == face_image:
            return face
    return None