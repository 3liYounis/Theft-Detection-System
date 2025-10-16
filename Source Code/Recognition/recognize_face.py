import os
import json
import cv2 as cv
import numpy as np
from numpy import dot
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def match_face_to_database(face_embedding, portrait_database, threshold=0.6):
    best_match = None
    best_score = -1
    for name, portrait_embs in portrait_database.items():
        for portrait_emb in portrait_embs:
            if isinstance(portrait_emb, list):
                portrait_emb = np.array(portrait_emb)
            score = cosine_similarity(face_embedding, portrait_emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_score >= threshold:
            return best_match, best_score
    return None, None


def build_portrait_database(portrait_folder, app):
    try:
        with open('portrait_database.json', 'r') as file:
            database = json.load(file)
        if database:
            print("Loaded existing portrait database")
            return database
    except:
        print("No existing database found, building new one...")

    database = {}
    for dir_name in os.listdir(portrait_folder):
        person_folder = os.path.join(portrait_folder, dir_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing person: {dir_name}")
        for filename in os.listdir(person_folder):
            print(f"  Processing file: {filename}")
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(person_folder, filename)
            image = cv.imread(path)
            if image is None:
                print(f"    Failed to load image: {path}")
                continue

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            faces = app.get(image)
            if faces:
                embedding = faces[0].embedding
                if dir_name not in database:
                    database[dir_name] = []
                database[dir_name].append(embedding.tolist())
                print(f"    Added embedding for {dir_name}")
            else:
                print(f"    No face detected in {filename}")
    with open('portrait_database.json', 'w') as file:
        json.dump(database, file, indent=2)

    print(f"Database built with {len(database)} people:")
    for name, embeddings in database.items():
        print(f"  {name}: {len(embeddings)} embeddings")

    return database


app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir.endswith('Recognition'):
    faces_folder = "../../Data/Faces"
else:
    faces_folder = "../Data/Faces"

if not os.path.exists(faces_folder):
    alternative_paths = [
        "../../Data/Faces",
        "../Data/Faces",
        "./Data/Faces",
        "Data/Faces"
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            faces_folder = alt_path
            print(f"Using alternative path: {faces_folder}")
            break
    else:
        print(
            f"ERROR: Could not find Data/Faces directory. Tried: {alternative_paths}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Current file directory: {current_dir}")
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
            name, score = match_face_to_database(
                face.embedding, portrait_database)
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

                cv.rectangle(frame, (bbox[0], bbox[1]),
                             (bbox[2], bbox[3]), (0, 0, 0), 2)
                cv.putText(frame, label, (bbox[0], bbox[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv.imshow("Live Face Recognition", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    return max(results, key=lambda x: x['score'])
    cap.release()
    return max(results, key=lambda x: x['score'])


def get_face_info(dir_name):
    json_file_path = "./../Data/Face.json"
    with open(json_file_path, 'r') as file:
        faces = json.load(file)
    for face in faces:
        if face.get("name") == dir_name:
            return face
    return None
