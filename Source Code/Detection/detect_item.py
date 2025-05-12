import cv2, time, json, yaml, math
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL   = Path(r"runs/detect/shop15_det10/weights/best.onnx")
DATA    = Path(r"datasets/products_ds/data.yaml")
CODES   = Path(r"datasets/products_ds/barcodes-price.json")
CAM     = 3
SIZE    = 640
THR     = 0.80

# --- load meta ---
names = yaml.safe_load(DATA.read_text())["names"]
codes = json.loads(CODES.read_text())      # str-key dict

# --- ORT session ---
sess = ort.InferenceSession(
    MODEL.as_posix(),
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)
print("Providers:", sess.get_providers())

# --- helper: integer-precise letterbox ---
def letterbox(im, new=640, color=(114,114,114)):
    h0, w0 = im.shape[:2]
    # scale ratio
    r = min(new / h0, new / w0)
    unpad_w, unpad_h = int(w0 * r), int(h0 * r)

    # total padding
    dw, dh = new - unpad_w, new - unpad_h
    pad_x, pad_y = dw // 2, dh // 2

    # resize and pad
    im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(
        im,
        pad_y, dh - pad_y,
        pad_x, dw - pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return im, r, pad_x, pad_y

# --- main detection function --------------------------------------
def detect_products(frame):
    """Return list of dicts: name, barcode, price, conf, bbox."""
    # 1) letterbox + blob
    img, r, pad_x, pad_y = letterbox(frame, SIZE)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255

    # 2) run ONNX
    outs = sess.run(None, {"images": blob})
    raw = outs[0]  # e.g. shape (1,N,5+) or (1,5+,N)

    # 3) normalize to (N, C)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim == 2 and raw.shape[0] in (5, 6, 7):
        raw = raw.T
    dets = raw  # now shape is (N, C>=5)

    out = []
    for row in dets:
        # unpack safely
        if len(row) == 5:
            x1, y1, x2, y2, conf = row
            cls = 0
        else:
            x1, y1, x2, y2, conf, cls = row[:6]

        if conf < THR:
            continue

        cls = int(cls)
        # undo integer padding & scaling
        x1 = (x1 - pad_x) / r;  y1 = (y1 - pad_y) / r
        x2 = (x2 - pad_x) / r;  y2 = (y2 - pad_y) / r
        box = tuple(map(int, (x1, y1, x2, y2)))

        info = codes[str(cls)]
        out.append({
            "name":    info["name"],
            "barcode": info["barcode"],
            "price":   info["price"],
            "conf":    float(conf),
            "bbox":    box
        })

    # 4) Non-Maximum Suppression
    boxes  = [d['bbox'] for d in out]    # [(x1,y1,x2,y2), …]
    scores = [d['conf']   for d in out]  # [0.87, 0.45, …]
    rects  = [[x1, y1, x2 - x1, y2 - y1] for x1,y1,x2,y2 in boxes]
    idxs   = cv2.dnn.NMSBoxes(rects, scores, THR, 0.45)

    keep = []
    if idxs is not None and len(idxs) > 0:
        if isinstance(idxs, np.ndarray):
            keep = idxs.flatten().tolist()
        else:
            for i in idxs:
                keep.append(i[0] if isinstance(i, (list,tuple, np.ndarray)) else i)

    filtered = [out[i] for i in keep] if keep else out
    return filtered

# --- live inference loop -------------------------------------------
def detect_item(theft_recording):
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    print("Press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        dets = detect_products(frame)

        # draw
        for det in dets:
            x1,y1,x2,y2 = det["bbox"]
            label = f"{det['name']} {det['conf']*100:.1f}%  ₪{det['price']}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        fps = 1 / (time.time() - t0)
        cv2.putText(frame, f"{fps:.1f} FPS", (20,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Shop live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_item(None)
