import time, cv2, numpy as np
from picamera2 import Picamera2

MODEL, IMGSZ, CONF, IOU = 'models/best.onnx', 320, 0.4, 0.5
W, H = 640, 480

net = cv2.dnn.readNetFromONNX(MODEL)
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={'format': 'RGB888', 'size': (W, H)}))
cam.start()

while True:
    t0 = time.time()
    frame = cam.capture_array()                       # BGR order, same as OpenCV
    net.setInput(cv2.dnn.blobFromImage(frame, 1/255, (IMGSZ, IMGSZ), swapRB=True))
    out = net.forward()[0].T                          # (N, 5): cx, cy, w, h, score
    out = out[out[:, 4] > CONF]
    n = 0
    if len(out):
        sx, sy = W / IMGSZ, H / IMGSZ
        boxes = np.column_stack([(out[:, 0] - out[:, 2] / 2) * sx, (out[:, 1] - out[:, 3] / 2) * sy,
                                 out[:, 2] * sx, out[:, 3] * sy])
        n = len(cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), CONF, IOU))
    print(f'{n} apples | {1 / (time.time() - t0):.1f} FPS')