import os
import tensorflow as tf
import numpy as np
import cv2
import pickle
import dlib
from flask import Flask, Response,render_template
app=Flask(__name__)
face_mask = ['Masked', 'No mask']
size = 224

# Load face detection and face mask model
path = r'face_detector/'
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
model1 = tf.keras.models.load_model('mask_detector.model')
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
FACE_DESC, FACE_NAME = pickle.load(open('trainset.dat', 'rb'))
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
sp =  dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(os.path.join(path, 'face_mask4.mp4'))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter(os.path.join(path, 'test4.avi'),
#                      cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.flip(frame,1)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            faceNet.setInput(blob)
            detections = faceNet.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < 0.5:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (size, size))
                face = np.reshape(face, (1, size, size, 3)) / 255.0
                result = np.argmax(model1.predict(face))

                if result == 0:
                    labels = face_mask[result] + ": "
                    color = (0, 255, 0)
                    cv2.putText(frame, labels, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
                else:
                    labels = face_mask[result] + ": "
                    color = (0, 0, 255)
                    cv2.putText(frame, labels, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)

                img = frame[startY-10:startY+h+10, startX-10:startX+w+10][:,:,::-1]
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc0 = model.compute_face_descriptor(img, shape, 0)
                    d = []
                    for face_desc in FACE_DESC:
                        d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
                    d = np.array(d)
                    idx = np.argmin(d)
                
                    if (d[idx]) <= 0.4:
                        name = FACE_NAME[idx]
                        print(d[idx])
                        print(name)
                        label = "{} : {:.2f}%".format(name,d[idx]*250)
                        cv2.putText(frame, label, (startX+120,startY-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                    else:
                        print(d[idx])
                        cv2.putText(frame, "Unknow", (startX+120,startY-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

        #cv2.putText(frame, labels, (startX, startY + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
        #cv2.rectangle(frame, (startX, startY - 60), (endX, startY), (0, 0, 255), -1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #if ret == True:
    #    out.write(frame)
    #else:
    #    pass

    #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Video', 800, 600)
    #cv2.imshow('Video', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

#cap.release()
#out.release()
#cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)