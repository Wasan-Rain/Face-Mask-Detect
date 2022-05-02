import numpy as np, cv2, dlib, pickle, os
import imutils
from sqlalchemy import null
import face_recognition
detector = dlib.get_frontal_face_detector()
sp =  dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.dat', 'rb'))
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_desc0 = model.compute_face_descriptor(img, shape, 0)
            d = []
            for face_desc in FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
            d = np.array(d)
            idx = np.argmin(d)
    
            if (d[idx]) <= 0.5:
                name = FACE_NAME[idx]
                print(d[idx])
                print(name)
                label = "{} : {:.2f}%".format(name,d[idx]*400)
                cv2.putText(frame, label, (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            else:
                print(d[idx])
                cv2.putText(frame, "Unknow", (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
#cv2.destroyAllWindows()