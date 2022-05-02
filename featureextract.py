import numpy as np, cv2, dlib, pickle, os
path = 'D:/facedata/' #เก็บรูปใบหน้าที่จำมาทำ face recognition หรือการจดจำใบหน้า
detector = dlib.get_frontal_face_detector() #เรียกใช้ฟังก์ชั่นตรวจจับใบหน้าของ library dlib
sp =  dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #เรียกใช้โมเดลตรวจจับใบหน้า 68 จุด
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') #เรียกใช้โมเดลในการจดจำใบหน้า
FACE_DESC = [] 
FACE_NAME = []
for fn in os.listdir(path):  
    img = cv2.imread(path + fn) #รูปภาพทั้งหมดที่จะจดจำ
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #แปลงสี BGR เป็น RGB
        #print(fn)
    dets = detector(img, 1) #ใช้ฟังก์ชั่นตรวจจับใบหน้า
    for d in dets: #วนลูปใบหน้าทั้งหมดที่ตรวจจับได้
        shape = sp(img, d) #กำหนด shape ของใบหน้าทั้ง 68 จุด
        #print(shape)
        face_desc = model.compute_face_descriptor(img, shape, 1) #ประมวลผลรายละเอียดของใบหน้า
        FACE_DESC.append(face_desc) #เพิ่มรายละเอียดของในหน้าไว้ในตัวแปร FACE_DESC
        print('loading...', fn)
        FACE_NAME.append(fn[:fn.index('_')]) #เพิ่มชื่อรูปใบหน้าไว้ในตัวแปร FACE_NAME โดยเอาเฉพาะชื่อ
        #print(fn)
print(len(FACE_DESC))
#print(FACE_DESC)
pickle.dump((FACE_DESC, FACE_NAME), open('trainset.dat', 'wb')) #สร้างโมเดลสำหรับการทำ test face recognition
