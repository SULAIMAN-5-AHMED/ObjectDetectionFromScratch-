import cv2 as cv
import numpy as np
from keras._tf_keras.keras.applications import ResNet50
from keras.applications.resnet50 import decode_predictions


def box_img(img_path):
    org_img = cv.imread(img_path)
    img = org_img.copy()
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img,(3,3),0) # 0 to auto-compute standard devation in x direction and sigmaY is set to sigmaX
    edges = cv.Canny(blur_img,threshold1=52,threshold2=130) #threshold1 = 0.4 * threshold2
    # tresholds here takes the sensitivity of edges into considration
    countours,_ = cv.findContours(edges.copy(),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cropped_img = []
    for countour in countours:
        x,y,w,h = cv.boundingRect(countour)
        cv.rectangle(org_img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped = org_img[y:y+h,x:x+w]
        cropped_img.append(cropped)

    return cropped_img, countours

def detect_img(org_img,cropped_img,countours):
    model = ResNet50(include_top=True,weights="imagenet")
    for i in range(len(cropped_img)):
       img = cropped_img[i]
       countour = countours[i]

       if img is None:
           continue

       x,y,w,h = cv.boundingRect(countour)
       img = cv.resize(cv.cvtColor(img,cv.COLOR_BGR2RGB),(224,224))

       img = img.astype('float32')
       img = np.expand_dims(np.array(img),axis=0)
       predict = model.predict(img) # getting top label for better prediction
       label, desc, confidence = decode_predictions(predict, top=1)[0][0]
       if confidence >= 0.10:
           cv.rectangle(org_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
           cv.putText(org_img, desc, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imshow("img",org_img)
    cv.waitKey(0)
    cv.destroyWindow()
cropped_img, countours = box_img("istockphoto-1779735855-612x612.jpg")
org_img = cv.imread("istockphoto-1779735855-612x612.jpg")

detect_img(org_img, cropped_img, countours)