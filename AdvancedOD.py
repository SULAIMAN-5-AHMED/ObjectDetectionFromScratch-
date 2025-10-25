import cv2 as cv
from keras.applications import ResNet50, VGG19
from keras.applications.resnet50 import decode_predictions, preprocess_input as resnet_preporcess
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np


def get_features(img_path):

    img = cv.imread(img_path)
    base_model = VGG19(include_top=False,weights="imagenet",input_shape=(224,224,3))
    img = cv.resize(cv.cvtColor(img,cv.COLOR_BGR2RGB),(224,224))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    #['input_layer', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'].

    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv1').output)
    feature_map = feature_extractor.predict(img)
    return feature_map

def box_img(img_path,feature_map):
    img = cv.imread(img_path)
    img = cv.resize(img,(224,224))
    """
    since feature map in nothing more than an array we must covert it such that
    it can be processed using opencv so we use the procedure shown below
    """
    # Remove batch dimension and select one channel
    feature_map = feature_map[0, :, :, 0]  # Shape: (224, 224)

    # Normalize to 0–255
    feature_map = cv.normalize(feature_map, None, 0, 255, cv.NORM_MINMAX)
    feature_map = feature_map.astype('uint8')

    blur_img = cv.GaussianBlur(feature_map,(3,3),0)
    edges = cv.Canny(blur_img,threshold1=100,threshold2=200)
    countours,_ = cv.findContours(edges.copy(),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    area = []
    cropped_lst = []

    for countour in countours:

        x,y,w,h = cv.boundingRect(countour)

        a = w*h

        if a >=500:

            cropped_img = img[y:y+h, x:x+w]
            cropped_lst.append({
                "image": cropped_img,
                "bbox": (x, y, w, h)
            })
            area.append(a)
    return area, cropped_lst

def sort_lst(lst1, lst2):
    combined = sorted(zip(lst1, lst2), key=lambda x: x[0], reverse=True)
    ulst, ulst2 = zip(*combined)
    return list(ulst), list(ulst2)




def detection(cropped_lst,img_path):

    img = cv.imread(img_path)
    img = cv.resize(img,(224,224))
    model = ResNet50(include_top=True, weights='imagenet',input_shape=(224,224,3))
    for item in cropped_lst:
        imgs = item["image"]
        x, y, w, h = item["bbox"]

        imgs = cv.resize(imgs, (224, 224))
        imgs = cv.cvtColor(imgs, cv.COLOR_BGR2RGB)
        imgs = imgs.astype('float32')
        imgs = np.expand_dims(np.array(imgs), axis=0)
        imgs = resnet_preporcess(imgs)
        predict = model.predict(imgs)  # getting top label for better prediction
        label, desc, confidence = decode_predictions(predict, top=1)[0][0]

        cv.putText(img, desc, (x, y-4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyWindow()



feature = get_features("gettyimages-664442962-612x612.jpg")
area, cropped_lst = box_img("gettyimages-664442962-612x612.jpg",feature)
area, cropped_lst = sort_lst(area,cropped_lst)
detection(cropped_lst,"gettyimages-664442962-612x612.jpg")
