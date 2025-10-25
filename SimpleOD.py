import cv2 as cv
import matplotlib.pyplot as plt


"""
    We create a few functions 
    -Load image(img_path) -return img, gray_img
    -processing_img(gray_img) -return edges
    -find_obj(edges) -return countours
    -draw_bounding_box(img, countours) -return img
    -detect_objects(img_path) return boxed_image
    -show_img(img,title) plot image
"""


def load_imd(img_path):
    img = cv.imread(img_path)

    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return  img, gray_img

def processing_img(gray_img):
    #Smooths the image and reduce noise
    blurred_img = cv.GaussianBlur(gray_img,(3,3),0) # 0 to auto-compute standard devation in x direction and sigmaY is set to sigmaX
    #detects edges from the smooth image
    edges = cv.Canny(blurred_img,threshold1=40,threshold2=100)
    plt.imshow(blurred_img)
    plt.title("Blurred img")
    plt.show()
    plt.imshow(edges)
    plt.title("Edges")
    plt.show()
    return edges

def find_obj(edges):

    countours,_ = cv.findContours(edges.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    """edges.copy() is done to make sure that original edges are not affected
    moreover the other parameters ensure that edges are detected"""

    return countours

def draw_bounding_box(img,countours):

    for countour in countours:
        x,y,w,h = cv.boundingRect(countour)
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def detect_objects(img_path):
    img,gray = load_imd(img_path)
    edges = processing_img(gray)
    countours = find_obj(edges)
    boxed_img = draw_bounding_box(img.copy(), countours)
    return  boxed_img

def show_img(img,title="Detected Img"):
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.show()

result = detect_objects("istockphoto-1779735855-612x612.jpg")
show_img(result)