from ipaddress import AddressValueError
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


##create the training and the test dataset
training_path = 'day_night_images/training'
test_path = 'day_night_images/test'

def create_dataset(path)->dict:
    classes = os.listdir(training_path)
    
    address = {}
    for i in classes:
        address[i] = path+'/'+i


    return address

def standardize_image(image,resize=True,resize_val=(1100,600))->np.array:
    if resize:
        stand_img = cv2.resize(image,resize_val)
        return stand_img

    else:
        return image


def get_avg_brightness(path)->dict:
    images_list = os.listdir(path)
    total_images = len(images_list)
    avg_brightness = 0
    for i in images_list:
        image = cv2.imread(path+'/'+i)
        image = standardize_image(image,resize=True)
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        avg_brightness += np.average(hsv_image[:,:,2])

    return avg_brightness/ total_images


def train_classifier(data_path:str)->dict:
    address = create_dataset(data_path)
    classes = ['night','day']
    brightness_score = {}
    for i in classes:
        brightness_score[i] = get_avg_brightness(address[i])

    return brightness_score

#build an image brightness function which takes input as an image and returns brghtnes score

def image_brightness_score(image):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return np.average(hsv_image[:,:,2])


def get_accuracy(test_path,scores):
    address = create_dataset(test_path)
    classes = ['night','day']
    score = 0
    total_images = 0
    wrong_labels = []
    for i in classes:
        images_list = os.listdir(address[i])
        total_images += len(images_list)
        for j in images_list:
            img_path  = address[i]+'/'+j
            img = cv2.imread(img_path)
            std_img = standardize_image(img)
            label = test_classifier(std_img,scores)
            if label == i:
                score +=1

            else:
                score +=0
                wrong_labels.append(img_path)

    return score/total_images, wrong_labels


def test_classifier(image,scores):
    std_img = standardize_image(image,resize=True)
    brightnes_score = image_brightness_score(std_img)
    mean_score = (scores['day'] + scores['night'])/2
    if brightnes_score>mean_score:
        return 'day'

    else:
        return 'night'




if __name__ == '__main__':
    brightness_score = train_classifier(training_path)
    a,b = get_accuracy(test_path,brightness_score)
    print("Accuracy of the classifier:",a, "\nNumber of misclassified images", len(b))


