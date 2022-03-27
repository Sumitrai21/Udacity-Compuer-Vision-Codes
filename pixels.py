import matplotlib
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 


#read in and display thi image
def change_binary(image,threshold)->np.array:
    if if_gray(image):
        mask = image >= threshold
        image[mask] = 255
        image[np.logical_not(mask)] = 0

        return image

    else:
        print("Not a grayscale image")
        return None

def if_gray(image):
    if len(image.shape) == 2:
        return True

    else: 
        return False


def get_pixel_value(image,pixel,display=False):
    y,x = pixel[1],pixel[0]
    if if_gray(image):
        
        pixel_value = image[y,x]
        return pixel_value

    else:
        r,g,b = rgb_colorspace(image, display=display)
        r_pixel_value = r[y,x]
        g_pixel_value = g[y,x]
        b_pixel_value = b[y,x]

        return (r_pixel_value,g_pixel_value,b_pixel_value)


def rgb_colorspace(image,display=True):
    if if_gray(image):
        print("The image is grayscale")
        return None

    else:
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]
        if display:
            f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,10))
            ax1.set_title("R Channel")
            ax1.imshow(r,cmap='gray')

            ax2.set_title("G Channel")
            ax2.imshow(g,cmap='gray')

            ax3.set_title("B Channel")
            ax3.imshow(b,cmap='gray')

            plt.show()

        return r,g,b




def read_image(filepath,grayscale=True,display=True):
    image = mpimg.imread(filepath)
    cmap = 'brg'
    if image.shape[0] >0:
        if grayscale:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            cmap = 'gray'

        if display:
            plt.imshow(image,cmap=cmap)
            plt.show()

        return image

    else:
        print("ERROR: No Image Found")



image = read_image("coffee.png",grayscale=False,display=False)
pixel_value = get_pixel_value(image,(300,300),display=False)
print(pixel_value)

            