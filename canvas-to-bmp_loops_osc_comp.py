# Extacts mask of certain color from an image, calculates coefficients for the masks digital compressibility, and sends those to SuperCollider via Open Sound Control
# basic implementation at https://www.geeksforgeeks.org/color-identification-in-images-using-python-opencv/
# In case it's useful, you can encode an image with opencv without saving it with cv2.imencode('jpg', img)

# Importing the libraries OpenCV and numpy
import time
import sys
import cv2
import numpy as np
from pythonosc import udp_client


# Get images from webcam
camera = cv2.VideoCapture(0)

for j in range(10):
    # Set number of iterations
    it = 2
    img, image, hsv = [], [], []
    for i in range(it):
        return_value, im = camera.read()
        img.append(im)
        # cv2.imwrite('/Users/andreas/Desktop/thesis playgrund/practical VI/capture'+str(i)+'.png', image)
        i += 1
        cv2.waitKey(500)
    # del(camera)

    
    # Normalize and get HSV verisons of the images
    for i in range(it):
        # Resizing the image
        image.append(cv2.resize(img[i], (400, 400)))
        # Convert Image to Image HSV
        hsv.append(cv2.cvtColor(image[i], cv2.COLOR_BGR2HSV))

    # Set the desired RGB color
    color = np.uint8([[[195, 153, 50]]])
    # Convert RGB color to HSV color
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    ## First mask
    # Get upper and lower HSV range values
    col_dev = 30 # tolerance for deviation from given color; higher number --> higher tolerance
    hsv_color_lower_range_val, hsv_color_upper_range_val= hsv_color[0][0][0] - col_dev, hsv_color[0][0][0] + col_dev
    # Fill range values into HSV arrays
    lower = np.array([hsv_color_lower_range_val, 100, 100])
    upper = np.array([hsv_color_upper_range_val, 255, 255])

    # Defining mask for detecting color
    masked = []
    for i in range(it):
        masked.append(cv2.inRange(hsv[i], lower, upper))
    # print("masked ", masked[0][0])


    # Denoise both images
    denoise = cv2.fastNlMeansDenoising(masked[0], None, h = 50, templateWindowSize = 7, searchWindowSize = 10)
    denoise1 = cv2.fastNlMeansDenoising(masked[1], None, h = 50, templateWindowSize = 7, searchWindowSize = 10)

    ## Set high threshold for greyscale intensities
    # Threshold value
    threshold = 200
    # Perform thresholding using NumPy operations
    denoise1[denoise1 >= threshold] = 255
    denoise1[denoise1 < threshold] = 0
    # Perform thresholding using NumPy operations
    denoise[denoise >= threshold] = 255
    denoise[denoise < threshold] = 0

    ## Caluclate difference between two images (as np arrays)
    im0 = masked[0] # later image at t1
    im1 = masked[1] # later image at t2
    # im_diff = im1 - im0
    denoise_diff = denoise1 - denoise
    denoise_diff_diff = cv2.fastNlMeansDenoising(denoise_diff, None, h = 50, templateWindowSize = 7, searchWindowSize = 10)
    val, png_diff = cv2.imencode(".png", denoise_diff)
    coef_diff = sys.getsizeof(png_diff)

    # Encode the difference as png, get byte size of the encoded object
    # val, im_diff_png = cv2.imencode(".png", im_diff)
    # coef_diff = sys.getsizeof(im_diff_png)

    # Encode each mask as png, calculate difference in byte size
    val, png_im0 = cv2.imencode(".png", denoise)
    val, png_im1 = cv2.imencode(".png", denoise1)
    coef = sys.getsizeof(png_im1) - sys.getsizeof(png_im0)
    
    print(coef, " Each image encoded separately")
    print(coef_diff, " From the difference")

    # Repeat Thresholding
        ## Set high threshold for greyscale intensities
    # Threshold value
    threshold = 200
    # Perform thresholding using NumPy operations
    denoise1[denoise1 >= threshold] = 255
    denoise1[denoise1 < threshold] = 0
    # Perform thresholding using NumPy operations
    denoise[denoise >= threshold] = 255
    denoise[denoise < threshold] = 0

    cv2.imwrite('/Users/andreas/Desktop/thesis playgrund/practical VIII/webcam_gouache/Test_mask_diff'+str(j)+".png", denoise_diff_diff)
    cv2.imwrite('/Users/andreas/Desktop/thesis playgrund/practical VIII/webcam_gouache/Test_nodiff_'+str(j)+".png", denoise1)
    # dif = ImageChops.difference(im1, im2)
    # np.mean(np.array(dif))

    # # Calculates absolute difference between images
    # coef = np.mean(np.abs(im1 - im2))
    # if coef < 0.1:
    #     coef = 0.1

    # Send number(s) to SC, add some form of transformation to match SC Synth
    osc_client = udp_client.SimpleUDPClient(address='127.0.0.1', port=57120)
    osc_client.send_message("/my_path", (np.log(coef*10), (coef*10)**2)) # for droplets synth, still crashes a lot
    # osc_client.send_message("/my_path", (coef*500, (coef*10)**2)) # for pulse synth
    # osc_client.send_message("/my_path", abs(coef*10)) # for pulse based on separate encoding








