
# Preprocessing converts images to greyscale, and adjusts the brightness if necessary (e.g. if average brightness is less than 0.4, increase brightness; if average brightness is greater than 0.6, reduce brightness)
# It additionally resizes the image to TWO different sizes: 200*200 and 50*50 and saves them to be used later on 
from PIL import Image, ImageStat, ImageEnhance
import os
from os import listdir
import cv2
from keras.preprocessing import image as krimage
from sklearn.model_selection import GridSearchCV, train_test_split

# get the path directory for the training images
trainingDir = "Train"

# processed directories
processedTrain = "processediMAGE"
bedroomDir = "BedroomProcessed"
coastDir = "CoastProcessed"
forestDir = "ForestProcessed"
highwayDir = "HighwayProcessed"

def preprocess():
    # check if directory already exists, if not then create it
    alreadyProcessed = makeProcessedDirectories()

    if alreadyProcessed:
        return

    for subdir, dirs, files in os.walk(trainingDir):
        for image in files:
            if (image.endswith(".jpg")): 
                img = Image.open(subdir + '/' + image)

                # convert to greyscale
                imgGray = img.convert('L')

                # declare enhancer for brightness
                enhancer = ImageEnhance.Brightness(imgGray)

                # get average pixel brightness
                stat = ImageStat.Stat(imgGray)
                mean = stat.mean[0]

                # if average brightness of image is greater than 100 then darken the image
                if mean > 130:
                    factor = 0.2
                    imgGray = enhancer.enhance(factor)
                
                #  if average brightness of image is less than 50 then brighten the image
                if mean < 50:
                    factor = 1.5
                    imgGray = enhancer.enhance(factor)

                largerResize = (300, 300)
                # resize image to 200*200
                largerImage = imgGray.resize(largerResize)

                smallerResize = (50,  50)
                # resize image to 50*50
                smallerImage = imgGray.resize(smallerResize)

                if "bedroom" in subdir:
                    smallerImage.save(bedroomDir + '/smaller' + image, 'JPEG')
                    largerImage.save(bedroomDir + '/larger' + image, 'JPEG')

                if "Coast" in subdir:
                    smallerImage.save(coastDir + '/smaller' + image, 'JPEG')
                    largerImage.save(coastDir + '/larger' + image, 'JPEG')
                
                if "Forest" in subdir:
                    smallerImage.save(forestDir + '/smaller' + image, 'JPEG')
                    largerImage.save(forestDir + '/larger' + image, 'JPEG')

                if "Highway" in subdir:
                    smallerImage.save(highwayDir + '/smaller' + image, 'JPEG')
                    largerImage.save(highwayDir + '/larger' + image, 'JPEG')
    return

# check if directories exist, if not then the images need to be
# pre-processed
def makeProcessedDirectories():
    if not os.path.exists(bedroomDir):
        os.makedirs(bedroomDir)
        print("created folder: ", bedroomDir)
    else:
        return True

    if not os.path.exists(coastDir):
        os.makedirs(coastDir)
        print("created folder: ", coastDir)
    else:
        return True

    if not os.path.exists(forestDir):
        os.makedirs(forestDir)
        print("created folder: ", forestDir)
    else:
        return True

    if not os.path.exists(highwayDir):
        os.makedirs(highwayDir)
        print("created folder: ", highwayDir)
    else:
        return True

    return False

# extractSIFTFeatures (SIFT stands for Scale Invariant Feature Transform)
# pulls out the features from each image content into local feature coordinates
# that are invariant to translation, scale, and other image transformations
def extractSIFTFeatures():
    bedroomsift = {}
    coastsift = {}
    forestsift = {}
    highwaysift = {}
    trainedImages = []

    # go through all the pre-processed training images and save the data from sift feature extractors
    for image in os.listdir(bedroomDir):
        if (image.endswith(".jpg")): 
            img = cv2.imread(bedroomDir + '/' + image)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            bedroomsift[image] = keypoints, descriptors
            img_arr = krimage.img_to_array(img)
            img_arr = img_arr / 255
            trainedImages.append(img_arr)

    # go through all the pre-processed training images
    for image in os.listdir(coastDir):
        if (image.endswith(".jpg")): 
            img = cv2.imread(coastDir + '/' + image)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            coastsift[image] = keypoints, descriptors
            img_arr = krimage.img_to_array(img)
            img_arr = img_arr / 255
            trainedImages.append(img_arr)

    # go through all the pre-processed training images
    for image in os.listdir(forestDir):
        if (image.endswith(".jpg")): 
            img = cv2.imread(forestDir + '/' + image)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            forestsift[image] = keypoints, descriptors
            img_arr = krimage.img_to_array(img)
            img_arr = img_arr / 255
            trainedImages.append(img_arr)

    # go through all the pre-processed training images
    for image in os.listdir(highwayDir):
        if (image.endswith(".jpg")): 
            img = cv2.imread(highwayDir + '/' + image)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            highwaysift[image] = keypoints, descriptors
            img_arr = krimage.img_to_array(img)
            img_arr = img_arr / 255
            trainedImages.append(img_arr)
    
    timages = np.array(trainedImages)
    
    return timages

def nearestNeighbor():

    return

if __name__ == "__main__":
    # preprocess all the training images
    preprocess()

    # extract the SIFT features on all of the training images
    images = extractSIFTFeatures()

    # represent the image directly using the 50*50 (2500) pixel values and use the Nearest Neighbor classifier
    nearestNeighbor()            