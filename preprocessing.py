
# Preprocessing converts images to greyscale, and adjusts the brightness if necessary (e.g. if average brightness is less than 0.4, increase brightness; if average brightness is greater than 0.6, reduce brightness)
# It additionally resizes the image to TWO different sizes: 200*200 and 50*50 and saves them to be used later on 
from PIL import Image

img = Image.open('test.jpg')
imgGray = img.convert('L')
imgGray.save('test_gray.jpg')