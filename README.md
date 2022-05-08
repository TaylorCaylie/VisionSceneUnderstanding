# VisionSceneUnderstanding

This project combines image processing, feature extraction, clustering and classification methods to achieve basic scene understanding. 

Scene recognition is implemented with small-scale images and nearest neighbor classification -- and then further uses quantized local features and linear classifiers learned by support vector machines 


Firstly the images are preprocessed using the Python Imaging Library (PIL) to obtain average brightness for each image
and adjust accordingly to prep the image, the image is additionally converted to grey scale and each image is resized
to two different sizes - one is 200*200 while the other is 
50*50
