The scanpath of a subject on an image is organized as a separate entry in the JSON files for the training and the testing sets.

The data in the JSON files is in the following format:

Name: Image name
Cluster: Layout cluster to which this webpage belongs
Split: used for 'train' or 'test' in the paper
X: list of the x-coordinate of the fixations
Y: list of the y-coordinates of the fixations


To overlay the fixations on a given image:

1) First construct a map of size 1280 x 1024 and fill in this map using the fixation points in the eye fixation data to construct the intermediate eye fixation map. 

2) Crop this map with a rectangular region centered around the image center with width, w = 1280 and height,h = 720. This produces the eye fixation map. 

3) Apply a 2D gaussian to blur this map with sigma = 25 pixels to obtain the corresponding fixation density map.