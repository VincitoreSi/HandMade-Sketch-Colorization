'''
file: sketch.py
author: @Vincit0re
objective: Colorize a grayscale sketch image using opencv methods only. This file contains the class for colorizing the sketch image. We first do segmentation on the given image and then fill color on the basis of pre-defined color map.
date: 2023-03-11
'''

from dependencies import *
from utils import *
from hyperparameters import Hyperparameters as hp

# class for sketch colorization
class SketchColor:
    '''This is a class for doing handmade sketches colorization using pre deep learning methods only. We first do segmentation on given image and then fill color on the basis of pre-defined color map.
        Args:
            image [Array]: sketch image
            image_name [str]: name of the image
            thresh [int]: threshold to be used for segmentation purpose
            thresh_type [str]: threshold type to be used
            color_map [str]: color map to be used to fill color (simple, rainbow, summer, spring, winter, autumn, ocean)
            show [bool] (default=True): Whether or not to show the images while creating the colorized version
            save [bool] (default=True): Whether to save the colorized version
    '''
    # initialize the model
    def __init__(self, image, image_name, thresh= 240, thresh_type= 'TOZERO_INV', color_map= 'cool', show=True, save=True) -> None:
        self.image = image
        self.image_name = image_name
        self.thresh = thresh
        self.thresh_type = thresh_type
        self.color_map = color_map
        self.show = show
        self.save = save
    
    # segmentation part
    def segmentation(self):
        '''Method to segment the given sketch into binary regions'''
        if self.show:
            imshow(self.image, "Original Image")
        
        # Perform image segmentation
        if self.thresh_type == 'TOZERO_INV':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_TOZERO_INV)
        elif self.thresh_type == 'TOZERO':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_TOZERO)
        elif self.thresh_type == 'BINARY':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_BINARY)
        elif self.thresh_type == 'BINARY_INV':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_BINARY_INV)
        elif self.thresh_type == 'TRUNC':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_TRUNC)
        elif self.thresh_type == 'TRIANGLE':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_TRIANGLE)
        elif self.thresh_type == 'BINARY_INV':
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.THRESH_BINARY_INV)
        else:
            ret, thresh = cv.threshold(self.image, self.thresh, 255, cv.ADAPTIVE_THRESH_MEAN_C)
            
        if self.show:
            imshow(thresh, 'Segmented Image')
            
        return thresh
    
    # fill color in segmented image
    def colorization(self):
        '''Method to fill the color in the given segmented image'''
        thresh = self.segmentation()
        # set the color map
        color_map = hp._COLOR_MAPS[self.color_map]
        # convert gray image to 3 channels
        sketch_image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        colorized_image = copy.deepcopy(sketch_image)
        
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # fill the color
        for i, contour in enumerate(contours):
            cv.fillPoly(colorized_image, [contour], color_map[i % len(color_map)])

        # Merge the colorized regions
        colorized_image = cv.addWeighted(colorized_image, 0.5, sketch_image, 0.5, 0)
        
        if self.show:
            # show the colorized image
            imshow(colorized_image, 'Colorized Image')
            
        if self.save:
            # Save the colorized image
            save_dir = os.path.join('output/', f'{self.image_name}')
            # if this dir does not exist then create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = save_dir + f'/{self.color_map}.jpg'
            # print(save_dir)
            cv.imwrite(save_dir, colorized_image)
            
        return colorized_image
    
    
# class for sketch colorization from scratch without OpenCV
class SketchColorScratch:
    '''This is a class for doing handmade sketches colorization using pre deep learning methods only. We first do segmentation on given image and then fill color on the basis of pre-defined color map.
        Args:
            image [Array]: sketch image
            image_name [str]: name of the image
            thresh [int]: threshold to be used for segmentation purpose
            thresh_type [str]: threshold type to be used
            color_map [str]: color map to be used to fill color (simple, rainbow, summer, spring, winter, autumn, ocean)
            show [bool] (default=True): Whether or not to show the images while creating the colorized version
            save [bool] (default=True): Whether to save the colorized version
    '''
    # initialize the model
    def __init__(self, image, image_name, thresh= 240, thresh_type= 'TOZERO_INV', color_map= 'cool', show=True, save=True) -> None:
        self.image = image
        self.image_name = image_name
        self.thresh = thresh
        self.thresh_type = thresh_type
        self.color_map = color_map
        self.show = show
        self.save = save

    def cvtColor(self, code):
        if 'BGR2GRAY':
            return np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        elif code == 'BGR2RGB':
            return self.cvtColor(self.image[..., ::-1], 'RGB2BGR')
        elif code == 'RGB2BGR':
            return self.image[..., ::-1]
        else:
            raise ValueError("Unsupported color conversion code")


    def threshold(self, max_value= 255):
        # Convert the input image to grayscale if it's not already
        if len(self.image.shape) > 2:
            gray = self.cvtColor('BGR2GRAY')
        else:
            gray = self.image

        # Initialize a new binary image with the same size as the input image
        binary = np.zeros_like(gray)

        # Perform thresholding based on the thresholding type
        if self.thresh_type == 'BINARY':
            binary[gray > self.thresh] = max_value
        elif self.thresh_type == 'BINARY_INV':
            binary[gray <= self.thresh] = max_value
        elif self.thresh_type == 'TRUNC':
            binary[gray > self.thresh] = self.thresh
        elif self.thresh_type == 'TOZERO':
            binary[gray < self.thresh] = 0
        elif self.thresh_type == 'TOZERO_INV':
            binary[gray >= self.thresh] = 0
        else:
            binary[gray >= self.thresh] = 0
        
        # Return the thresholded image
        return binary

    
    # segmentation part
    def segmentation(self):
        '''Method to segment the given sketch into binary regions'''
        if self.show:
            imshow(self.image, "Original Image")
        
        thresh = self.threshold()
            
        if self.show:
            imshow(thresh, 'Segmented Image')
            
        return thresh
    
    # fill color in segmented image
    def colorization(self):
        '''Method to fill the color in the given segmented image'''
        thresh = self.segmentation()
        # set the color map
        color_map = hp._COLOR_MAPS[self.color_map]
        # convert gray image to 3 channels
        sketch_image = self.cvtColor('GRAY2BGR')
        colorized_image = copy.deepcopy(sketch_image)
        
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # fill the color
        for i, contour in enumerate(contours):
            cv.fillPoly(colorized_image, [contour], color_map[i % len(color_map)])

        # Merge the colorized regions
        colorized_image = cv.addWeighted(colorized_image, 0.5, sketch_image, 0.5, 0)
        
        if self.show:
            # show the colorized image
            imshow(colorized_image, 'Colorized Image')
            
        if self.save:
            # Save the colorized image
            save_dir = os.path.join('output/', f'{self.image_name}')
            # if this dir does not exist then create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = save_dir + f'/{self.color_map}_scratch.jpg'
            # print(save_dir)
            cv.imwrite(save_dir, colorized_image)
            
        return colorized_image
    
    
