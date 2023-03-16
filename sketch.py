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
            save_path [str] (default='output'): Where to save the colorized version
    '''
    # initialize the model

    def __init__(self, image, image_name, thresh=240, thresh_type='TOZERO_INV', color_map='cool', show=True, save=True, save_path='output') -> None:
        self.image = image
        self.image_name = image_name
        self.thresh = thresh
        self.thresh_type = thresh_type
        self.color_map = color_map
        self.show = show
        self.save = save
        self.save_path = save_path

    # segmentation part
    def segmentation(self):
        '''Method to segment the given sketch into binary regions'''
        if self.show:
            imshow(self.image, "Original Image")

        # Perform image segmentation
        if self.thresh_type == 'TOZERO_INV':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_TOZERO_INV)
        elif self.thresh_type == 'TOZERO':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_TOZERO)
        elif self.thresh_type == 'BINARY':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_BINARY)
        elif self.thresh_type == 'BINARY_INV':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_BINARY_INV)
        elif self.thresh_type == 'TRUNC':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_TRUNC)
        elif self.thresh_type == 'TRIANGLE':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_TRIANGLE)
        elif self.thresh_type == 'BINARY_INV':
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.THRESH_BINARY_INV)
        else:
            ret, thresh = cv.threshold(
                self.image, self.thresh, 255, cv.ADAPTIVE_THRESH_MEAN_C)

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
            cv.fillPoly(colorized_image, [contour],
                        color_map[i % len(color_map)])

        # Merge the colorized regions
        colorized_image = cv.addWeighted(
            colorized_image, 0.5, sketch_image, 0.5, 0)

        if self.show:
            # show the colorized image
            imshow(colorized_image, 'Colorized Image')

        if self.save:
            # Save the colorized image
            save_dir = os.path.join(f"{self.save_path}/", f'{self.image_name}')
            # if this dir does not exist then create it
            print(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = save_dir + f'/{self.color_map}.jpg'
            # print(save_dir)
            cv.imwrite(save_dir, colorized_image)

        return colorized_image


# class for sketch colorization


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
            save_path [str] (default='output'): Where to save the colorized version
    '''
    # initialize the model

    def __init__(self, image, image_name, thresh=240, thresh_type='TOZERO_INV', color_map='cool', show=True, save=True, save_path='output') -> None:
        self.image = image
        self.image_name = image_name
        self.thresh = thresh
        self.thresh_type = thresh_type
        self.color_map = color_map
        self.show = show
        self.save = save
        self.save_path = save_path

    # changing color space
    def cvtColor(self, image, code):
        # Convert the input image to a numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Determine the input and output color spaces based on the code
        if code == 'BGR2GRAY':
            input_space = "BGR"
            output_space = "GRAY"
        elif code == 'GRAY2BGR':
            input_space = "GRAY"
            output_space = "BGR"
        elif code == 'BGR2RGB':
            input_space = "BGR"
            output_space = "RGB"
        elif code == 'RGB2BGR':
            input_space = "RGB"
            output_space = "BGR"
        # Add more color space conversions as needed

        # Create a blank output image of the appropriate size and data type
        if output_space == "GRAY":
            output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            output = np.zeros_like(image, dtype=np.uint8)

        # Convert the image from the input to output color space
        if output_space == "GRAY":
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    output[i, j] = np.dot(image[i, j], [0.299, 0.587, 0.114])
        elif input_space == "BGR" and output_space == "RGB":
            output[:, :, 0] = image[:, :, 2]
            output[:, :, 1] = image[:, :, 1]
            output[:, :, 2] = image[:, :, 0]
        elif input_space == "RGB" and output_space == "BGR":
            output[:, :, 0] = image[:, :, 2]
            output[:, :, 1] = image[:, :, 1]
            output[:, :, 2] = image[:, :, 0]
        elif input_space == "GRAY" and output_space == "BGR":
            output = np.dstack((image, image, image))
        return output

    # thresholding part

    def threshold(self, image, thresh_value, max_value, thresh_type):
        # Convert the input image to grayscale if it's not already
        if len(image.shape) == 3:
            raise Exception("Input image must be grayscale")

        # Create a blank image of the same size as the input image
        output = np.zeros_like(image)

        # Apply the thresholding operation based on the threshold type
        if thresh_type == 'BINARY':
            output[image >= thresh_value] = max_value
        elif thresh_type == 'BINARY_INV':
            output[image < thresh_value] = max_value
        elif thresh_type == 'TRUNC':
            output[image >= thresh_value] = thresh_value
            output[image < thresh_value] = image[image < thresh_value]
        elif thresh_type == 'TOZERO':
            output[image >= thresh_value] = image[image >= thresh_value]
        elif thresh_type == 'TOZERO_INV':
            output[image < thresh_value] = image[image < thresh_value]
        else:
            raise Exception("Invalid threshold type")

        return output

    # find contours
    def findContours(self, binary):
        # Create a copy of the binary image for processing
        processed = copy.deepcopy(binary)
        print(processed.shape)

        # Find contours in the processed image
        contours = []
        for i in range(processed.shape[0]):
            for j in range(processed.shape[1]):
                if processed[i, j] >= 200:
                    # Create a new contour and add the first point to it
                    contour = [(i, j)]
                    processed[i, j] = 0
                    current_pixel = (i, j)
                    previous_direction = None

                    # Trace the contour until the starting pixel is reached again
                    while True:
                        # Find the next pixel in the contour
                        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            next_pixel = (
                                current_pixel[0] + direction[0], current_pixel[1] + direction[1])
                            if (next_pixel[0] >= 0 and next_pixel[0] < processed.shape[0] and
                                next_pixel[1] >= 0 and next_pixel[1] < processed.shape[1] and
                                    processed[next_pixel] == 255 and direction != (-previous_direction[0], -previous_direction[1])):
                                contour.append(next_pixel)
                                previous_direction = direction
                                current_pixel = next_pixel
                                processed[current_pixel] = 0
                                break
                        else:
                            # If no next pixel is found, the contour is closed
                            contours.append(np.array(contour))
                            break

        return contours

    # draw contours
    def fillPoly(self, image, polygons, color):
        # Create a new numpy array with the same size as the input image
        result = np.zeros_like(image)

        # Iterate over each polygon
        for poly in polygons:
            # Create a new numpy array with the same size as the input image
            mask = np.zeros_like(image)

            # Create a pandas DataFrame from the polygon vertices
            df = pd.DataFrame(poly, columns=['x', 'y'])

            # Compute the minimum and maximum x and y values of the polygon
            min_x = int(df['x'].min())
            max_x = int(df['x'].max())
            min_y = int(df['y'].min())
            max_y = int(df['y'].max())

            # Iterate over each pixel within the bounding box of the polygon
            for y in range(min_y, max_y+1):
                for x in range(min_x, max_x+1):
                    # Check if the pixel is within the polygon
                    if self.pointInPolygon(x, y, poly):
                        # Set the corresponding pixel in the mask to 1
                        mask[y, x] = 1

            # Apply the color to the mask
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3)) * np.array(color)

            # Apply the mask to the result image
            result = np.where(mask != 0, mask, result)

        # Return the result
        return result

    # check if a point is inside a polygon
    def pointInPolygon(self, x, y, poly):
        # Determine if a point is inside a polygon using the "ray casting" algorithm
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (y - p1y) * (p2x - p1x) / \
                                (p2y - p1y) + p1x
                            if p1x == p2x or x <= x_inters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # add weighted two images
    def addWeighted(self, src1, alpha, src2, beta, gamma=0):
        # Ensure that the inputs have the same shape
        assert src1.shape == src2.shape, "Input arrays must have the same shape"

        # Calculate the weighted sum
        output = alpha * src1 + beta * src2 + gamma

        # Clip the output to the range [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output

    # segmentation part

    def segmentation(self):
        '''Method to segment the given sketch into binary regions'''
        if self.show:
            imshow(self.image, "Original Image")

        thresh = self.threshold(self.image, self.thresh, 255, self.thresh_type)

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
        sketch_image = self.cvtColor(self.image, 'GRAY2BGR')
        colorized_image = copy.deepcopy(sketch_image)

        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours = self.findContours(thresh)
        # print(contours)

        # fill the color
        for i, contour in enumerate(contours):
            cv.fillPoly(colorized_image, [contour],
                        color_map[i % len(color_map)])

        # Merge the colorized regions
        colorized_image = self.addWeighted(
            colorized_image, 0.5, sketch_image, 0.5, 0)

        if self.show:
            # show the colorized image
            imshow(colorized_image, 'Colorized Image')

        if self.save:
            # Save the colorized image
            save_dir = os.path.join(f"{self.save_path}/", f'{self.image_name}')
            # if this dir does not exist then create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = save_dir + f'/{self.color_map}_scratch.jpg'
            # print(save_dir)
            cv.imwrite(save_dir, colorized_image)

        return colorized_image
