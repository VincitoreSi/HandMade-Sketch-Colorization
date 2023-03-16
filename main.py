'''
file: main.py
author: @Vincit0re
objective: Colorize a grayscale sketch image using a pre-trained model. This is main file to run the code.
date: 2023-03-12
'''

from dependencies import *
from utils import *
from hyperparameters import Hyperparameters as hp
from sketch import SketchColor

# main function
def main():
    # Load the grayscale sketch image
    img, img_name = random_img('data')
    parser = argparse.ArgumentParser('Colorize a grayscale sketch image using a pre-trained model')
    parser.add_argument('-i','--image', type=str, default=img, help='path to the image')
    parser.add_argument('-n','--image_name', type=str, default=img_name, help='name of the image')
    parser.add_argument('-t','--thresh', type=int, default=180, help='threshold to be used for segmentation purpose')
    parser.add_argument('-tt','--thresh_type', type=str, default='TOZERO_INV', help='threshold type to be used')
    parser.add_argument('-c','--color_map', type=str, default='cool', help='color map to be used to fill color (simple, rainbow, summer, spring, winter, autumn, ocean)')
    parser.add_argument('-s','--show', type=bool, default=True, help='Whether or not to show the images while creating the colorized version')
    
    args = parser.parse_args()
    
    sketch_image = cv.imread(args.image, 0)
    colorized_sketch = SketchColor(image=sketch_image, image_name= args.image_name, thresh=args.thresh, color_map=args.color_map, thresh_type=args.thresh_type, show=args.show).colorization()


if __name__ == '__main__':
    main()