'''
file: main.py
author: @Vincit0re
objective: Colorize a grayscale sketch image using a pre-trained model. This is main file to run the code.
date: 2023-03-12
'''

from dependencies import *
from utils import *
from hyperparameters import Hyperparameters as hp
from sketch import SketchColor, SketchColorScratch

# main function


def main():
    # Load the grayscale sketch image
    img, img_name = random_img('data')
    parser = argparse.ArgumentParser(
        'Colorize a grayscale sketch image using a pre-trained model')
    parser.add_argument('-i', '--image', type=str,
                        default=img, help='path to the image')
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='path to the folder where to save output')
    parser.add_argument('-n', '--image_name', type=str,
                        default=img_name, help='name of the image')
    parser.add_argument('-t', '--thresh', type=int, default=180,
                        help='threshold to be used for segmentation purpose')
    parser.add_argument('-tt', '--thresh_type', type=str,
                        default='TOZERO_INV', help='threshold type to be used')
    parser.add_argument('-c', '--color_map', type=str, default='cool',
                        help='color map to be used to fill color (simple, rainbow, summer, spring, winter, autumn, ocean)')
    parser.add_argument('-s', '--show', type=bool, default=True,
                        help='Whether or not to show the images while creating the colorized version')
    parser.add_argument('-sv', '--save', type=bool, default=True,
                        help='Whether to save the output or not')

    # parse the arguments
    args = parser.parse_args()
    
    if args.image != img and args.image_name == img_name:
        args.image_name = (args.image.split('/')[-1]).split('.')[0]

    sketch_image = cv.imread(args.image, 0)
    median_val = np.median(sketch_image)/255
    
    # print(f"Average value: {median_val:.4f}")
    if args.thresh == 180:
        if median_val < 0.3:
            args.thresh = 100
        elif median_val < 0.5:
            args.thresh = 120
        elif median_val < 0.7:
            args.thresh = 150
        elif median_val < 0.9:
            args.thresh = 180
        elif median_val < 0.95:
            args.thresh = 190
        else:
            args.thresh = 200
    
    start_time = time.time()
    colorized_sketch = SketchColor(image=sketch_image, image_name=args.image_name, thresh=args.thresh,
                                          color_map=args.color_map, thresh_type=args.thresh_type, show=args.show, save=args.save, save_path=args.output).colorization()
    
    end_time = time.time()
    print(f"Time taken to colorize the image: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    main()
