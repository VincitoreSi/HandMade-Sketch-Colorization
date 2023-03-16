'''
file: utils.py
author: @Vincit0re
objective: Utility functions for the project. This file contains all the functions and classes that are used in the project.
date: 2023-03-11
'''

from dependencies import *

# Function to display an image


def imshow(img, title: str) -> None:
    '''Display an image using OpenCV library with the given title. Function will wait for a key press and then close the window.
        Args:
            img [numpy.ndarray]: Image to be displayed.
            title [str]: Title of the window.
    '''
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# class for QuickDraw dataset


class QuickDraw:
    '''Class for QuickDraw dataset. It will read all the images from the given directory and store them in a pandas DataFrame with their respective labels.
        Args:
            path_dir [str]: Path to the directory containing the images.
    '''
    # Constructor

    def __init__(self, path_dir: str) -> None:
        self.path_dir = path_dir
        self.data = pd.DataFrame(columns=['image', 'label'])
        self.classes = []
        self.read_images()

    # Read images from the given directory
    def read_images(self) -> None:
        for folder in os.listdir(self.path_dir):
            self.classes.append(folder)
            for file in os.listdir(self.path_dir + '/' + folder):
                img = cv.imread(self.path_dir + '/' + folder + '/' + file)
                self.data = pd.concat([self.data, pd.DataFrame(
                    [[img, folder]], columns=['image', 'label'])], ignore_index=True)

    # get the data
    def data(self) -> pd.DataFrame:
        if self.classes == []:
            self.read_images()
        return self.data

    # get the classes
    def classes(self) -> list:
        if self.classes == []:
            self.read_images()
        return self.classes


# get train, validation and test data
def get_train_val_test_data(path_dir, val_size=0.2, test_size=0.2):
    '''Function to get the train, validation and test data from the given directory.
        Args:
            path_dir [str]: Path to the directory containing the images.
            val_size [float]: Size of the validation data.
            test_size [float]: Size of the test data.
        Returns:
            train_data [pd.DataFrame]: Training data.
            val_data [pd.DataFrame]: Validation data.
            test_data [pd.DataFrame]: Test data.
    '''
    quickdraw = QuickDraw(path_dir)
    val_len = int(len(quickdraw.data) * val_size)
    test_len = int(len(quickdraw.data) * test_size)

    X, y = shuffle(quickdraw.data['image'], quickdraw.data['label'])

    X_train, y_train, X_, y_ = tts(
        X, y, test_size=val_len + test_len, shuffle=True)
    X_val, y_val, X_test, y_test = tts(
        X_, y_, test_size=test_len, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

# get a random image from the given data and return its path


def random_sketch(path_dir: str) -> str:
    '''Function to get a random image from the given directory of quickdraw dataset.
        Args:
            path_dir [str]: Path to the directory containing the images.
        Returns:
            res [str]: Path to the random image.
    '''
    res = ''
    folders = []
    for folder in os.listdir(path_dir):
        folders.append(folder)

    # choose a random folder
    folder = folders[np.random.randint(0, len(folders))]
    res += path_dir + '/' + folder + '/'
    files = []
    for file in os.listdir(res):
        files.append(file)
    # choose a random file
    file = files[np.random.randint(0, len(files))]
    res += file

    return res

# get random image from the given directory


def random_img(path_dir: str):
    '''This function will return a random image from the given directory.
        Args:
            path_dir [str]: Path to the directory containing the images.
        Returns:
            img [numpy.ndarray]: Random image.
    '''
    files = os.listdir(path_dir)
    file = files[np.random.randint(0, len(files))]
    # find filename by removing extension
    filename = file.split('.')[0]
    file_path = path_dir + '/' + file
    return file_path, filename
