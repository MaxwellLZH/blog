""" Code snippet from opencv documentation:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
"""
import os
import glob
import cv2
import warnings
warnings.filterwarnings('ignore')


face_data_path = '/Users/Max/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_data_path)

img_file = './Images/Me/me_14.jpg'


def get_faces(image):
    faces = face_cascade.detectMultiScale(image, 1.3, 3)
    if len(faces) == 0:
        return None
    else:
        return faces


def crop_face(image, x, w, y, h):
    return image[y:y + h, x:x + w]


def crop_file(img_file, to_file):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = get_faces(image)
    if faces is None:
        return 0
    x, y, w, h = faces[0]
    cv2.imwrite(to_file, image[y:y+h, x:x+w])
    return 1


def crop_folder(name):
    """ Crop faces from all the images in the folder and save it under
        a folder with matching name under the Faces folder
    """
    n_success = 0
    for i, img_file in enumerate(glob.iglob('./Images/{}/*'.format(name))):
        print(img_file)
        to_folder = './Faces/{}/'.format(name)
        try:
            os.makedirs(to_folder)
        except FileExistsError:
            pass
        file_base_name = os.path.basename(img_file)
        to_file = os.path.join(to_folder, file_base_name)
        n_success += crop_file(img_file, to_file)

        if i % 100 == 0:
            print('Done cropping {} images for {}.'.format(i, name))
    print('Total number of images successfully cropped: {}'.format(n_success))
