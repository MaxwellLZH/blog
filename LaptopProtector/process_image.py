import numpy as np
import glob
import cv2


def load_face(path, width=180, height=180, gray_scale=True):
    image = cv2.imread(path)
    image = cv2.resize(image, (width, height))
    if gray_scale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape([width, height, -1])
    return image


def load_faces(name, width=180, height=180, gray_scale=True):
    """ Load all the face images for `name` into a single list """
    faces = list()
    image_dir = './Faces/{}/'.format(name)
    for face in glob.iglob(image_dir + '*'):
        faces.append(load_face(face, width=width, height=height, gray_scale=gray_scale))
    return faces


def load_faces_to_array(name, width=180, height=180, gray_scale=True):
    """" Vertically stack the loaded faces into a single numpy matrix"""
    faces = load_faces(name, width=width, height=height, gray_scale=gray_scale)
    return np.vstack([f[np.newaxis, :, :] for f in faces])

