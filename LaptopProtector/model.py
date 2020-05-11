from keras.models import load_model
import numpy as np

model = load_model('./model.h5')


def whoisthis(image):
    """ Returns 1 if it's me """
    if image.ndim == 2:
        image = image[np.newaxis, :, :, np.newaxis]
    elif image.ndim == 3:
        image = image[np.newaxis, :, :]
    return np.argmax(model.predict(image), axis=1)[0]


# from process_image import load_face
#
# file_1 = './Faces/Me/me_1.jpg'
# file_2 = './Faces/notMe/me_17.jpg'
#
# image = load_face(file_2, 64, 64, True)
# print(whoisthis(image))