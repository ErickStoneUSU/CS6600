import cv2


def fit_image_ann(ann, image_path):
    # preprocess image
    l = 90
    im = cv2.imread(image_path)
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    grey.resize((l, l), refcheck=False)