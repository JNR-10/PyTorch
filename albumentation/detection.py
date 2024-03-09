import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = cv2.imread("images/cat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ? opencv loads image default as BGR
bboxes = [[13, 170, 224, 410]]
# ? ideally we would have a list for each of the bounding box for all images (list of lists)

# ? for Pascal_voc format is (x_min, y_min, x_max, y_max), (YOLO, COCO) =>  they have differnt format for bounding box

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048, # min_area so that the cat always remains in the picture while cropping
                                # ? bounding box in the resulting image should atleast have 2048 pixels of data
                                min_visibility=0.3, # bounding box should contain this much amount of resulting image (this can be wring)
                                label_fields=[])
)

images_list = [image]
# ? no need to convert to np array as we used opencv
saved_bboxes = [bboxes[0]]
for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_img = augmentations["image"]

    # ? so that the list does not remain empty due to the constrains that we gave above in BboxParams
    if len(augmentations["bboxes"]) == 0:
        continue

    images_list.append(augmented_img)
    saved_bboxes.append(augmentations["bboxes"][0]) # this will return list of list but we want only the list

plot_examples(images_list, saved_bboxes)