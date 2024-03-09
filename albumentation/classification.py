import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image
from tqdm import tqdm

image = Image.open("images/elon.jpeg")

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ],
            p=1.0, # ? a 100% of time one of these is going to be chosen
        ),
    ]
)

image_list = [image]
image = np.array(image) # converting image into np array
for i in tqdm(range(15)):
    augmentations = transform(image=image)
    augmented_image = augmentations["image"]
    image_list.append(augmented_image)
plot_examples(image_list)


    