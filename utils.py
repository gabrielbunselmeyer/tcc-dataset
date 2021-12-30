import cv2 as cv2
import numpy as np
import math
import os


# Paints hough circles in green on the passed image. The circles param must be the output of a cv2.HoughCircles call.
# (or some other function that returns cv2.circles)
def paint_circles_on_image(image, circles):
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    return image


# Calculates the biggest square possible within the circle with the passed in radius, and crops the image to it.
def crop_square_inside_circle(image, radius):
    square_side = math.sqrt(radius * radius * 2)
    square_half = square_side * 0.5

    x = int(radius - square_half)
    y = int(radius - square_half)
    w = int(square_side)
    h = int(square_side)

    img_mask = np.zeros_like(image)
    img_out = np.zeros_like(image)

    cv2.rectangle(img_mask, (x, y), ((x + w), (y + h)), (255, 255, 255), -1)

    img_out[img_mask == 255] = image[img_mask == 255]
    return img_out[y: y + h, x: x + w]


# Cuts square borders around a given contour (usually a circle).
def cut_circular_borders(image, chosen_contour):
    x, y, w, h = cv2.boundingRect(chosen_contour)
    return image[y: y + h, x: x + w]


# Removes all the images within the output directory.
def clean_processed_images(output_dir, is_debug_enabled):
    for subdir, dirs, files in os.walk(output_dir):
        for species_dir, species_dirs, species_files in os.walk(subdir):
            for filename in os.listdir(species_dir):
                file_path = os.path.join(species_dir, filename)
                if is_debug_enabled:
                    print("Deleting " + filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


# Displays the given image and waits for user input for closing the window/proceeding with the execution.
def show_image(image, name="image"):
    image_resized = resize_with_aspect_ratio(image, height=960)
    cv2.imshow(name, image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Resize image keeping the given ratio. Copied from some StackOverflow answer.
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
