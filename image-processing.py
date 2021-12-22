import cv2 as cv2
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = ROOT_DIR + "\\dataset-cropped\\"
output_dir = ROOT_DIR + "\\dataset-processed\\"

output_types = ["cut-circles", "cut-circles-grayscale", "cut-squares", "cut-squares-grayscale"]

debug_mode = False
show_images = False


def generate_processed_images():
    print("\n" * 4)
    print("### Starting image processing: called generate_processed_images()")

    # We just walk though the whole input_dir and process any JPEGs found.
    for subdir, dirs, files in os.walk(input_dir):
        print("# Going through " + subdir)
        for file in files:
            original_img = cv2.imread(subdir + "\\" + file)
            species_name = os.path.basename(subdir)

            for index, out_type in enumerate(output_types):
                img_result = process_image_hough_circles(original_img, out_type)
                out_path = output_dir + "\\" + out_type + "\\" + species_name + "\\" + file

                cv2.imwrite(out_path, img_result)

                if debug_mode:
                    print("Successfully processed and saved " + out_path + " with processed image.")

        print("# Finished walking " + subdir)

    print("### Finished image processing: ending generate_processed_images()")


def process_image_hough_circles(image, out_type):
    # Histogram equalization using contrast limits.
    img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cl_ahe = cv2.createCLAHE(2.0, (8, 8))
    img_equalized = cl_ahe.apply(img_grayscale)
    if show_images: show_image(img_equalized)

    # Then we blur the image so it's easier to work with.
    # We don't want edges as strong as the ones in the dataset for recognizing circles.
    img_blurred = cv2.medianBlur(img_equalized, 15)
    if show_images: show_image(img_blurred)

    # HoughCircles takes care of the whole process for detecting the circles.
    # It does a CannyEdge pass automatically. Param1 is the higher threshold for it.
    # minDist sets the min distance for circle centers. Doesn't work 100% of the time, but helps with
    # detecting the same thing more than once. Radius setting is also specific to the dataset.
    # The rest are recommended values.
    circles = cv2.HoughCircles(
        image=img_blurred,
        method=cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=500000,
        param1=50,
        param2=120,
        minRadius=240,
        maxRadius=400)

    circles = np.uint16(np.around(circles))

    img_output = None

    # Here we go through the circles found (usually 1-2) and create a mask with them.
    # This mask is used to match the original cropped image so we're left with the end result.
    for c in circles[0, :]:
        if "grayscale" in out_type:
            img_cropped = img_grayscale[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]
        else:
            img_cropped = image[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]

        img_mask = np.zeros_like(img_cropped)
        cv2.circle(img_mask, (c[2], c[2]), c[2], (255, 255, 255), -1)
        img_output = np.zeros_like(img_cropped)
        img_output[img_mask == 255] = img_cropped[img_mask == 255]

    return img_output


# Not in use. Favoured the Hough approach in the functions above.
def process_image_manually(image, out_type):
    # Histogram equalization using contrast limits.
    img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cl_ahe = cv2.createCLAHE(2.0, (8, 8))
    img_equalized = cl_ahe.apply(img_grayscale)
    if show_images: show_image(img_equalized)

    # Then we blurry the image so it's easier to work with.
    # We don't want edges as strong as the ones in the dataset for recognizing circles.
    img_blurred = cv2.medianBlur(img_equalized, 5)
    if show_images: show_image(img_blurred)

    # Then we use Canny Edge Detection to, well, get edges.
    # Context: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    img_edges = cv2.Canny(img_blurred, 70, 450)
    if show_images: show_image(img_edges)

    # Dilate the edges to try and always have them closing in a circle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img_dilated = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel)
    if show_images: show_image(img_dilated)

    # Find all contours in the image.
    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Then we use some black magic (math) to try and find the correct circular one.
    contour_index, chosen_contour = find_correct_contour(contours, img_edges)

    # Use the contour to mask off just the circle in the original image.
    # This changes everything inside the contour to white.
    img_mask = np.zeros_like(img_edges)
    img_mask = cv2.drawContours(img_mask, contours, contour_index, 255, -1)
    if show_images: show_image(img_mask)

    # Then we copy the white part of the image to img_output.
    img_output = np.zeros_like(image)
    img_output[img_mask == 255] = image[img_mask == 255]
    if show_images: show_image(img_output)

    # Now we just need to resize the output image and cut its borders.
    img_resized_output = cut_circular_borders(img_output, chosen_contour)
    if show_images: show_image(img_resized_output)

    return img_resized_output


def find_correct_contour(contours, img_edges):
    contour_index = -1
    chosen_contour = -1

    # We know the area of the circle is between 150k and 250k, so we just filter by those.
    # Pretty simple, but because our dataset is very uniform, it works well.
    # See https://stackoverflow.com/questions/41561731/opencv-detecting-circular-shapes for context and alternatives.
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 150000 <= area <= 250000:
            contour_index = i
            chosen_contour = contour

            if debug_mode:
                print("# Contour area: " + str(area))

    if contour_index == -1:
        show_image(img_edges)
        raise Exception("Failed to find correct circular contour in find_correct_contour!")

    return contour_index, chosen_contour


def cut_circular_borders(image, chosen_contour):
    x, y, w, h = cv2.boundingRect(chosen_contour)
    return image[y: y + h, x: x + w]


def clean_processed_images():
    for subdir, dirs, files in os.walk(output_dir):
        for species_dir, species_dirs, species_files in os.walk(subdir):
            for filename in os.listdir(species_dir):
                file_path = os.path.join(species_dir, filename)
                if debug_mode:
                    print("Deleting " + filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


def show_image(image, name="image"):
    image_resized = resize_with_aspect_ratio(image, height=960)
    cv2.imshow(name, image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def paint_circles_on_image(image, circles):
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    return image


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
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


clean_processed_images()
generate_processed_images()
