import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pyautogui
import tensorflow as tf

print("Package Imported")

# reading file
profile_cascade = cv.CascadeClassifier("Lib\\site-packages\\cv2\\data\\haarcascade_profileface.xml")
frontal_cascade = cv.CascadeClassifier("Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")
img = cv.imread(r"C:\Users\PC Programming\Pictures\py_resources\face\test3.jpg")
# img_2 = cv.imread(r"D:\Zaim\image process\OpencvProject\Resources\tetris_blocks.png")

# Calculate screen size
screen_width, screen_height = pyautogui.size()

# Calculate initial image size
(image_height, image_width, _) = img.shape

print("Screen size :", (screen_height, screen_width))
print("inital image size : ({}, {})\n".format(image_width, image_height))

# Set resize metric
resize_metric = 0.8

while True:
    if(image_height < screen_height and image_width < screen_width):
        break

        # Calculate new size
        image_height = int(image_height * resize_metric)
        image_width = int(image_width * resize_metric)

        print("Resizing image to :({}, {})\n".format(image_width, image_height))

        # Calculate new image size
        img = cv.resize(img, (image_height, image_width))

# edit file
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(img, (17, 7), 0)
imgCanny = cv.Canny(img, 30, 150)

# histogram testing
assert img is not None, "file could not be read, check with os.path.exists()"

# histogram formula
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

imgGray2 = cdf[imgGray]

"""faces = haarcascade.detectMultiScale(
    imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
"""

faces_profile = profile_cascade.detectMultiScale \
    (imgGray2, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

faces_frontal = frontal_cascade.detectMultiScale \
    (imgGray2, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))

# Check if faces were detected before concatenating
if len(faces_frontal) > 0 and len(faces_profile) > 0:
    # Combine the results
    faces = np.vstack((faces_profile, faces_frontal))
else:
    # Use the results from the cascade that detected faces
    faces = faces_frontal if len(faces_profile) > 0 else faces_frontal

print("face :", *faces)
print('Faces found: ', len(faces))
for face in faces:
    x, y, w, h = face

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

    img[y: y + h, x: x + w] = cv.GaussianBlur(
        img[y: y + h, x: x + w], (29, 29), 0
    )

# ---------------------------------------------------------------------------
# output file

"""
print("Package Imported")
plt.imshow(img)
plt.figure(figsize=(20,10))
plt.imshow(img)
plt.axis('off')
"""

# Create window with freedom of dimensions

cv.imshow("Output", img)
cv.waitKey(0)
cv.destroyAllWindows()
