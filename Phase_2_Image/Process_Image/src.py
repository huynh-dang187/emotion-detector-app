import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("many_people.jpg"))

if img is None:
    sys.exit("Could not read the image.")

# Đổi ảnh sang grayscale  -> Mục đích của việc đổi màu : giảm dung lượng , dễ xử lí hơn
gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

# Đổi ảnh dạng RGB để matplotlib và deepface có thể xử lí 
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


cv.imshow("Display window", img)
cv.imshow("gray",gray)
cv.imshow("Non_color",img_rgb)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("many_people.jpg", img)
    
