from skimage.metrics import structural_similarity
from PIL import Image
import imutils
import cv2
import requests
import os

original = Image.open(requests.get("https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg", stream=True).raw)
tampered = Image.open(requests.get("https://bl-i.thgim.com/public/migration_catalog/article18354371.ece/alternates/LANDSCAPE_1200/new-pan-card", stream=True).raw)

print("Original image format: ", original.format)
print("Tampred image format: ", tampered.format)


print("Original image size: ", original.size)
print("Tampered image size: ", tampered.size)

original = original.resize((250, 160))
tampered = tampered.resize((250, 160))



if not os.path.exists("1\pan_card_tampering\image\original.png"):
    os.makedirs(os.path.dirname("1\pan_card_tampering\image\original.png"), exist_ok=True)
    original.save("1\pan_card_tampering/image/original.png")

if not os.path.exists("1\pan_card_tampering\image\tampered.png"):
    os.makedirs(os.path.dirname("1\pan_card_tampering\image\tampered.png"), exist_ok=True)    
    tampered.save("1/pan_card_tampering/image/tampered.png")
    
original = cv2.imread("1\pan_card_tampering\image\original.png")
tampered = cv2.imread("1/pan_card_tampering/image/tampered.png")



original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff , 0 , 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts  = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x,y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x,y), (x + w, y + h), (0, 0, 255), 2)

Image.fromarray(original)
Image.fromarray(tampered)
Image.fromarray(thresh).show()


