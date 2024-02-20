import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\tesseract.exe'


# Read the image using OpenCV
image = cv2.imread('C:\\Users\\VK\\Desktop\\miniproject\\d.jpg')

# Preprocess the image (optional)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Perform text extraction using Tesseract
text = pytesseract.image_to_string(gray)

# Print the extracted text
print(text)