import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

def cvtImg(img):
    result = img.copy()

    for i in range(3):
        gray = result[:, :, i]
        ret, gray_1bit = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        result[:, :, i] = gray_1bit
    return result

def compareImg(img1, img2):
#    result = np.zeros(img1.shape, np.uint8)
    result = np.where(img1==img2, (255,255,255), (0,0,0))
    result = result.astype(np.uint8)
    return result

def main():
    filename1 = "line_digital.png"
    img_digital = cv2.imread(filename1)

    filename2 = "line_analog.png"
    img_analog = cv2.imread(filename2)
    img_3bit = cvtImg(img_analog)

    y_true = img_digital.flatten()
    y_pred = img_3bit.flatten()

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    #cv2.imshow("", comparison)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()