import cv2
import numpy as np
import sys

def find_best_contour(contours):
    """دالة مساعدة لاختيار الكنتور بشكل آمن لتجنب أخطاء out of range"""
    if not contours or len(contours) == 0:
        return None
    sorted_cnts = sorted(contours, key=cv2.contourArea)
    # نأخذ أكبر كنتور متاح إذا كان هناك واحد فقط، أو الثاني إذا توفر
    if len(sorted_cnts) >= 2:
        return sorted_cnts[-2]
    else:
        return sorted_cnts[-1]

def getAreaOfFood(img1):
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_filt = cv2.medianBlur(img, 5)
    img_th = cv2.adaptiveThreshold(img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, np.zeros(img.shape, np.uint8), img1, 0, None, 0

    mask = np.zeros(img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_areas[-1]], 0, (255), -1)
    img_bigcontour = cv2.bitwise_and(img1, img1, mask=mask)

    hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
    mask_plate = cv2.inRange(hsv_img, np.array([0, 0, 100]), np.array([255, 90, 255]))
    mask_not_plate = cv2.bitwise_not(mask_plate)
    fruit_skin = cv2.bitwise_and(img_bigcontour, img_bigcontour, mask=mask_not_plate)

    hsv_img_skin = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv_img_skin, np.array([0, 10, 60]), np.array([10, 160, 255]))
    not_skin = cv2.bitwise_not(skin)
    fruit = cv2.bitwise_and(fruit_skin, fruit_skin, mask=not_skin)

    fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    fruit_bin = cv2.inRange(fruit_bw, 10, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode_fruit = cv2.erode(fruit_bin, kernel, iterations=1)

    img_th_fruit = cv2.adaptiveThreshold(erode_fruit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours_f, _ = cv2.findContours(img_th_fruit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- تعديل هام هنا لاستدعاء الدالة بشكل صحيح ---
    fruit_contour = find_best_contour(contours_f)
    
    mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
    if fruit_contour is not None:
        cv2.drawContours(mask_fruit, [fruit_contour], 0, (255), -1)
        fruit_area = cv2.contourArea(fruit_contour)
    else:
        fruit_area = 0

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_fruit2 = cv2.dilate(mask_fruit, kernel2, iterations=1)
    fruit_final = cv2.bitwise_and(img1, img1, mask=mask_fruit2)

    # معالجة منطقة المرجع (Skin)
    skin2 = cv2.subtract(skin, mask_fruit2)
    skin_e = cv2.erode(skin2, kernel, iterations=1)
    img_th_skin = cv2.adaptiveThreshold(skin_e, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours_s, _ = cv2.findContours(img_th_skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- استخدام الدالة الآمنة للمرجع أيضاً ---
    skin_contour = find_best_contour(contours_s)
    
    skin_area = 0
    pix_to_cm_multiplier = 0
    
    if skin_contour is not None:
        skin_rect = cv2.minAreaRect(skin_contour)
        box = cv2.boxPoints(skin_rect)
        box = np.intp(box)
        skin_area = cv2.contourArea(box)
        pix_height = max(skin_rect[1])
        pix_to_cm_multiplier = 5.0 / pix_height if pix_height != 0 else 0

    return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier
