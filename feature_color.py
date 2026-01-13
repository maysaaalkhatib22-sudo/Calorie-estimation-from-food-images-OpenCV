import cv2
import math
import sys
import numpy as np

def getColorFeature(img):
    '''
    Computes the color feature vector of the image
    based on HSV histogram
    '''
    # تحويل الصورة إلى فضاء HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # حساب الهيستوجرام لـ 3 قنوات (H, S, V)
    # التقسيم: 6 مستويات لـ H، مستويان لـ S، ومستويان لـ V
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [6, 2, 2], [0, 180, 0, 256, 0, 256])
    
    # تحويل الهيستوجرام إلى مصفوفة أحادية البعد (Flatten)
    featurevec = hist.flatten()
    
    # استثناء العنصر الأول كما في الكود الأصلي (غالباً ما يكون للخلفية السوداء)
    feature = featurevec[1:]
    
    # تحويل إلى مصفوفة Numpy لضمان عمل العمليات الحسابية في بايثون 3
    feature = np.array(feature, dtype=np.float64)
    
    # تصحيح العمليات الحسابية
    # ملاحظة: في بايثون 3 لا يمكن ضرب 'map' مباشرة، لذا نستخدم مصفوفات numpy
    feature = feature * 2
    
    M = np.max(feature)
    m = np.min(feature)
    
    # تجنب القسمة على صفر وتصحيح المعادلة الرياضية (Normalization)
    if (M - m) != 0:
        feature = (feature - m) / (M - m)
    
    # التقييس (Standardization)
    mean = np.mean(feature)
    dev = np.std(feature)
    
    if dev != 0:
        feature = (feature - mean) / dev

    return feature.tolist()

if __name__ == '__main__':
    # التحقق من أن المستخدم أدخل مسار الصورة
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            featureVector = getColorFeature(img)
            print(featureVector)
            # waitKey و destroyAllWindows تعمل فقط إذا كان هناك نافذة عرض cv2.imshow
        else:
            print("Error: Image not found.")
    else:
        print("Usage: python script_name.py <image_path>")
