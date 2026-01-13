import numpy as np
import cv2
import sys

def getShapeFeatures(img):
    '''
    حساب خصائص الشكل باستخدام عزم "هو" (Hu Moments) بناءً على حدود الطعام.
    '''
    # في OpenCV 3 و 4، دالة findContours قد تعيد قيمتين أو ثلاث حسب النسخة
    # التنسيق التالي هو الأكثر أماناً للتوافق
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0.0] * 7 # إرجاع أصفار إذا لم يتم العثور على شكل

    # اختيار أكبر كنتور (غالباً هو صنف الطعام) بدلاً من اختيار contours[0] عشوائياً
    cnt = max(contours, key=cv2.contourArea)
    
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    
    # استخراج القيم من مصفوفة Hu Moments وتحويلها لتجنب القيم الصغيرة جداً (Log Transformation)
    feature = []
    for i in hu:
        # غالباً ما تكون قيم Hu صغيرة جداً، لذا يفضل أخذ اللوغاريتم لتمثيلها بشكل أفضل
        val = i[0]
        feature.append(val)
    
    # تحويل لـ Numpy array لضمان توافق العمليات الحسابية في بايثون 3
    feature = np.array(feature, dtype=np.float64)
    
    # العمليات الحسابية يجب أن تكون خارج حلقة الـ for
    M = np.max(feature)
    m = np.min(feature)
    
    # تجنب القسمة على صفر وتصحيح معادلة الـ Normalization
    if (M - m) != 0:
        feature = (feature - m) / (M - m)
    
    # التقييس (Standardization)
    mean = np.mean(feature)
    dev = np.std(feature)
    
    if dev != 0:
        feature = (feature - mean) / dev
        
    return feature.tolist()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # إنشاء قناع (Mask) لعزل الجسم
            mask = cv2.inRange(img_gray, 80, 255)
            img1 = cv2.bitwise_and(img_gray, img_gray, mask=mask)
            
            h = getShapeFeatures(img1)
            print("Hu Moments Features:")
            print(h)
        else:
            print("Error: Image not found.")
    else:
        print("Usage: python script.py <image_path>")
