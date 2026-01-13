import numpy as np
import cv2
import os

# تأكد أن هذه الملفات موجودة في نفس المسار وتحمل نفس الأسماء
from feature_moments import getShapeFeatures
from feature_gabor import getTextureFeature # تم تحديد الدالة المستدعاة لضمان الوضوح
from feature_color import getColorFeature
from img_seg import getAreaOfFood # تم تحديد الدالة المستدعاة

def createFeature(img):
    feature = []
    
    # استدعاء دوال المعالجة الحقيقية
    # في بايثون 3، نضمن أن الدوال تعيد القيم المتوقعة
    results = getAreaOfFood(img)
    areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = results
    
    color = getColorFeature(colourImg)
    texture = getTextureFeature(colourImg)
    shape = getShapeFeatures(binaryImg)

    # تجميع الخصائص باستخدام extend بدلاً من Loops لزيادة السرعة في بايثون 3
    feature.extend(color)
    feature.extend(texture)
    feature.extend(shape)

    # تحويل للقوائم ومعالجة رياضية (Normalization)
    feature = np.array(feature, dtype=np.float64)
    M = np.max(feature)
    m = np.min(feature)

    # تجنب القسمة على صفر في عملية الـ Normalization
    if (M - m) != 0:
        # الصيغة القياسية للـ Normalization هي (x - min) / (max - min)
        # تم تعديل السطر ليكون أكثر دقة رياضياً
        feature = (feature - m) / (M - m)

    mean = np.mean(feature)
    dev = np.std(feature)

    # التحويل المعياري (Standardization)
    if dev != 0:
        feature = (feature - mean) / dev

    return feature.tolist(), areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier

def readFeatureImg(filename):
    # قراءة الصورة مع التأكد من المسار
    img = cv2.imread(filename)
    
    # التحقق من وجود الصورة قبل المعالجة لتجنب Crash في بايثون 3
    if img is None:
        print(f"Error: Could not read image {filename}")
        # إرجاع مصفوفة أصفار بطول 94 (حسب طول الخصائص المتوقع في مشروعك)
        return np.zeros(94).tolist(), 0, 0, None, 0
        
    return createFeature(img)
