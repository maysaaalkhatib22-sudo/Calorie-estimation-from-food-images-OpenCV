import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

def build_filters():
    '''
    حساب نواة جابور (Gabor kernel) لاستخدامها في استخراج خصائص النسيج.
    '''
    filters = []
    ksize = 31
    # تقسيم الزوايا والترددات لإنشاء مجموعة مرشحات متنوعة
    for theta in np.arange(0, np.pi, np.pi / 8):
        for wav in [8.0, 13.0]:
            for ar in [0.8, 2.0]:
                kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, wav, ar, 0, ktype=cv2.CV_32F)
                filters.append(kern)
    
    # ملاحظة: cv2.imshow قد تسبب توقفاً إذا لم تكن هناك واجهة رسومية، يفضل استخدامها للاختبار فقط
    # cv2.imshow('filt', filters[9]) 
    return filters

def process_threaded(img, filters, threadn=8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)
    
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    pool.close() # إغلاق الـ pool لتحرير الذاكرة في بايثون 3
    pool.join()
    return accum

def EnergySum(img):
    # حساب المتوسط والانحراف المعياري للفلتر
    mean, dev = cv2.meanStdDev(img)
    # في بايثون 3 نصل للقيم مباشرة من المصفوفة الناتجة
    return mean[0][0], dev[0][0]

def process(img, filters):
    '''
    حساب الخصائص بناءً على المرشحات.
    '''
    feature = []
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)  
        a, b = EnergySum(fimg)
        feature.append(a)
        feature.append(b)
        np.maximum(accum, fimg, accum)
    
    # تحويل لـ Numpy Array لضمان التوافق مع العمليات الحسابية في بايثون 3
    feature = np.array(feature, dtype=np.float64)
    
    M = np.max(feature)
    m = np.min(feature)
    
    # تصحيح مشكلة map: في بايثون 3 map تعيد iterator، لذا نستخدم مصفوفات numpy مباشرة
    feature = feature * 2
    
    if (M - m) != 0:
        feature = (feature - m) / (M - m)
    
    mean_val = np.mean(feature)
    dev_val = np.std(feature)
    
    if dev_val != 0:
        feature = (feature - mean_val) / dev_val
        
    return feature.tolist()

def getTextureFeature(img):
    '''
    تحويل الصورة لرمادي ثم استخراج خصائص النسيج.
    '''
    filters = build_filters()
    # التأكد من تحويل الصورة لرمادي لأن مرشحات جابور تعمل على قناة واحدة عادة
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
        
    res1 = process(gray_image, filters)
    return res1

if __name__ == '__main__':
    import sys
    
    # التعامل مع مدخلات سطر الأوامر
    if len(sys.argv) > 1:
        img_fn = sys.argv[1]
    else:
        img_fn = 'test.JPG'
    
    img = cv2.imread(img_fn)
    
    if img is not None:
        features = getTextureFeature(img)
        print(f"Extracted {len(features)} texture features.")
        print(features)
    else:
        print("Image not found!")
    
    # cv2.waitKey() # تعمل فقط في البيئات التي تدعم النوافذ المنبثقة
