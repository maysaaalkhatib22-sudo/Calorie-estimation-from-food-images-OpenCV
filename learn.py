import numpy as np
import cv2
import os
import csv
from create_feature import readFeatureImg # تأكد من أن اسم الملف هو create_feature.py

def get_svm():
    # في OpenCV 4+ يتم إنشاء الـ SVM بهذه الطريقة
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    # المعاملات (Hyperparameters)
    svm.setC(2.67)
    svm.setGamma(5.383)
    return svm

def training():
    feature_mat = []
    response = []
    # التأكد من المسار الصحيح للمجلد
    base_path = "./All_Images/" 
    
    # التحقق من وجود المجلد قبل البدء
    if not os.path.exists(base_path):
        print(f"❌ المجلد {base_path} غير موجود!")
        return

    print("🚀 بدء استخراج الخصائص للتدريب...")
    
    for j in range(1, 15): # الأصناف (مثلاً أنواع الفواكه)
        for i in range(1, 21): # الصور لكل صنف
            img_path = os.path.join(base_path, f"{j}_{i}.jpg")
            if os.path.exists(img_path):
                try:
                    # قراءة الصورة واستخراج الخصائص
                    fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
                    feature_mat.append(fea)
                    response.append(j) # رقم الصنف
                except Exception as e:
                    print(f"⚠️ خطأ في معالجة {img_path}: {e}")
            else:
                # هذا السطر اختياري، يطبع الصور المفقودة فقط
                pass

    if len(feature_mat) == 0:
        print("❌ فشل التدريب: لم يتم العثور على أي بيانات صالحة في All_Images!")
        return

    # تحويل البيانات إلى تنسيق Numpy المتوافق مع OpenCV 4
    trainData = np.array(feature_mat, dtype=np.float32)
    responses = np.array(response, dtype=np.int32).reshape(-1, 1)

    print(f"📊 تم جمع {len(feature_mat)} نموذج تدريبي. جاري التدريب...")

    svm = get_svm()
    # استخدام ROW_SAMPLE لأن كل صف يمثل صورة واحدة
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    
    # حفظ النموذج بصيغة XML (وهي الصيغة الموصى بها في OpenCV 4)
    svm.save('svm_data.xml')
    print("✅ تم التدريب وحفظ النموذج بنجاح في الملف: svm_data.xml")

if __name__ == '__main__': # تصحيح الخطأ الإملائي من main إلى __main__
    training()
