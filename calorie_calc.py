import cv2
import numpy as np

# الكثافة - جرام / سم^3
density_dict = { 
    1:0.609, 2:0.94, 3:0.577, 4:0.641, 5:1.151, 
    6:0.482, 7:0.513, 8:0.641, 9:0.481, 10:0.641, 
    11:0.521, 12:0.881, 13:0.228, 14:0.650 
}

# السعرات لكل 100 جرام
calorie_dict = { 
    1:52, 2:89, 3:92, 4:41, 5:360, 
    6:47, 7:40, 8:158, 9:18, 10:16, 
    11:50, 12:61, 13:31, 14:30 
}

# معامل تحويل مساحة المرجع (مثلاً عملة معدنية أو علامة)
skin_multiplier = 5 * 2.3

def getCalorie(label, volume):
    '''
    حساب الكتلة وإجمالي السعرات الحرارية بناءً على الحجم المحسوب.
    '''
    label = int(label)
    calorie_per_100g = calorie_dict.get(label, 0)
    
    if volume is None:
        return None, None, calorie_per_100g
    
    density = density_dict.get(label, 0.6) # 0.6 قيمة افتراضية
    mass = volume * density
    
    # حساب السعرات الإجمالية: (السعرات لكل 100 جرام / 100) * الكتلة بالجرام
    calorie_tot = (calorie_per_100g / 100.0) * mass
    
    return mass, calorie_tot, calorie_per_100g

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    '''
    تقدير الحجم بناءً على الشكل الهندسي لكل صنف (كرة، أسطوانة، أو مسطح).
    '''
    label = int(label)
    
    # تجنب القسمة على صفر إذا لم يتم رصد المرجع
    if skin_area == 0:
        skin_area = 1 
        
    # تحويل المساحة من بكسل إلى سم مربع بناءً على المرجع
    area_fruit = (area / skin_area) * skin_multiplier 
    
    volume = 0
    
    # 1. الأشكال الكروية (تفاح، طماطم، برتقال، كيوي، بصل)
    if label in [1, 9, 7, 6, 12]:
        radius = np.sqrt(area_fruit / np.pi)
        # في بايثون 3، (4/3) تعطي 1.333 تلقائياً، وهي صحيحة هنا
        volume = (4.0/3.0) * np.pi * (radius ** 3)
    
    # 2. الأشكال الأسطوانية (موز، خيار، وجزرة كبيرة)
    elif label in [2, 10] or (label == 4 and area_fruit > 30):
        if fruit_contour is not None:
            fruit_rect = cv2.minAreaRect(fruit_contour)
            # عرض أو طول الفاكهة بالسم
            height = max(fruit_rect[1]) * pix_to_cm_multiplier
            if height == 0: height = 1 # لتجنب القسمة على صفر
            radius = area_fruit / (2.0 * height)
            volume = np.pi * (radius ** 2) * height
        else:
            volume = area_fruit * 2.0 # تقدير احتياطي
            
    # 3. الأشكال المسطحة أو الصغيرة (جبنة، جزر صغير، صلصة)
    elif (label == 4 and area_fruit <= 30) or label in [5, 11]:
        volume = area_fruit * 0.5 # افتراض أن السمك 0.5 سم
    
    # 4. أصناف يصعب تقدير حجمها من صورة واحدة
    elif label in [8, 14, 3, 13]:
        volume = None
    
    else:
        # افتراض عام لأي صنف غير معرف
        volume = area_fruit * 1.0 

    return volume
