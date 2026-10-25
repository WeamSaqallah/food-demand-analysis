---
name: demand-volatility-kpis
description: مهارة متخصصة في حساب مؤشرات الأداء التشغيلية ومؤشرات الطلب، وتصنيف الوجبات والمراكز وفق معامل التذبذب (Coefficient of Variation) ومبدأ باريتو لتحديد مستويات المخزون الاحتياطي.
---

# مهارة حساب مؤشرات الطلب والتذبذب التشغيلي (Demand Volatility & KPIs)

## الهدف
تمكين المحلل والنموذج من:
1. احتساب مؤشرات الأداء الرئيسية (KPIs) للطلب والإيرادات والخصومات بدقة متناهية.
2. قياس استقرار وتذبذب الطلب لكل وجبة ومركز عبر الزمن.
3. فرز الوجبات لتحديد التي تحتاج إلى مخزون احتياطي مرتفع (Safety Stock Buffer) لمنع نفادها وتلف المواد الخام.

---

## المؤشرات الرياضية المعتمدة (Mathematical Formulations)

### 1. مؤشرات الحجم والمالية
* **إجمالي الطلبات (Total Orders):** $\sum \text{num\_orders}$
* **متوسط الطلب الأسبوعي (Weekly Average):** $\frac{\sum \text{num\_orders}}{\text{Number of Weeks}}$
* **إجمالي الإيرادات (Total Revenue):** 
  $$\text{Revenue} = \sum (\text{checkout\_price} \times \text{num\_orders})$$
* **نسبة الخصم الفعلي (Discount Rate):**
  $$\text{Discount Rate} = \frac{\text{base\_price} - \text{checkout\_price}}{\text{base\_price}}$$

### 2. مؤشر تذبذب الطلب (Coefficient of Variation - CV)
* يقيس التشتت النسبي للطلب بغض النظر عن الحجم الإجمالي:
  $$CV = \frac{\sigma}{\mu} = \frac{\text{الانحراف المعياري لطلبات الوجبة}}{\text{متوسط طلبات الوجبة}}$$
* **قاعدة اتخاذ القرار للمخزون:**
  * **$CV < 0.5$ (طلب مستقر / نمط منتظم):** يحتاج مخزوناً احتياطياً منخفضاً (Just-In-Time).
  * **$0.5 \le CV \le 1.0$ (طلب متوسط التذبذب):** يحتاج مخزون أمان متوازن مع تتبع أسبوعي.
  * **$CV > 1.0$ (طلب عالي التذبذب / غير متوقع):** يحتاج مخزوناً احتياطياً أعلى وسياسات طلب مرنة لمنع نفاذ المخزون أو تلف المواد.

### 3. تحليل باريتو 80/20 (Pareto Analysis)
* ترتيب الوجبات/الفئات تنازلياً حسب إجمالي الطلب.
* حساب النسبة التراكمية (Cumulative Percentage) لمعرفة المنتجات الحيوية التي تمثل 80% من النشاط التشغيلي لتركيز الرقابة عليها.

---

## الكود البرمجي المرجعي (Python Implementation)

```python
import pandas as pd
import numpy as np

def calculate_meal_volatility_kpis(df):
    """
    يحسب مؤشرات الطلب ومعامل الاختلاف وتصنيف باريتو لكل وجبة
    """
    meal_stats = df.groupby(['meal_id', 'category', 'cuisine']).agg(
        total_orders=('num_orders', 'sum'),
        mean_orders=('num_orders', 'mean'),
        std_orders=('num_orders', 'std'),
        total_revenue=('revenue', 'sum'),
        avg_checkout_price=('checkout_price', 'mean'),
        avg_base_price=('base_price', 'mean')
    ).reset_index()
    
    # حساب معامل التذبذب
    meal_stats['cv'] = meal_stats['std_orders'] / meal_stats['mean_orders']
    
    # ترتيب حسب الحجم الإجمالي وتحليل باريتو
    meal_stats = meal_stats.sort_values(by='total_orders', ascending=False)
    meal_stats['cumulative_orders'] = meal_stats['total_orders'].cumsum()
    meal_stats['cumulative_pct'] = (meal_stats['cumulative_orders'] / meal_stats['total_orders'].sum()) * 100
    
    # تصنيف باريتو
    meal_stats['pareto_class'] = np.where(meal_stats['cumulative_pct'] <= 80, 'Top 80% (Core)', 'Long Tail (20%)')
    
    return meal_stats
```

---

## معايير إخراج النتائج
* تقديم جدول مرتب تنازلياً يجمع بين حجم الطلب ومستوى التذبذب ($CV$).
* تقديم توصية صريحة لفريق سلاسل الإمداد بالوجبات الأكثر خطورة في التلف والنفاذ.
