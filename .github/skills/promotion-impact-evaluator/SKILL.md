---
name: promotion-impact-evaluator
description: مهارة إحصائية متقدمة لتقييم واختبار الأثر الفعلي لحملات البريد الترويجي والظهور في الصفحة الرئيسية والخصومات السعرية على حجم الطلب، باستخدام الاختبارات اللابارامترية وعزل المتغيرات.
---

# مهارة تقييم أثر العروض والترويج (Promotion Impact Evaluator)

## الهدف
تمكين المحلل والنموذج من قياس الفرق الحقيقي في حجم الطلب الناتج عن:
1. البريد الترويجي (`emailer_for_promotion`).
2. الظهور في الصفحة الرئيسية (`homepage_featured`).
3. الخصم السعري (`discount_percent = (base_price - checkout_price) / base_price`).

مع التحقق الصارم من الدلالة الإحصائية وعزل أثر التداخل بين الخصومات والعروض.

---

## المنهجية الإحصائية (Methodology)

### 1. المقارنة الزوجية في نفس السياق (Within Center-Meal Control)
* **المشكلة:** المقارنة الإجمالية لجميع الوجبات عبر جميع المراكز متحيزة، لأن المراكز الكبيرة والوجبات الشائعة تطلب أكثر بطبيعتها.
* **الحل الإحصائي:** مقارنة نفس الوجبة (`meal_id`) داخل نفس المركز (`center_id`) في الأسابيع التي شملها العرض مقابل الأسابيع التي لم يشملها العرض.

### 2. اختيار الاختبار الإحصائي المناسب (Mann-Whitney U Test)
* **فحص التوزيع:** توزيع الطلبات (`num_orders`) ذو التواء إيجابي شديد (Right-Skewed with long tail) ولا يتبع التوزيع الطبيعي.
* **الاختبار المعتمد:** استخدام **Mann-Whitney U Test** (Wilcoxon Rank-Sum Test) اللابارامتري لمقارنة الوسيط ورتب الطلبات بين مجموعتي:
  * المجموعة الأولى: $A$ (بدون عرض: `promo = 0`).
  * المجموعة الثانية: $B$ (مع عرض: `promo = 1`).
* **مستوى الدلالة:** $\alpha = 0.05$.
* **حساب الأثر النسبي (% Uplift):**
  $$\text{Relative Lift (\%)} = \frac{\text{Median}(B) - \text{Median}(A)}{\text{Median}(A)} \times 100$$
  أو بمقارنة المتوسطات مع التنويه للأثر اللابارامتري.

### 3. تحليل حساسية الخصم (Spearman Rank Correlation)
* استخدام معامل ارتباط سبيرمان للرتب ($r_s$) بين نسبة الخصم وحجم الطلب:
  $$r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
* التنويه الإلزامي: التحقق مما إذا كانت العروض تأتي مقترنة بخصومات سعرية كبيرة لبيان أثر التداخل (Collinearity).

---

## الكود البرمجي المرجعي (Python Implementation)

```python
from scipy import stats
import pandas as pd
import numpy as np

def evaluate_promotion_impact(df, promo_col='homepage_featured'):
    """
    يقيم أثر العرض الترويجي إحصائياً لكل وجبة ومركز، وبشكل إجمالي
    """
    # عزل الحالات مع العرض وبدونه
    with_promo = df[df[promo_col] == 1]['num_orders']
    without_promo = df[df[promo_col] == 0]['num_orders']
    
    # اختبار مان ويتني
    stat, p_val = stats.mannwhitneyu(with_promo, without_promo, alternative='greater')
    
    # حساب المتوسط والوسيط
    mean_with = with_promo.mean()
    mean_without = without_promo.mean()
    median_with = with_promo.median()
    median_without = without_promo.median()
    
    lift_pct = ((mean_with - mean_without) / mean_without) * 100
    
    return {
        'metric': promo_col,
        'u_statistic': stat,
        'p_value': p_val,
        'is_statistically_significant': p_val < 0.05,
        'mean_with_promo': mean_with,
        'mean_without_promo': mean_without,
        'median_with_promo': median_with,
        'median_without_promo': median_without,
        'percentage_lift': lift_pct
    }
```

---

## معايير إخراج النتائج للعميل
1. ذكر القيمة الاحتمالية ($p\text{-value}$) وقرار رفض أو قبول الفرضية الصفرية.
2. ذكر مقدار الارتفاع بالنسبة المئوية ومعدل الطلب الإضافي لكل أسبوع.
3. التأكيد الصارم على تجنب لفظ «يسبب» واعتماد لفظ «يقترن بـ / يرتبط بزيادة بمقدار...».
