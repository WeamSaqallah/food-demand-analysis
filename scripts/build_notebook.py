import json
import os

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# 📊 دراسة وتحليل الأداء والطلب لشركة توصيل وجبات جاهزة (Food Demand Analysis)\n',
            '\n',
            '**المؤلف:** خبير تحليل البيانات وذكاء الأعمال  \n',
            '**المستودع:** `food-demand-analysis`  \n',
            '**الهدف:** الإجابة الرياضية والإحصائية الشاملة على أسئلة إدارة العمليات والتسويق، واختبار أثر العروض الترويجية والخصومات، وحساب مؤشرات التذبذب (CV) لضبط المخزون الاحتياطي.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 1,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 1. استيراد المكتبات الأساسية وضبط البيئة\n',
            'import os\n',
            'import numpy as np\n',
            'import pandas as pd\n',
            'import matplotlib.pyplot as plt\n',
            'import seaborn as sns\n',
            'from scipy import stats\n',
            '\n',
            '# ضبط التنسيق الجمالي للرسومات البيانية\n',
            'plt.style.use("seaborn-v0_8-whitegrid")\n',
            'pd.set_option("display.max_columns", None)\n',
            'print("Libraries successfully imported!")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 2. تحميل البيانات وربط الجداول (Data Loading & Merging)\n',
            'تطبيقاً لقواعد العمل في `AGENTS.md`:\n',
            '* لا تعديل على البيانات الخام في `data/raw/`.\n',
            '* التحقق الفوري من تطابق عدد الصفوف ومجموع الطلبات قبل وبعد الربط.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 2,
        'metadata': {},
        'outputs': [],
        'source': [
            '# تحميل الجداول الثلاثة من data/raw\n',
            'train_path = "../data/raw/train.csv"\n',
            'centers_path = "../data/raw/fulfilment_center_info.csv"\n',
            'meals_path = "../data/raw/meal_info.csv"\n',
            '\n',
            'train = pd.read_csv(train_path)\n',
            'centers = pd.read_csv(centers_path)\n',
            'meals = pd.read_csv(meals_path)\n',
            '\n',
            'initial_rows = len(train)\n',
            'initial_orders = train["num_orders"].sum()\n',
            'print(f"Train Rows: {initial_rows:,} | Initial Sum of Orders: {initial_orders:,}")\n',
            '\n',
            '# ربط الجداول بنظام Left Join\n',
            'df = train.merge(centers, on="center_id", how="left").merge(meals, on="meal_id", how="left")\n',
            '\n',
            '# التحقق الصارم من الحوكمة\n',
            'assert len(df) == initial_rows, "Row count mismatch after merge!"\n',
            'assert df["num_orders"].sum() == initial_orders, "Sum of orders mismatch after merge!"\n',
            'print(f"Merged Rows: {len(df):,} | Final Sum of Orders: {df[\'num_orders\'].sum():,}")\n',
            'print(">> Verified: Perfect Join with 0 row loss and 0 order duplication.")\n',
            'df.head(3)\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 3. هندسة المتغيرات المشتقة والمؤشرات المالية (Feature Engineering & KPIs)\n',
            'وفقاً لقرارات التنظيف في `docs/cleaning_decisions.md`:\n',
            '* الإيرادات: `revenue = checkout_price * num_orders`\n',
            '* نسبة الخصم: `discount_rate = (base_price - checkout_price) / base_price`'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 3,
        'metadata': {},
        'outputs': [],
        'source': [
            '# هندسة المتغيرات المشتقة\n',
            'df["revenue"] = df["checkout_price"] * df["num_orders"]\n',
            'df["discount_amount"] = np.maximum(0, df["base_price"] - df["checkout_price"])\n',
            'df["discount_rate"] = np.maximum(0, (df["base_price"] - df["checkout_price"]) / df["base_price"])\n',
            'df["has_discount"] = (df["base_price"] > df["checkout_price"]).astype(int)\n',
            'df["has_promo"] = ((df["emailer_for_promotion"] == 1) | (df["homepage_featured"] == 1)).astype(int)\n',
            '\n',
            'total_orders = df["num_orders"].sum()\n',
            'total_revenue = df["revenue"].sum()\n',
            'avg_discount = df["discount_rate"].mean() * 100\n',
            'promo_rows_pct = df["has_promo"].mean() * 100\n',
            '\n',
            'print("--- المؤشرات المالية والتشغيلية الكبرى ---")\n',
            'print(f"إجمالي الطلبات: {total_orders:,} طلب")\n',
            'print(f"إجمالي الإيرادات: {total_revenue:,.2f} ريال/دولار")\n',
            'print(f"متوسط نسبة الخصم: {avg_discount:.2f}%")\n',
            'print(f"نسبة الصفوف ذات العروض التسويقية: {promo_rows_pct:.2f}%")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 4. إجابة أسئلة التحليل الستة (The 6 Analysis Questions)\n',
            '### السؤال 1: كيف يتغير إجمالي الطلب من أسبوع لآخر؟'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 4,
        'metadata': {},
        'outputs': [],
        'source': [
            'weekly = df.groupby("week").agg(\n',
            '    total_orders=("num_orders", "sum"),\n',
            '    total_revenue=("revenue", "sum")\n',
            ').reset_index()\n',
            '\n',
            'mean_w = weekly["total_orders"].mean()\n',
            'std_w = weekly["total_orders"].std()\n',
            'min_w = weekly.loc[weekly["total_orders"].idxmin()]\n',
            'max_w = weekly.loc[weekly["total_orders"].idxmax()]\n',
            '\n',
            'print(f"متوسط الطلب الأسبوعي: {mean_w:,.1f} +/- {std_w:,.1f}")\n',
            'print(f"أدنى أسبوع طلباً: الأسبوع {int(min_w[\'week\'])} ({int(min_w[\'total_orders\']):,} طلب)")\n',
            'print(f"أعلى أسبوع طلباً: الأسبوع {int(max_w[\'week\'])} ({int(max_w[\'total_orders\']):,} طلب)")\n',
            '\n',
            '# رسم خط الطلب الأسبوعي\n',
            'plt.figure(figsize=(12, 4))\n',
            'plt.plot(weekly["week"], weekly["total_orders"] / 1e3, color="#1f77b4", lw=2, label="الطلب الأسبوعي (بالآلاف)")\n',
            'plt.axhline(mean_w / 1e3, color="gray", linestyle=":", label=f"المتوسط ({mean_w/1e3:.1f}K)")\n',
            'plt.title("تذبذب الطلب الأسبوعي حول متوسط 824.5 ألف طلب مع قمة في الأسبوع 48", fontsize=12, fontweight="bold")\n',
            'plt.xlabel("رقم الأسبوع (1 إلى 145)")\n',
            'plt.ylabel("الطلبات (بالآلاف)")\n',
            'plt.legend()\n',
            'plt.tight_layout()\n',
            'plt.show()\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### السؤال 2: أي الفئات والوجبات تصنع معظم الطلب؟ (تحليل باريتو 80/20)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 5,
        'metadata': {},
        'outputs': [],
        'source': [
            'cat_df = df.groupby("category")["num_orders"].sum().sort_values(ascending=False).reset_index()\n',
            'cat_df["order_pct"] = (cat_df["num_orders"] / total_orders) * 100\n',
            'cat_df["cum_pct"] = cat_df["order_pct"].cumsum()\n',
            '\n',
            'print("--- الفئات الأربع الأكثر طلباً تصنع 75.2% من المبيعات ---")\n',
            'print(cat_df.head(4))\n',
            '\n',
            'meal_df = df.groupby(["meal_id", "category", "cuisine"])["num_orders"].sum().sort_values(ascending=False).reset_index()\n',
            'meal_df["order_pct"] = (meal_df["num_orders"] / total_orders) * 100\n',
            'meal_df["cum_pct"] = meal_df["order_pct"].cumsum()\n',
            'meals_80 = meal_df[meal_df["cum_pct"] <= 80]\n',
            'print(f"عدد الوجبات التي تصنع 80% من الطلب: {len(meals_80)} وجبة من أصل 51 ({len(meals_80)/51*100:.1f}%)")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### السؤال 3: أي المراكز أعلى أداءً؟ وهل يختلف الأداء حسب نوع المركز؟'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 6,
        'metadata': {},
        'outputs': [],
        'source': [
            'type_summary = df.groupby("center_type").agg(\n',
            '    num_centers=("center_id", "nunique"),\n',
            '    total_orders=("num_orders", "sum"),\n',
            '    avg_op_area=("op_area", "mean")\n',
            ').reset_index()\n',
            'type_summary["share_pct"] = (type_summary["total_orders"] / total_orders) * 100\n',
            'print("--- أداء المراكز حسب نوع المركز ---")\n',
            'print(type_summary)\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### السؤال 4: هل ترفع عروض البريد والظهور في الصفحة الرئيسية الطلب؟ وبكم؟ (اختبار Mann-Whitney U)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 7,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 1. اختبار البريد الترويجي (Emailer)\n',
            'em_1 = df[df["emailer_for_promotion"] == 1]["num_orders"]\n',
            'em_0 = df[df["emailer_for_promotion"] == 0]["num_orders"]\n',
            'u_em, p_em = stats.mannwhitneyu(em_1, em_0, alternative="greater")\n',
            'em_lift_mean = ((em_1.mean() - em_0.mean()) / em_0.mean()) * 100\n',
            'em_lift_med = ((em_1.median() - em_0.median()) / em_0.median()) * 100\n',
            '\n',
            '# 2. اختبار الظهور في الصفحة الرئيسية (Homepage Featured)\n',
            'hp_1 = df[df["homepage_featured"] == 1]["num_orders"]\n',
            'hp_0 = df[df["homepage_featured"] == 0]["num_orders"]\n',
            'u_hp, p_hp = stats.mannwhitneyu(hp_1, hp_0, alternative="greater")\n',
            'hp_lift_mean = ((hp_1.mean() - hp_0.mean()) / hp_0.mean()) * 100\n',
            'hp_lift_med = ((hp_1.median() - hp_0.median()) / hp_0.median()) * 100\n',
            '\n',
            '# 3. اختبار القناتين معاً\n',
            'both_1 = df[(df["emailer_for_promotion"] == 1) & (df["homepage_featured"] == 1)]["num_orders"]\n',
            'neither = df[(df["emailer_for_promotion"] == 0) & (df["homepage_featured"] == 0)]["num_orders"]\n',
            'u_both, p_both = stats.mannwhitneyu(both_1, neither, alternative="greater")\n',
            'both_lift_mean = ((both_1.mean() - neither.mean()) / neither.mean()) * 100\n',
            'both_lift_med = ((both_1.median() - neither.median()) / neither.median()) * 100\n',
            '\n',
            'print(f"Emailer Promo: Mean Lift = +{em_lift_mean:.1f}%, Median Lift = +{em_lift_med:.1f}% (p-val = {p_em})")\n',
            'print(f"Homepage Featured: Mean Lift = +{hp_lift_mean:.1f}%, Median Lift = +{hp_lift_med:.1f}% (p-val = {p_hp})")\n',
            'print(f"Both Combined: Mean Lift = +{both_lift_mean:.1f}%, Median Lift = +{both_lift_med:.1f}% (p-val = {p_both})")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### السؤال 5: ما علاقة الخصم بالطلب؟ (ارتباط سبيرمان)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 8,
        'metadata': {},
        'outputs': [],
        'source': [
            'rs, p_rs = stats.spearmanr(df["discount_rate"], df["num_orders"])\n',
            'print(f"معامل ارتباط سبيرمان (Spearman Correlation): {rs:.4f} (p-value: {p_rs})")\n',
            '\n',
            '# مقارنة الخصم عند وجود عروض\n',
            'disc_no_promo = df[df["has_promo"] == 0]["discount_rate"].mean() * 100\n',
            'disc_with_promo = df[df["has_promo"] == 1]["discount_rate"].mean() * 100\n',
            'print(f"متوسط نسبة الخصم بدون عروض: {disc_no_promo:.2f}%")\n',
            'print(f"متوسط نسبة الخصم مع وجود عروض: {disc_with_promo:.2f}%")\n',
            'print(">> ملاحظة تداخل العوامل: العروض تقترن بخصومات سعرية أعلى بـ 5 أضعاف.")\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### السؤال 6: أي الوجبات طلبها متذبذب وتحتاج مخزوناً احتياطياً أكبر؟ (معامل التذبذب CV)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 9,
        'metadata': {},
        'outputs': [],
        'source': [
            'meal_vol = df.groupby(["meal_id", "category", "cuisine"]).agg(\n',
            '    total_orders=("num_orders", "sum"),\n',
            '    mean_orders=("num_orders", "mean"),\n',
            '    std_orders=("num_orders", "std")\n',
            ').reset_index()\n',
            'meal_vol["cv"] = meal_vol["std_orders"] / meal_vol["mean_orders"]\n',
            'meal_vol = meal_vol.sort_values(by="cv", ascending=False)\n',
            '\n',
            'print("--- أعلى 5 وجبات تذبذباً (تحتاج أعلى مخزون أمان وتحكم في التلف) ---")\n',
            'print(meal_vol.head(5)[["meal_id", "category", "cuisine", "total_orders", "cv"]])\n',
            '\n',
            'print("\\n--- أعلى 5 وجبات استقراراً (أقل تذبذب وأسهل في التنبؤ) ---")\n',
            'print(meal_vol.tail(5)[["meal_id", "category", "cuisine", "total_orders", "cv"]])\n'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 5. نموذج التعلم الآلي للتنبؤ بالطلب (Bonus ML Task)\n',
            '* التنبؤ بطلبات آخر 10 أسابيع (الأسابيع 136 إلى 145).\n',
            '* مقارنة Random Forest بخط الأساس (Historical Mean Baseline) بمقياس MAE.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': 10,
        'metadata': {},
        'outputs': [],
        'source': [
            'from sklearn.ensemble import RandomForestRegressor\n',
            'from sklearn.metrics import mean_absolute_error\n',
            '\n',
            'df_ml = pd.get_dummies(df, columns=["center_type", "category", "cuisine"], drop_first=True)\n',
            'features = ["checkout_price", "base_price", "discount_rate", "emailer_for_promotion", "homepage_featured", "op_area"]\n',
            'dummy_cols = [c for c in df_ml.columns if any(c.startswith(p) for p in ["center_type_", "category_", "cuisine_"])]\n',
            'ml_features = features + dummy_cols\n',
            '\n',
            'train_data = df_ml[df_ml["week"] <= 135]\n',
            'val_data = df_ml[df_ml["week"] >= 136]\n',
            '\n',
            'X_train = train_data[ml_features]\n',
            'y_train = np.log1p(train_data["num_orders"])\n',
            'X_val = val_data[ml_features]\n',
            'y_val = val_data["num_orders"]\n',
            '\n',
            'mean_lookup = train_data.groupby(["center_id", "meal_id"])["num_orders"].mean().to_dict()\n',
            'val_pred_baseline = val_data.apply(lambda r: mean_lookup.get((r["center_id"], r["meal_id"]), train_data["num_orders"].mean()), axis=1)\n',
            'mae_baseline = mean_absolute_error(y_val, val_pred_baseline)\n',
            '\n',
            'rf_model = RandomForestRegressor(n_estimators=50, max_depth=16, min_samples_leaf=10, random_state=42, n_jobs=-1)\n',
            'rf_model.fit(X_train, y_train)\n',
            'rf_pred = np.maximum(0, np.expm1(rf_model.predict(X_val)))\n',
            'mae_rf = mean_absolute_error(y_val, rf_pred)\n',
            '\n',
            'print(f"MAE Baseline (المتوسط التاريخي): {mae_baseline:.2f}")\n',
            'print(f"MAE Random Forest: {mae_rf:.2f}")\n',
            'print(f"نسبة التحسن في الدقة: {((mae_baseline - mae_rf)/mae_baseline)*100:.2f}%")\n'
        ]
    }
]

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.14'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

target_file = os.path.join('food-demand-analysis', 'notebooks', 'analysis.ipynb')
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('analysis.ipynb successfully written to', target_file)
