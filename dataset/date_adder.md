# توضیحات کد افزودن ستون‌های تاریخ حفاری به دیتاست‌های FDMS

---

## هدف کد
این اسکریپت برای افزودن دو ستون تاریخ به دیتاست‌های حفاری چاه‌ها در قالب فایل‌های Parquet استفاده می‌شود:

- **spud_date**: تاریخ شروع حفاری (شروع عملیات چاه)
- **completion_date**: تاریخ پایان حفاری (اتمام عملیات چاه)

---

## ورودی‌ها
- مسیر ورودی فایل‌ها در متغیر `input_dir` مشخص شده است (`fdms_well_datasets`).
- فایل‌های ورودی باید فرمت Parquet داشته باشند.

---

## شرح عملکرد کد

### 1. تولید تاریخ‌ها

- تابع `generate_dates(num_rows)` برای تولید دو لیست تاریخ استفاده می‌شود:

  - **spud_date**: تاریخ شروع حفاری که به صورت تصادفی در بازه 3 سال از 1 ژانویه 2020 تولید می‌شود.
  
  - **completion_date**: تاریخ پایان حفاری که بین ۵ تا ۹۰ روز پس از `spud_date` قرار دارد.

### 2. خواندن و پردازش فایل‌ها

- اسکریپت وارد پوشه `fdms_well_datasets` شده و تمام فایل‌های با پسوند `.parquet` را شناسایی می‌کند.
- برای هر فایل:
  - فایل را با `pandas.read_parquet` به یک دیتافریم تبدیل می‌کند.
  - تاریخ‌های `spud_date` و `completion_date` را تولید و به دیتافریم اضافه می‌کند.
  - ستون‌ها را مرتب می‌کند تا ترتیب به صورت زیر باشد:
    1. `WELL_ID`
    2. `spud_date`
    3. `completion_date`
    4. سایر ستون‌ها
  - دیتافریم به همان مسیر و فرمت Parquet ذخیره می‌شود.

---

## نکات مهم

- تاریخ‌های تولید شده به صورت تصادفی هستند و در هر اجرای کد تغییر می‌کنند.
- اگر نیاز به تکرارپذیری دارید، می‌توانید قبل از تولید تاریخ‌ها از `np.random.seed()` استفاده کنید.
- مرتب کردن ستون‌ها کمک می‌کند تا اطلاعات کلیدی راحت‌تر در دسترس باشند.
- کد فایل‌های اصلی را با نسخه جدید بازنویسی می‌کند.

---

## نمونه کد اصلی

```python
def generate_dates(num_rows):
    base_date = datetime(2020, 1, 1)
    spud_dates = [base_date + timedelta(days=int(np.random.uniform(0, 365*3))) for _ in range(num_rows)]
    completion_dates = [spud + timedelta(days=int(np.random.uniform(5, 90))) for spud in spud_dates]
    return spud_dates, completion_dates

for filename in os.listdir(input_dir):
    if filename.endswith(".parquet"):
        filepath = os.path.join(input_dir, filename)
        print(f"📂 در حال پردازش فایل: {filename}")

        df = pd.read_parquet(filepath)

        spud_dates, completion_dates = generate_dates(len(df))
        df["spud_date"] = spud_dates
        df["completion_date"] = completion_dates

        first_cols = ["WELL_ID", "spud_date", "completion_date"]
        remaining_cols = [col for col in df.columns if col not in first_cols]
        df = df[first_cols + remaining_cols]

        df.to_parquet(filepath, index=False)
        print(f"✅ ستون‌ها افزوده و فایل ذخیره شد: {filename}")
