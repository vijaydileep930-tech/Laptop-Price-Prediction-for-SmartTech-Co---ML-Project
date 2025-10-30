
# laptop_price_pipeline.py
# Reproducible pipeline for Laptop Price Prediction (run in Colab or local)
import pandas as pd, numpy as np, re, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def extract_numeric(val):
    try:
        s = str(val); m = re.search(r"[\\d.]+", s)
        return float(m.group(0)) if m else np.nan
    except: return np.nan

def parse_resolution(res):
    s = str(res)
    touchscreen = 1 if "Touchscreen" in s or "touchscreen" in s else 0
    ips = 1 if "IPS" in s or "ips" in s else 0
    m = re.search(r"(\\d{3,4})\\s*x\\s*(\\d{3,4})", s)
    if not m: m = re.search(r"(\\d{3,4})x(\\d{3,4})", s)
    if m: x = int(m.group(1)); y = int(m.group(2))
    else: x = np.nan; y = np.nan
    return touchscreen, ips, x, y

def cpu_brand(cpu):
    s = str(cpu); tokens = s.split(); return tokens[0] if len(tokens)>0 else "Unknown"

def parse_memory(mem):
    s = str(mem); parts = re.split(r"\\s*\\+\\s*", s); total=0.0; ssd=0; hdd=0; flash=0; eMMC=0
    for p in parts:
        m_tb = re.search(r"(\\d+\\.?\\d*)\\s*TB", p, re.IGNORECASE)
        m_gb = re.search(r"(\\d+\\.?\\d*)\\s*GB", p, re.IGNORECASE)
        if m_tb: gb = float(m_tb.group(1))*1024
        elif m_gb: gb = float(m_gb.group(1))
        else: gb = 0.0
        total += gb
        if re.search(r"ssd", p, re.IGNORECASE): ssd += gb
        if re.search(r"hdd", p, re.IGNORECASE): hdd += gb
        if re.search(r"flash", p, re.IGNORECASE): flash += gb
        if re.search(r"emmc", p, re.IGNORECASE): eMMC += gb
    return total, ssd, hdd, flash, eMMC

def opsys_group(x):
    s = str(x)
    if "Windows" in s or "windows" in s: return "Windows"
    if "macOS" in s or "Mac OS" in s or "mac" in s: return "macOS"
    if "Linux" in s or "linux" in s: return "Linux"
    if "No OS" in s or "NoOS" in s or "no os" in s: return "No OS"
    return "Other"

# Load dataset (update path as needed)
df = pd.read_csv("laptop.csv")

# Rename price
price_col = [c for c in df.columns if "price" in c.lower()]
if len(price_col)==0: raise Exception("No price column found")
df = df.rename(columns={price_col[0]: "Price"})

# Feature engineering
df['Inches'] = df['Inches'].apply(extract_numeric)
df[['Touchscreen','IPS','X_res','Y_res']] = df['ScreenResolution'].apply(lambda r: pd.Series(parse_resolution(r)))
df['PPI'] = ((df['X_res']**2 + df['Y_res']**2)**0.5) / df['Inches']
df['Ram'] = df['Ram'].astype(str).str.replace('GB','',regex=False)
df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')
df['Weight'] = df['Weight'].astype(str).str.replace('kg','',regex=False).str.replace('kgs','',regex=False)
df['Weight'] = df['Weight'].replace('?', np.nan)
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Cpu_brand'] = df['Cpu'].apply(cpu_brand)
df['Gpu_brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])
df[['Storage_GB','SSD_GB','HDD_GB','Flash_GB','eMMC_GB']] = df['Memory'].apply(lambda m: pd.Series(parse_memory(m)))
df['OpSys_group'] = df['OpSys'].apply(opsys_group)

# Drop unused columns and rows missing critical features
df_model = df.drop(columns=['ScreenResolution','Cpu','Memory','Gpu','OpSys'])
df_model = df_model.dropna(subset=['Price','Inches','X_res','Y_res','Ram','Storage_GB'])

X = df_model.drop(columns=['Price']); y = df_model['Price'].astype(float)

numeric_features = ['Inches','Ram','Weight','PPI','Storage_GB','SSD_GB','HDD_GB','Flash_GB','eMMC_GB','X_res','Y_res']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_features = ['Company','TypeName','Cpu_brand','Gpu_brand','OpSys_group']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
pipeline_lr = Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
pipeline_rf = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(random_state=42, n_jobs=1))])

pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Evaluate
for name, model in [("LinearRegression", pipeline_lr), ("RandomForest", pipeline_rf)]:
    preds = model.predict(X_test)
    print(name, "RMSE:", mean_squared_error(y_test, preds, squared=False), "MAE:", mean_absolute_error(y_test, preds), "R2:", r2_score(y_test, preds))

# Save best model (choose based on RMSE)
best_model = pipeline_rf
joblib.dump(best_model, "best_laptop_price_model.pkl")
print("Saved model to best_laptop_price_model.pkl")
