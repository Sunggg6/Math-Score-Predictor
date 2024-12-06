from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# โหลดข้อมูล
data = pd.read_csv("StudentsPerformance.csv")
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# แปลงข้อมูลหมวดหมู่
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# เตรียมข้อมูล
X = data.drop(columns=['math score'])
y = data['math score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# เทรนโมเดล
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# บันทึกโมเดลและ Scaler
joblib.dump(model, "math_score_predictor.pkl")
joblib.dump(scaler, "scaler.pkl")
