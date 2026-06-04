import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for nice plots
sns.set(style='whitegrid')

# 1. Load Data
data_path = 'california_housing_preprocessing/california_housing.csv'
df = pd.read_csv(data_path)

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 2. Train Model to get predictions and metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 3. Create 'screenshoot_artifak.jpg' (Actual vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('screenshoot_artifak.jpg', dpi=300)
print("Saved screenshoot_artifak.jpg")
plt.clf()

# 4. Create 'screenshoot_dashboard.jpg' (Metrics Dashboard)
fig, ax = plt.subplots(figsize=(8, 6))
metrics = ['MSE', 'MAE', 'R2 Score']
values = [mse, mae, r2]
bars = ax.bar(metrics, values, color=['#ff9999','#66b3ff','#99ff99'])

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=12)

ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics (Dashboard)')
ax.set_ylim(0, max(values) + 0.2)
plt.tight_layout()
plt.savefig('screenshoot_dashboard.jpg', dpi=300)
print("Saved screenshoot_dashboard.jpg")
