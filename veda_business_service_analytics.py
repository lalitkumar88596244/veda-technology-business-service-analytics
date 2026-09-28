import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("veda_business_service_data.csv")
df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nKPIs:")
print("Visitors:", df.website_visitors.sum())
print("Internship applications:", df.internship_applications.sum())
print("Training enrollments:", df.training_enrollments.sum())
print("IT inquiries:", df.it_service_inquiries.sum())
print("Revenue:", round(df.revenue.sum(),2))

monthly=df.groupby("date")[["website_visitors","internship_applications","training_enrollments","it_service_inquiries","revenue"]].sum()
monthly.plot(figsize=(11,6),marker="o",title="Monthly Business Metrics")
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

service=df.groupby("service").it_service_inquiries.sum().sort_values(ascending=False)
service.plot(kind="bar",figsize=(9,5),title="IT Service Inquiries")
plt.tight_layout(); plt.show()

segment=df.groupby("customer_type").agg(records=("customer_type","size"),avg_revenue=("revenue","mean"),requests=("customer_service_requests","sum"))
print("\nCustomer segmentation:\n",segment)

sns.heatmap(df.select_dtypes("number").corr(),annot=True,fmt=".2f")
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()

features=["website_visitors","internship_applications","training_enrollments","it_service_inquiries","customer_service_requests"]
X_train,X_test,y_train,y_test=train_test_split(df[features],df.revenue,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=150,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("MAE:",round(mean_absolute_error(y_test,pred),2))
print("R2:",round(r2_score(y_test,pred),3))
