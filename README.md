# ecommerce_linear_reg_with_scikit_learn
# Customer Spending Prediction using Linear Regression

This project uses a **Linear Regression** model to predict the **Yearly Amount Spent** by customers in an e-commerce platform. The dataset contains customer usage metrics such as session length, app usage time, website usage time, and membership duration. The goal is to understand which features contribute most to customer spending and to build a predictive model.

---

## 📂 Dataset Description

The dataset used is **`ecommerce_customers.csv`**, which includes features such as:

| Feature Name            | Description |
|------------------------|-------------|
| Avg. Session Length     | Average session time of a user |
| Time on App             | Average time spent on the mobile app |
| Time on Website         | Average time spent on the website |
| Length of Membership    | How long the customer has been a member |
| Yearly Amount Spent     | **Target variable** – amount spent per year |

---

## 🧪 Project Workflow

1. **Import Dependencies**
pandas, seaborn, matplotlib, sklearn.linear_model.LinearRegression

2. **Load Data**
df = pd.read_csv("ecommerce_customers.csv")
Exploratory Data Analysis (EDA)

df.info() and df.describe()

Jointplots and pairplots to view relationships

Best relationship observed:
Length of Membership ↔ Yearly Amount Spent

sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df)
Feature Selection

X = df[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]
y = df["Yearly Amount Spent"]
Train/Test Split

python
Copy code
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Model Training

python
Copy code
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

predictions = lm.predict(x_test)
Residual Analysis

residuals = y_test - predictions
sns.displot(residuals)
📊 Visualizations Used
Plot	Purpose
Jointplot	Understand relationship between two individual variables
Pairplot	See general correlation structures
LM Plot	Visualize linear relationships
Residual Distribution	Check model error patterns

Example scatterplot of predictions vs actual values:

python
Copy code
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
✅ Model Evaluation
Interpret the learned coefficients:

python
Copy code
lm.coef_
These values show how much each feature impacts spending.

Evaluate error distribution to confirm if linear regression is appropriate.

🧱 Project Structure

/ecommerce_linear_reg_with_scikit_learn
│── main.ipynb                 # Notebook with full workflow
│── ecommerce_customers.csv    # Dataset (optional if private)
└── README.md                  # This documentation
🔧 Requirements Installation

pip install pandas seaborn matplotlib scikit-learn
Generate full requirements file:

pip freeze > requirements.txt
🎯 Conclusion
Length of Membership is the strongest predictor of yearly spending.

The model can reliably estimate customer value.

Business can use these insights to:

Improve customer retention strategies

Focus on enhancing membership benefits

🤝 Contributions
Pull requests are welcome.
