import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Numpy

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("")

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr_2d)
print("")

print("Sum of elements:", np.sum(arr))
print("")

print("Mean of elements:", np.mean(arr))
print("")

print("Standard deviation of elements:", np.std(arr))
print("")


print("First two elements:", arr[:2])
print("")

print("Elements from index 2:", arr[2:])
print("")


# Pandas

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

print("DataFrame:\n", df)
print("")
print("Age column:\n", df['Age'])
print("")

filtered_df = df[df['Age'] > 25]

print("Filtered DataFrame:\n", filtered_df)
print("")

df['Salary'] = [50000, 60000, 70000]

print("DataFrame with Salary:\n", df)
print("")

print("Descriptive statistics:\n", df.describe())
print("")


# Matplotlib.pyplot

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.legend()
plt.show()

x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, c='blue', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.show()

categories = ['A', 'B', 'C']
values = [1, 2, 3]
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()

# Seaborn

tips = sns.load_dataset('tips')

sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Total Bill vs Tip')
plt.show()

sns.histplot(tips['total_bill'], bins=20, kde=True)
plt.title('Histogram of Total Bill')
plt.show()

sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Box Plot of Total Bill by Day')
plt.show()


numeric_tips = tips.select_dtypes(include=[np.number])
corr = numeric_tips.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Sklearn

data = {'feature1': [10, 20, 30, 40, 50],
        'feature2': [100, 200, 300, 400, 500]}
df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print("Scaled data:\n", scaled_data)
print("")

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)