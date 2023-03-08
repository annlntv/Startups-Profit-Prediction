import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Startups.csv")
print(df.head())

print(df.describe())

sns.heatmap(df.corr(), annot=True)
plt.show()

x = df[["R&D Spend", "Administration", "Marketing Spend"]]
y = df["Profit"]

x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
df = pd.DataFrame(data={"Predicted Profit": ypred.flatten()})
print(df.head())