import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from torch import nn, optim

# Load data
data = pd.read_csv("Instagram.csv", encoding='latin1')
print(data.head())

# Preprocess data
data = data.dropna()
X = data[['Impressions']]
y = data['Likes']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear regression using sklearn
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_sklearn = regressor.predict(X_test)

# Neural Network using Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, verbose=0)

# Predictions using the trained neural network
y_pred_keras = model.predict(X_test).flatten()

# Display metrics
print("Sklearn Linear Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_sklearn))
print("Keras Neural Network Mean Squared Error:", mean_squared_error(y_test, y_pred_keras))

# PyTorch Linear Regression
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Initialize PyTorch model, loss function, and optimizer
input_size = 1
output_size = 1
pytorch_model = LinearRegressionModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(pytorch_model.parameters(), lr=0.01)

# Training the PyTorch model
num_epochs = 100
for epoch in range(num_epochs):
    inputs = X_train_tensor
    labels = y_train_tensor

    optimizer.zero_grad()
    outputs = pytorch_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Convert test data to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Make predictions using the trained PyTorch model
y_pred_pytorch_tensor = pytorch_model(X_test_tensor).detach().numpy()

# Display metrics for PyTorch Linear Regression
print("PyTorch Linear Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_pytorch_tensor))

# Data Mining (Example: Finding Most Frequent Words in Captions)
text_captions = " ".join(i for i in data.Caption.dropna())
wordcloud_captions = WordCloud(stopwords=set(STOPWORDS), background_color="white").generate(text_captions)

# Display WordCloud
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud_captions, interpolation='bilinear')
plt.axis("off")
plt.show()
