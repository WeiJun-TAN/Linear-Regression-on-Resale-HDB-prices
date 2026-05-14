import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np  # Ensure numpy is imported
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Step 1: Load the CSV file
file_path = r'new_2020_housePrice.csv'
# file_path = r'Resale flat prices based on registration date from Jan-2017 onwards.csv'
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing

# Convert "remaining_lease" to numeric (in years) to represent the age of the flat
df['remaining_lease_years'] = df['remaining_lease'].apply(lambda x: int(x.split(' ')[0]))

# Drop unnecessary columns and handle missing values
df = df.dropna(subset=['resale_price'])  # Drop rows with missing target variable
df = df.drop(columns=['month'])  # Drop 'month' column, since it's not needed

# Apply log transformation to 'resale_price' to stabilize variance and handle skewness
df['log_resale_price'] = np.log1p(df['resale_price'])  # Log transformation

# 1. **Location**: 'town' as the location feature
# One-hot encode 'town' to create dummy variables
df = pd.get_dummies(df, columns=['town'], drop_first=True)

# 2. **Flat-related features**: Combine 'flat_type' and 'flat_model'
df['flat_type_model'] = df['flat_type'] + "_" + df['flat_model']
df = df.drop(columns=['flat_type', 'flat_model'])  # Drop the original columns for flat_type and flat_model

# One-hot encode the 'flat_type_model' column
df = pd.get_dummies(df, columns=['flat_type_model'], drop_first=True)

# 3. **Flat details**: 'floor_area_sqm' as the size of the flat
# 4. **Age**: 'remaining_lease_years' as the age of the flat
df = df.drop(columns=['remaining_lease'])  # Drop original 'remaining_lease' since we have 'remaining_lease_years'

# Features and target variable
X = df.drop(columns=['resale_price', 'log_resale_price'])  # Drop both 'resale_price' and 'log_resale_price' from features
y = df['log_resale_price']  # Set the target to 'log_resale_price'

# Step 3: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the entire dataset

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Step 4: Define the Linear Regression Model (Single Linear Layer)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        # Single linear layer (no hidden layers)
        self.fc1 = nn.Linear(input_dim, 1)  # Input -> Output (Linear regression)

    def forward(self, x):
        # No activation function (linear output)
        return self.fc1(x)

# Step 5: Initialize the Model
model = LinearRegressionModel(input_dim=X_train_tensor.shape[1])

# Step 6: Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Using Stochastic Gradient Descent for linear regression

# Step 7: Training the Model
epochs = 1000
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()  # Zero the gradients
    predictions = model(X_train_tensor).squeeze()  # Forward pass
    loss = criterion(predictions, y_train_tensor)  # Calculate loss
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model weights

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Step 8: Evaluating the Model
model.eval()
with torch.no_grad():
    y_pred_log = model(X_test_tensor).squeeze().numpy()  # log space
y_actual_log = y_test_tensor.numpy()                     # log space

# 关键改动:用 expm1 把 log 空间还原到 S$ 尺度,再算指标
y_pred_real = np.expm1(y_pred_log)
y_actual_real = np.expm1(y_actual_log)

# 现在在原始 S$ 尺度上算 RMSE / MAE / R²
rmse = np.sqrt(mean_squared_error(y_actual_real, y_pred_real))
mae = mean_absolute_error(y_actual_real, y_pred_real)
r2 = r2_score(y_actual_real, y_pred_real)

print(f'RMSE on test set (S$): {rmse:,.2f}')
print(f'MAE  on test set (S$): {mae:,.2f}')
print(f'R² Score:              {r2:.4f}')
