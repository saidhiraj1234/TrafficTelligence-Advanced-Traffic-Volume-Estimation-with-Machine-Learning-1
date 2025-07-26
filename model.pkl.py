import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv("traffic volume.csv")

# Print columns for confirmation
print("Available columns:", df.columns)

# Combine 'date' and 'Time' into 'date_time' with proper parsing
df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['Time'], dayfirst=True)

# Drop original columns that are now merged
df.drop(['date', 'Time'], axis=1, inplace=True)

# Handle missing values if any
df.dropna(inplace=True)

# Features (X) and target (y)
X = df[['temp', 'rain', 'snow']]  # You can add more features if relevant
y = df['traffic_volume']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'model.pkl'")
