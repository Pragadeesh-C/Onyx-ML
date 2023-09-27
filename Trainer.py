import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Load the dataset
data = pd.read_csv("Trainer.csv")

# Preprocessing
# One-hot encode the "category" column

data.replace({"category":{'gain':0,'loss':1,'fit':2}}, inplace=True)

# Split data into features (X) and target variables (y)
X = data.drop(columns=["set", "reps", "weights", "Duration"])
y = data[["set", "reps", "weights", "Duration"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

pickle.dump(model,open('Trainer.pkl','wb'))
Trainer=pickle.load(open('Trainer.pkl','rb'))