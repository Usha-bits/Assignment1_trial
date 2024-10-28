# To run the file in terminal type > python workflow.py 
# Has to connect with Prefect cloud -> https://app.prefect.cloud/
# Prefect Login [Prefect Cloud] - https://www.prefect.io/opensource  -> get started

# Step 1: Import Required Libraries
import pandas as pd
from prefect import flow, task
from sklearn.preprocessing import MinMaxScaler
import nest_asyncio
nest_asyncio.apply()

# Step 2: Load the Dataset
@task
def load_dataset():
    # Load the dataset
    url = "https://github.com/Usha-bits/Assignment1_trial/blob/main/data/earthquakes.csv"
    column_names = ['magnitude', 'depth', 'magType', 'latitude', 'place', 'country', 'subnational', 'city', 'locality',
                    'date','time','updated','felt','cdi','mmi','tsunami','alert','distanceKM']
    return pd.read_csv(url, names=column_names)

# Step 3: Data Preprocessing
@task(log_prints=True)
def preprocess_data(df):
    # Print columns with missing values and their count
    missing_values = df.isna().sum()
    columns_with_missing = missing_values[missing_values > 0]
    print("Columns with missing values: ")
    print(columns_with_missing)

    # Replace with Median value
    df.fillna(df.median(), inplace=True)

    # Normalize using Min-Max Scaling
    scaler = MinMaxScaler()
    features = df.drop('class', axis=1)  # Exclude the target variable
    df_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_normalized['class'] = df['class']  # Add the target variable back to the dataframe

    # Print the normalized dataframe
    print("Normalized DataFrame:")
    print(df_normalized.head())  # Printing only the first few rows for brevity

    return df_normalized

# Step 4: Model Training
@task
def train_model(df):
    # Train your machine learning model with Logistic Regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Step 5: Define Prefect Flow
@flow(log_prints=True)
def workflow_earthquake_prediction():
    # step 1 = loading data
    data = load_dataset()
    # step 2 = preprocessing
    preprocessed_data = preprocess_data(data)
    # step 3 = data modeling
    accuracy = train_model(preprocessed_data)

    print("Accuracy: ", accuracy)
   
# Step 6: Run the Prefect Flow
if __name__ == "__main__":
    workflow_earthquake_prediction.serve(name="pima-indian-workflow",
                      tags=["first workflow"],
                      parameters={},
                      interval=120)