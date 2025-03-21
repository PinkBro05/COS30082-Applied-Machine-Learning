import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_prediction(y_pred):
    # Visualize the predictions
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_pred, palette='Set2')
    plt.title('Predicted Survival Counts')
    plt.xlabel('Predicted Survival')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Did not survive', 'Survived'])
    plt.show()
def visualize_data(df):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def main():
    # Load the dataset
    df = pd.read_csv('Week_3/Data/Titanic-Dataset.csv')

    # Preprocess the data
    # Encoding categorical data
    sex_encoder = LabelEncoder()
    name_encoder = LabelEncoder()
    ticket_encoder = LabelEncoder()
    cabin_encoder = LabelEncoder()

    df['Sex'] = sex_encoder.fit_transform(df['Sex'])  # Convert 'Sex' to numerical
    df['Name'] = name_encoder.fit_transform(df['Name'])  # Convert 'Name' to numerical
    df['Ticket'] = ticket_encoder.fit_transform(df['Ticket'])  # Convert 'Ticket' to numerical
    df['Cabin'] = cabin_encoder.fit_transform(df['Cabin'])  # Convert 'Cabin' to numerical

    df['Embarked'].fillna('S', inplace=True)  # Fill missing 'Embarked' with 'S' for Southampton
    #df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Convert 'Embarked' to numerical
    df = pd.get_dummies(df, columns=['Embarked'])
    # Convert True/False values to 0/1
    df['Embarked_C'] = df['Embarked_C'].astype(int)
    df['Embarked_Q'] = df['Embarked_Q'].astype(int)
    df['Embarked_S'] = df['Embarked_S'].astype(int)

    # Handling missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Replace missing 'Age' with median value

    # Drop unnecessary columns
    df.drop(['PassengerId'], axis=1, inplace=True)
    df.drop(['Name'], axis=1, inplace=True)
    df.drop(['Ticket'], axis=1, inplace=True)
    df.drop(['Embarked_C'], axis=1, inplace=True)
    df.drop(['Pclass'], axis=1, inplace=True)

    # Visualize the data
    # visualize_data(df)

    # Split the data into features and target variable
    X = df
    y = df['Survived']
    X.drop('Survived', axis=1, inplace=True)  # Drop target variable from features

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1011)

    # Create and train the model
    model = LogisticRegression(max_iter=100)
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Recall: {recall:.2f}")

    # Visualize the predictions
    # visualize_prediction(y_pred)

    # Display theta values
    # weight = model.coef_
    # print("Weight values (coefficients):")
    # print(weight)

    # bias = model.intercept_
    # print("Bias value (intercept):")
    # print(bias)

    # Forwad testing
    test_data = {
        'Sex': [0, 1, 0],
        'Age': [25, 30, 22],
        'SibSp': [0, 1, 0],
        'Parch': [0, 0, 0],
        'Fare': [50.0, 70.0, 30.0],
        'Cabin': [1, 2, 3],
        'Embarked_Q': [0, 1, 0],
        'Embarked_S': [1, 0, 1]
    }
    test_df = pd.DataFrame(test_data)

    # Make predictions on synthetic test data
    # synthetic_predictions = model.predict(test_df)

    # Decode predictions
    # decoded_predictions = ['Survived' if pred == 1 else 'Not Survived' for pred in synthetic_predictions]
    # print("Decoded Predictions:", decoded_predictions)

if __name__ == "__main__":
    main()