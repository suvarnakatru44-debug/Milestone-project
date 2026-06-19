import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv("/content/student_attendance_dataset_1500.csv")

df.columns = df.columns.str.strip()

label_encoder = LabelEncoder()
df['Final_Label'] = label_encoder.fit_transform(df['Final_Label'])

X = df[['Attendance_Percentage', 'Assignment_Score', 'Internal_Marks']]
y = df['Final_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully! Please enter the details below:\n")

try:
    user_attendance = float(input("Enter Attendance Percentage (e.g., 77.39): "))
    user_assignment = float(input("Enter Assignment Score (e.g., 62): "))
    user_internal = float(input("Enter Internal Marks (e.g., 83): "))

    manual_data = pd.DataFrame(
        [[user_attendance, user_assignment, user_internal]],
        columns=X.columns
    )

    numeric_prediction = model.predict(manual_data)

    decoded_prediction = label_encoder.inverse_transform(numeric_prediction)

    print("\n--- Prediction Result ---")
    print(f"The student is predicted to be: {decoded_prediction[0]}")

except ValueError:
    print("Please enter valid numeric values only.")

