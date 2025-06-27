import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load your landmark data
with open('asl_landmarks.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['data']
y = dataset['labels']

# Check if all data points have the same length
lengths = [len(x) for x in X]
if len(set(lengths)) > 1:
    print(f"Error: Data points have inconsistent lengths. Found lengths: {set(lengths)}")
    print("This is likely due to variations in the number of hands detected in the images.")
    print("Please revisit the data collection step (cell SYneibg6OZmK) to ensure consistent data representation.")
else:
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    with open('asl_knn_model.pickle', 'wb') as f:
        pickle.dump(model, f)

    print("\n💾 Model saved as 'asl_knn_model.pickle'")