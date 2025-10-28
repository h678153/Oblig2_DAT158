import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


#Importerer dataset
data = pd.read_csv("../dataset_wine&food/wine_food_pairings.csv")


# Setter X lik læringsdata og y lik target data
X = data[["food_item", "food_category", "cuisine",]]
y = data["wine_type"]


#encoder target til numerisk verdi
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#splitter data inn i 25% testdata, 75% treningsdata
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)


#identifiserer de kategoriske kolonnene i X (treningsverdiene)
categorical_cols = X.columns


#lager encoder og tar vekk kolonner
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)],
    remainder='drop'
)


#lager RFC model og trener på data
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

pipe = Pipeline(steps=[
    ('transformer', ct),
    ('model', model)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


#printer treffsikkerhetscore

print(accuracy_score(y_test, y_pred))

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


food_items = sorted(data["food_item"].dropna().unique().tolist())
food_category = sorted(data["food_category"].dropna().unique().tolist())
cuisine = sorted(data["cuisine"].dropna().unique().tolist())

def recommend_wine(food_item, food_category, cuisine):
    input_data = pd.DataFrame({
        'food_item': [food_item],
        'food_category': [food_category],
        'cuisine': [cuisine]
    })
    input_encoded = ct.transform(input_data)

    prediction = model.predict(input_encoded)
    wine_recommendation = label_encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(input_encoded)[0]
    wine_scores = list(zip(label_encoder.classes_, probabilities))
    wine_scores.sort(key=lambda x: x[1], reverse=True)


    output_text = f"  Best match: {wine_recommendation} ({probabilities.max() * 100:.1f}% confidence)\n"
    output_text+=f"\nTop 5 Recommendations:\n"
    for i, (wine, prob) in enumerate(wine_scores[:5], 1):
        output_text+=f"  {i}. {wine}: {prob * 100:.1f}%\n"


    print(f"\n{'=' * 60}")
    print(f"WINE RECOMMENDATION FOR:")
    print(f"  Food: {food_item}")
    print(f"  Category: {food_category}")
    print(f"  Cuisine: {cuisine}")
    print(f"{'=' * 60}")
    print(f"\n Best Wine: {wine_recommendation} ({probabilities.max() * 100:.1f}% confidence)")
    print(f"\nTop 5 Recommendations:")
    for i, (wine, prob) in enumerate(wine_scores[:5], 1):
        print(f"  {i}. {wine}: {prob * 100:.1f}%")

    return output_text

print ("\n" + "="*50)
print("TESTING RECOMMENDATION SYSTEM")
print("="*50)

recommend_wine("grilled ribeye", "Red Meat", "Italian")
recommend_wine("oysters", "Seafood", "French")
recommend_wine("Thai Curry", "Spicy", "Thai")
