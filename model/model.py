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

# Sorterer data etter pairing_quality
# Setter X lik læringsdata og y lik target data, filtrerer data så den bare bruker bra anbefalinger
data = data[data['pairing_quality'] >= 4].copy()

data = data.sort_values('pairing_quality', ascending=False).drop_duplicates(
    subset=['food_item', 'wine_type'],
    keep='first'
)

X = data[["food_item", "food_category"]]

categorical_cols = ["food_item", "food_category"]

#lager encoder og tar vekk kolonner
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)],
    remainder='drop'
)

#
y_category = data["wine_category"]
label_encoder_category = LabelEncoder()
y_category_encoded = label_encoder_category.fit_transform(y_category)

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_category_encoded, test_size=0.25, random_state=42, stratify=y_category_encoded
)

model_category = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_category = Pipeline(steps=[
    ('transformer', ct),
    ('model', model_category)
])
pipe_category.fit(X_train_cat, y_train_cat)
y_pred_cat = pipe_category.predict(X_test_cat)

print("="*50)
print("WINE CATEGORY MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cat, y_pred_cat, target_names=label_encoder_category.classes_))


y_type = data["wine_type"]
label_encoder_type = LabelEncoder()
y_type_encoded = label_encoder_type.fit_transform(y_type)

X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X, y_type_encoded, test_size=0.25, random_state=42, stratify=y_type_encoded
)

ct_type = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)],
    remainder='drop'
)

model_type = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_type = Pipeline(steps=[
    ('transformer', ct_type),
    ('model', model_type)
])
pipe_type.fit(X_train_type, y_train_type)
y_pred_type = pipe_type.predict(X_test_type)

print("\n" + "="*50)
print("WINE TYPE MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test_type, y_pred_type):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_type, y_pred_type, target_names=label_encoder_type.classes_))


food_items = sorted(data["food_item"].dropna().unique().tolist())


def recommend_wine(food_item):
    row = data.loc[data["food_item"].str.lower() == food_item.lower()]
    
    if row.empty:
        return f"Error: Food item '{food_item}' not found in the dataset."
    
    category = row.iloc[0]["food_category"]
    
    input_data = pd.DataFrame({
        'food_item': [food_item],
        'food_category': [category]
    })
    
    category_prediction = pipe_category.predict(input_data)
    wine_category_recommendation = label_encoder_category.inverse_transform(category_prediction)[0]
    
    category_probabilities = pipe_category.predict_proba(input_data)[0]
    category_scores = list(zip(label_encoder_category.classes_, category_probabilities))
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    type_prediction = pipe_type.predict(input_data)
    wine_type_recommendation = label_encoder_type.inverse_transform(type_prediction)[0]
    
    type_probabilities = pipe_type.predict_proba(input_data)[0]
    
    predicted_category = wine_category_recommendation
    category_wines = data[data['wine_category'] == predicted_category]['wine_type'].unique()
    
    all_wine_scores = list(zip(label_encoder_type.classes_, type_probabilities))
    filtered_wine_scores = [(wine, prob) for wine, prob in all_wine_scores if wine in category_wines]
    filtered_wine_scores.sort(key=lambda x: x[1], reverse=True)
    
    output_text = f"Wine Category: {wine_category_recommendation} ({category_probabilities.max() * 100:.1f}% confidence)\n\n"
    output_text += f"Recommended {wine_category_recommendation} Wines:\n"
    
    for i, (wine, prob) in enumerate(filtered_wine_scores[:5], 1):
        output_text += f"  {i}. {wine}: {prob * 100:.1f}%\n"
    
    print(f"\n{'=' * 60}")
    print(f"WINE RECOMMENDATION FOR: {food_item}")
    print(f"  Food Category: {category}")
    print(f"{'=' * 60}")
    print(f"\n Wine Category: {wine_category_recommendation} ({category_probabilities.max() * 100:.1f}% confidence)")
    print(f"\nTop Wine Category Predictions:")
    for i, (cat, prob) in enumerate(category_scores[:3], 1):
        print(f"  {i}. {cat}: {prob * 100:.1f}%")
    
    print(f"\n Recommended {wine_category_recommendation} Wines:")
    for i, (wine, prob) in enumerate(filtered_wine_scores[:5], 1):
        print(f"  {i}. {wine}: {prob * 100:.1f}%")
    
    return output_text


print("\n" + "="*50)
print("TESTING RECOMMENDATION SYSTEM")
print("="*50)

recommend_wine("oysters")
recommend_wine("Thai Curry")
