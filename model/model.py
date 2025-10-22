import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn


data = pd.read_csv("../dataset_wine&food/wine_food_pairings.csv")

predictors =[ "wine_category","food_item", "food_category", "cuisine",
             "quality_label", "description"]


from sklearn.model_selection import train_test_split

X = data[["wine_type", "food_item", "food_category", "cuisine",
             "quality_label"]]
y = data["wine_category"]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#identifiserer de kategoriske kolonnene i X (treningsverdiene)
categorical_cols = X.columns

#lager encoder og tar vekk kolonner
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='drop'
)

X_encoded = ct.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

y_encoded = LabelEncoder().fit_transform(y)




X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


