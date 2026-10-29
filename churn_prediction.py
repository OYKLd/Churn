"""
Prédiction du churn (désabonnement) de clients télécom
Dataset : IBM Telco Customer Churn (7 043 clients) - public, disponible sur Kaggle
Auteure : Lydie OUATTARA
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report, RocCurveDisplay

# 1. Chargement et nettoyage
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # 11 valeurs vides
df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"Clients : {len(df)} | Taux de churn : {df['Churn'].mean():.1%}")

# 2. Analyse exploratoire : churn par type de contrat
print(df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).round(3))

# 3. Préparation
X, y = df.drop(columns="Churn"), df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in X.columns if c not in num]
prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Modèles (classes déséquilibrées : ~27 % de churn)
models = {
    "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
results, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)]).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results.append({"Modèle": name, "ROC-AUC": roc_auc_score(y_test, proba),
                    "Rappel": recall_score(y_test, pred), "Précision": precision_score(y_test, pred),
                    "F1": f1_score(y_test, pred)})
    fitted[name] = pipe
res = pd.DataFrame(results).round(3).sort_values("ROC-AUC", ascending=False)
print(res.to_string(index=False))
res.to_csv("resultats_modeles.csv", index=False)

# 5. Facteurs de churn (régression logistique)
lr = fitted["Régression logistique"]
names = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
top = pd.concat([coefs.head(5), coefs.tail(5)])
plt.figure(figsize=(8, 5)); top.plot.barh(color=["#2a7" if v < 0 else "#c44" for v in top])
plt.title("Facteurs qui augmentent (rouge) ou réduisent (vert) le churn"); plt.tight_layout()
plt.savefig("facteurs_churn.png", dpi=120)

# 6. Courbes ROC
fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.set_title("Courbes ROC - prédiction du churn"); plt.tight_layout(); plt.savefig("courbes_roc.png", dpi=120)
print(classification_report(y_test, (lr.predict_proba(X_test)[:, 1] >= 0.5).astype(int)))
