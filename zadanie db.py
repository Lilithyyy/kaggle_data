# -*- coding: utf-8 -*-                                   # zabezpečí podporu slovenských znakov v komentároch a výstupoch
"""
Kompletná analýza datasetu db1.csv pomocou pandas, numpy a sklearn.
- načítanie a preskúmanie dát
- čistenie a príprava
- výpočty a vizualizácie
- trénovanie modelov a predikcia
"""

# ===== 1. IMPORT KNÍŽNIC =====
import os                                                # práca so súbormi a priečinkami
import sys                                               # umožní ukončiť program ak niečo chýba
import numpy as np                                       # výpočty s poľami a číslami
import pandas as pd                                      # práca s tabuľkovými dátami (DataFrame)
import matplotlib.pyplot as plt                          # kreslenie grafov

# knižnice pre strojové učenie zo sklearn
from sklearn.model_selection import train_test_split, cross_val_score   # rozdelenie dát a krížová validácia
from sklearn.preprocessing import StandardScaler, LabelEncoder          # štandardizácia čísel a kódovanie textu
from sklearn.pipeline import Pipeline                                   # pipeline = viac krokov spracovania naraz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # metriky výkonu modelu
from sklearn.linear_model import LogisticRegression                     # lineárny klasifikátor
from sklearn.ensemble import RandomForestClassifier                     # stromový model
from sklearn.svm import SVC                                             # SVM klasifikátor

# ===== 2. KONFIGURÁCIA =====
DATA_PATH = "db1.csv"                                    # názov vstupného CSV súboru
OUTPUT_DIR = "outputs"                                   # priečinok pre uloženie výstupov (grafy, CSV)
TARGET_COL = None                                        # cieľový stĺpec (label), None = skript ho skúsí nájsť automaticky
TARGET_CANDIDATES = ["target", "label", "class", "species", "y"]  # názvy, ktoré skript hľadá ako cieľové
RANDOM_STATE = 42                                        # seed = zaručí, že výsledky budú stále rovnaké

os.makedirs(OUTPUT_DIR, exist_ok=True)                  # ak priečinok 'outputs' neexistuje, vytvor ho

# ===== 3. NAČÍTANIE DÁT =====
try:
    df = pd.read_csv(DATA_PATH)                         # načítanie CSV do DataFrame
except FileNotFoundError:                               # ak súbor neexistuje
    sys.exit(f"Nenájdený súbor: {DATA_PATH}.")          # ukončí program s chybou

print("✅ Dataset načítaný:", DATA_PATH)                # info že súbor bol načítaný
print("\n🔎 Ukážka dát:")
print(df.head())                                        # zobrazí prvých 5 riadkov

print("\nℹ️ Informácie o dátach:")
print(df.info())                                        # info o počte riadkov, stĺpcov, typoch dát

print("\n📊 Základné štatistiky:")
print(df.describe(include='all'))                       # základné štatistiky číselných aj textových stĺpcov

# ===== 4. ČISTENIE DÁT =====
rename_map_candidates = {                               # ak dataset používa iné názvy, tu ich môžeme premenovať
    'sepal.length': 'sepal_length',
    'sepal.width': 'sepal_width',
    'petal.length': 'petal_length',
    'petal.width': 'petal_width',
    'variety': 'species',
    'class': 'species'
}
df = df.rename(columns=rename_map_candidates)           # premenovanie stĺpcov podľa mapy vyššie

before = len(df)                                        # uložíme si počet riadkov pred odstránením duplicit
df = df.drop_duplicates()                               # odstránime duplicitné riadky
after = len(df)                                         # počet riadkov po odstránení
print(f"\n🧹 Odstránených duplicitných riadkov: {before - after}")

print("\n📉 Počet chýbajúcich hodnôt:")
print(df.isnull().sum())                                # vypíšeme počet chýbajúcich hodnôt v každom stĺpci

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  # nájdeme číselné stĺpce
for col in numeric_cols:                                # pre každý číselný stĺpec
    if df[col].isnull().any():                          # ak obsahuje chýbajúce hodnoty
        df[col] = df[col].fillna(df[col].median())      # doplníme ich mediánom (robustné riešenie)

# ===== 5. ŠTATISTIKY + NUMPY TRANSFORMÁCIE =====
if len(numeric_cols) > 0:
    print("\n📈 Štatistiky len pre číselné stĺpce:")
    print(df[numeric_cols].describe())                  # základné štatistiky len pre čísla

# funkcia na min-max normalizáciu (0 až 1)
def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)                                 # pretypujeme na float
    mn, mx = np.nanmin(x), np.nanmax(x)                 # nájdeme min a max
    return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)  # ak max = min, vrátime nuly (inak výpočet)

for col in numeric_cols:                                # pre každý číselný stĺpec
    df[f"{col}_norm"] = minmax_normalize(df[col].to_numpy())  # pridáme nový *_norm stĺpec s normalizovanými dátami

ref_col = numeric_cols[0] if len(numeric_cols) else None  # prvý číselný stĺpec zoberieme ako referenčný
if ref_col:
    arr = df[ref_col].to_numpy(dtype=float)            # prevedieme ho na numpy pole
    pct_change = np.empty_like(arr)                    # vytvoríme prázdne pole rovnakého tvaru
    pct_change[0] = np.nan                             # prvý riadok nemá predchádzajúci → NaN
    denominator = np.where(arr[:-1] == 0, np.nan, arr[:-1])  # ak je predchádzajúca hodnota 0 → NaN (delenie nulou)
    pct_change[1:] = (arr[1:] - arr[:-1]) / denominator * 100 # výpočet percentuálnej zmeny
    df[f"{ref_col}_pct_change"] = pct_change           # uložíme do nového stĺpca

# ===== 6. FILTROVANIE, GROUPBY, GRAFY =====
if ref_col:
    mean_val = df[ref_col].mean()                      # priemerná hodnota referenčného stĺpca
    filtered = df[df[ref_col] > mean_val]              # filtrovanie: hodnoty väčšie ako priemer
    print(f"\n📌 Prvých 10 riadkov kde {ref_col} > priemer:")
    print(filtered.head(10))                           # ukážeme prvých 10

if 'species' in df.columns and len(numeric_cols) > 0:
    grouped = df.groupby('species')[numeric_cols].mean(numeric_only=True)  # priemery podľa skupiny 'species'
    print("\n📊 Priemery podľa 'species':")
    print(grouped)

# Histogram = rozdelenie hodnôt
plt.figure()
if ref_col:
    df[ref_col].hist()
    plt.title(f"Histogram - {ref_col}")
    plt.xlabel(ref_col)
    plt.ylabel("Frekvencia")
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{ref_col}.png"))  # uložíme graf
    plt.close()

# Boxplot = rozptyl dát
plt.figure()
if len(numeric_cols) >= 2:
    df.boxplot(column=list(numeric_cols[:2]))
    plt.title("Boxplot prvých dvoch číselných stĺpcov")
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot.png"))
    plt.close()

# Scatter = vzťah dvoch číselných premenných
plt.figure()
if len(numeric_cols) >= 2:
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.title(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter.png"))
    plt.close()

# ===== 7. KLASIFIKÁCIA =====
# funkcia, ktorá skúsi automaticky nájsť cieľový stĺpec
def autodetect_target(dataframe, candidates):
    for c in candidates:
        if c in dataframe.columns:
            return c
    last = dataframe.columns[-1]                       # posledný stĺpec ako záložný plán
    if (dataframe[last].dtype == 'object') or (dataframe[last].nunique() <= 10):  # textový alebo málo unikátov
        return last
    return None

if TARGET_COL is None:
    TARGET_COL = autodetect_target(df, TARGET_CANDIDATES)  # pokus o nájdenie cieľa

if TARGET_COL is None or TARGET_COL not in df.columns:
    print("\n⚠️ Žiadny cieľový stĺpec – klasifikácia sa preskočí.")
    do_ml = False
else:
    do_ml = True

if do_ml:
    print(f"\n🎯 Cieľový stĺpec: '{TARGET_COL}'")

    df = df.dropna(subset=[TARGET_COL])                # odstránime riadky bez cieľa

    y_raw = df[TARGET_COL]                             # cieľový stĺpec
    if y_raw.dtype == 'object':                        # ak je textový
        le = LabelEncoder()                            # prevedieme na čísla
        y = le.fit_transform(y_raw)
        classes_ = list(le.classes_)                   # názvy tried
    else:
        y = y_raw.to_numpy()
        classes_ = sorted(list(pd.Series(y).unique()))

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  # nájdeme číselné vstupy
    feature_cols = [c for c in numeric_cols if c != TARGET_COL]           # X = všetko okrem cieľa
    if len(feature_cols) == 0:
        print("⚠️ Žiadne číselné vstupy – koniec ML.")
        do_ml = False

if do_ml:
    X = df[feature_cols].to_numpy(dtype=float)         # vstupné premenné ako numpy pole

    # rozdelenie na tréningovú (75 %) a testovaciu (25 %) množinu
    stratify_opt = y if pd.Series(y).nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify_opt
    )

    # definícia 3 modelov
    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),             # štandardizácia dát
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    }

    results = []                                      # uložíme výsledky modelov
    trained = {}                                      # uložíme natrénované modely

    print("\n🧪 Tréning a hodnotenie modelov:")
    for name, pipe in models.items():                 # iterácia cez všetky modely
        pipe.fit(X_train, y_train)                    # tréning modelu
        trained[name] = pipe

        y_pred = pipe.predict(X_test)                 # predikcia na testovacej množine
        acc = accuracy_score(y_test, y_pred)          # výpočet presnosti
        results.append((name, acc))                   # uloženie výsledku

        print(f"\n— {name} —")
        print("Presnosť:", round(acc, 4))
        print("Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0, target_names=[str(c) for c in classes_]))

        try:
            cv_scores = cross_val_score(pipe, X, y, cv=5)  # 5-násobná krížová validácia
            print("CV skóre:", round(cv_scores.mean(), 4), "±", round(cv_scores.std(), 4))
        except Exception as e:
            print("CV nespustené:", e)

        cm = confusion_matrix(y_test, y_pred)         # confusion matrix = matica chýb
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion matrix - {name}")
        plt.xlabel("Predikované")
        plt.ylabel("Skutočné")
        plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name}.png"))
        plt.close()

    best_name, best_acc = sorted(results, key=lambda x: x[1], reverse=True)[0]  # nájdeme najlepší model
    best_model = trained[best_name]
    print(f"\n🏆 Najlepší model: {best_name} (accuracy={round(best_acc, 4)})")

    if best_name == "RandomForest":                   # ak vyhrá RandomForest, zobrazíme dôležitosť znakov
        rf = best_model.named_steps["clf"]
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None:
            fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            print("\n🔎 Dôležitosť príznakov:")
            print(fi)

    n_preview = min(5, len(X_test))                   # ukážeme 5 predikcií
    preview_X = X_test[:n_preview]
    preview_true = y_test[:n_preview]
    preview_pred = best_model.predict(preview_X)

    def inv_label(vals):                              # funkcia na spätné dekódovanie textových tried
        try:
            return le.inverse_transform(vals)
        except NameError:
            return vals

    preview_df = pd.DataFrame({
        "true": inv_label(preview_true),
        "pred": inv_label(preview_pred)
    })
    print("\n🔮 Ukážka predikcií:")
    print(preview_df)

# ===== 8. ULOŽENIE OČISTENÝCH DÁT =====
out_csv = os.path.join(OUTPUT_DIR, "db1_clean.csv")   # názov výstupného CSV
df.to_csv(out_csv, index=False)                       # uložíme DataFrame ako CSV
print(f"\n💾 Očistené dáta uložené do: {out_csv}")
print("\n✅ HOTOVO.")                                 # hotovo!
