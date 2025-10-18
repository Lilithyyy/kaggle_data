# -*- coding: utf-8 -*-  # podpora pre diakritiku a UTF-8 v súbore
"""
UNIVERSAL KAGGLE ANALYZER – kompletne okomentovaný skript pre univerzálnu analýzu CSV z Kaggle.
Tento súbor je rozdelený do 4 častí v chate. Tu je ČASŤ 1/4.
Po skopírovaní všetkých častí za seba vznikne jeden súvislý .py skript.

Čo tento skript robí:
- Načíta ľubovoľný CSV súbor (predvolene db1.csv alebo cez env CSV_PATH)
- Urobí EDA (head/info/describe), základné čistenie (duplicity, NaN)
- Automaticky rozpozná dátumové stĺpce a vytvorí year/month/day/hour
- Pokúsi sa autodetekovať cieľový stĺpec (target)
- Rozlíši typ úlohy (klasifikácia alebo regresia)
- Pripraví nástroje na spracovanie číselných a textových stĺpcov (imputer, škálovanie, OneHot)
- Natrénuje viaceré modely (zvolené v konfigurácii), vyberie najlepší a uloží ho
- Uloží výstupy: grafy, predictions_test.csv, summary.txt, cleaned_data.csv, .joblib model
"""

# ====== Importy knižníc ======
import os  # práca so súbormi, priečinkami a prostredím
import sys  # umožní korektne ukončiť program (sys.exit) pri chybách
import json  # formát JSON pre uloženie súhrnov a výsledkov
import joblib  # ukladanie a načítanie natrénovaných modelov (pickle-like)
import numpy as np  # rýchle numerické výpočty a polia
import pandas as pd  # práca s tabuľkovými dátami (DataFrame)
import matplotlib.pyplot as plt  # kreslenie grafov

from typing import Optional, List  # typové anotácie pre čitateľnosť

# Z balíka scikit-learn importujeme stavebnice na spracovanie dát a modely
from sklearn.compose import ColumnTransformer  # kombinuje spracovanie numerických/kategórií
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder  # kódovanie a škálovanie
from sklearn.pipeline import Pipeline  # pospája kroky (preprocessing + model) do jedného objektu
from sklearn.impute import SimpleImputer  # dopĺňanie chýbajúcich hodnôt
from sklearn.model_selection import train_test_split, cross_val_score  # rozdelenie dát a krížová validácia

# Klasifikačné modely
from sklearn.linear_model import LogisticRegression  # lineárny klasifikátor (baseline)
from sklearn.ensemble import RandomForestClassifier  # stromová metóda (často silná out-of-the-box)
from sklearn.svm import SVC  # Support Vector Classifier (nelineárny RBF)

# Regresné modely
from sklearn.linear_model import LinearRegression  # klasická lineárna regresia
from sklearn.ensemble import RandomForestRegressor  # stromová regresia
from sklearn.svm import SVR  # Support Vector Regressor (nelineárny RBF)

# Metriky pre klasifikáciu a regresiu
from sklearn.metrics import (
    accuracy_score,                # presnosť (klasifikácia)
    f1_score,                      # F1 skóre (vážené pri nerovnováhe)
    classification_report,         # precision/recall/F1 po triedach
    confusion_matrix,              # matica chýb
    r2_score,                      # R^2 (regresia)
    mean_absolute_error,           # MAE (regresia)
    mean_squared_error             # MSE / RMSE (regresia)
)

# ====== Konfigurácia skriptu ======
DATA_PATH = os.environ.get("CSV_PATH", "db1.csv")  # cesta k CSV (ak je nastavená env premenná CSV_PATH, použije ju)
OUTPUT_DIR = "outputs"  # priečinok pre uloženie grafov, modelov a CSV výstupov
RANDOM_STATE = 42  # seed náhodného generátora pre reprodukovateľnosť

TARGET_COL: Optional[str] = None  # cieľový stĺpec (ak None, skript sa ho pokúsi nájsť sám)
TARGET_CANDIDATES = [             # bežné názvy, pod ktorými sa na Kaggle vyskytuje label
    "target", "label", "class", "species", "y", "output", "category", "outcome", "type"
]
FORCE_TASK: Optional[str] = None  # môžeš vynútiť "classification" alebo "regression"; inak auto

# Modely, ktoré budeme skúšať pre každý typ úlohy (poradie = poradie tréningu)
SELECTED_MODELS_CLASS = ["LogReg", "RandomForest", "SVC"]   # pre klasifikáciu
SELECTED_MODELS_REGR  = ["LinReg", "RandomForestReg", "SVR"]  # pre regresiu

MAX_OHE_UNIQUES = 50  # ak má kategória viac unikátov, nebudeme ju OneHot-ovať (aby nevybuchla dimenzia)

# Uisti sa, že výstupný priečinok existuje
os.makedirs(OUTPUT_DIR, exist_ok=True)  # ak neexistuje, vytvor; ak existuje, nič sa nestane

# ====== Pomocné funkcie ======
def safe_print(*args, **kwargs):
    """
    Bezpečné printovanie – ak by terminál nevedel zobraziť niektoré znaky, použijeme byte výstup.
    Toto len znižuje riziko chyby pri diakritike na rôznych systémoch.
    """
    try:
        print(*args, **kwargs)  # pokus o štandardný print
    except Exception:
        msg = " ".join(str(a) for a in args)  # fallback: spoj argumenty do jedného reťazca
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="ignore"))  # zapíš bajty v UTF-8

def detect_datetime_and_expand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vyhľadá stĺpce s názvami obsahujúcimi 'date', 'time', 'timestamp', 'dt',
    skúsi ich konvertovať na datetime a vytvorí nové featury: year, month, day, hour.
    """
    for col in list(df.columns):  # iterujeme cez mená stĺpcov (list, aby sme počas úprav neiterovali nad živým view)
        low = str(col).lower()  # názov stĺpca v malých písmenách
        if any(k in low for k in ["date", "time", "timestamp", "dt"]):  # jednoduché pravidlo pre detekciu dátumu/času
            try:
                dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)  # konverzia na datetime
                if dt.notna().sum() > 0:  # ak sa aspoň niektoré hodnoty podarilo konvertovať
                    df[col] = dt  # prepíš pôvodný stĺpec datetime hodnotami (NaN ostanú NaT)
                    df[f"{col}_year"]  = dt.dt.year   # extrahuj rok
                    df[f"{col}_month"] = dt.dt.month  # extrahuj mesiac
                    df[f"{col}_day"]   = dt.dt.day    # extrahuj deň
                    try:
                        df[f"{col}_hour"] = dt.dt.hour  # ak je čas k dispozícii, extrahuj hodinu
                    except Exception:
                        pass  # ak hodina nie je k dispozícii (napr. len dátum), ignoruj
            except Exception:
                pass  # pri chybe konverzie jednoducho nič neurob
    return df  # vráť rozšírený DataFrame

def autodetect_target(df: pd.DataFrame) -> Optional[str]:
    """
    Pokúsi sa nájsť cieľový stĺpec:
    1) ak sa niektorý zo známych názvov TARGET_CANDIDATES nachádza v stĺpcoch, vráti ho
    2) ak existuje presne 1 kateg. stĺpec (object/category), vráti ho
    3) inak prechádza stĺpce odzadu a hľadá taký s <= 20 unikátmi a ktorý nie je float
    4) fallback: ak posledný stĺpec je kateg. alebo má málo unikátov, vráti posledný
    """
    for c in TARGET_CANDIDATES:  # najprv pokus podľa bežných názvov
        if c in df.columns:
            return c
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()  # kategórie
    if len(cat_cols) == 1:  # ak je len jedna kategória, pravdepodobne label
        return cat_cols[0]
    for col in df.columns[::-1]:  # skús od konca (v mnohých CSV býva label vpravo)
        nunq = df[col].nunique(dropna=True)  # počet unikátnych hodnôt
        if nunq <= 20 and str(df[col].dtype) != "float64":  # málo unikátov a nie je to float (často je to trieda)
            return col
    last = df.columns[-1]  # fallback: posledný stĺpec
    nunq = df[last].nunique(dropna=True)  # počet unikátov posledného stĺpca
    if (df[last].dtype in ["object", "category"]) or (nunq <= 20):  # ak málo unikátov alebo kategória
        return last
    return None  # nenašiel sa vhodný cieľ

def decide_task_type(series: pd.Series) -> str:
    """
    Rozhodnutie typu úlohy:
    - ak je FORCE_TASK nastavený, vráť ho
    - ak je stĺpec numerický a má veľa unikátov -> REGRESIA
      inak -> KLASIFIKÁCIA
    - ak nie je numerický -> KLASIFIKÁCIA
    """
    if FORCE_TASK in {"classification", "regression"}:  # manuálne prebitie
        return FORCE_TASK
    if pd.api.types.is_numeric_dtype(series):  # ak je číslo
        return "classification" if series.nunique(dropna=True) <= 20 else "regression"
    return "classification"  # textové/kat. ciele sú klasifikácia

def limit_categorical_uniques(df: pd.DataFrame, cats: List[str]) -> List[str]:
    """
    Z danej množiny kategórií vráť len tie, ktoré majú ≤ MAX_OHE_UNIQUES unikátov.
    Zabraňuje to vytvoreniu obrovského OneHot priestoru pri high-cardinality stĺpcoch.
    """
    return [c for c in cats if df[c].nunique(dropna=True) <= MAX_OHE_UNIQUES]

def summarize_df(df: pd.DataFrame) -> str:
    """
    Zhrnutie: počet riadkov/stĺpcov, typy stĺpcov a počty chýbajúcich hodnôt.
    Výsledok sa uloží do summary.txt pre prehľad.
    """
    lines = []
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")  # celkové rozmery
    lines.append("\nColumn dtypes:\n" + str(df.dtypes))  # dátové typy
    lines.append("\nMissing values per column:\n" + str(df.isnull().sum()))  # počty NaN
    return "\n".join(lines)  # spoj riadky do jedného textu

# ====== Načítanie CSV ======
try:
    df = pd.read_csv(DATA_PATH)  # pokus načítať CSV do DataFrame
except FileNotFoundError:
    sys.exit(f"❌ Súbor {DATA_PATH} neexistuje. Umiestni CSV k skriptu alebo nastav env CSV_PATH.")  # koniec s chybou

# Základné EDA výpisy do konzoly (bezpečné printy)
safe_print("✅ Dataset načítaný:", DATA_PATH)  # informácia, že CSV sa načítalo
safe_print("\n🔎 Ukážka dát:\n", df.head())  # prvých 5 riadkov pre vizuálnu kontrolu
safe_print("\nℹ️ info():")  # nadpis
safe_print(df.info())  # typy stĺpcov a počty neprázdnych hodnôt
safe_print("\n📊 describe():\n", df.describe(include='all', datetime_is_numeric=True))  # štatistiky (aj pre datetime)

# ====== Základné čistenie a rozšírenie o dátumové featury ======
before = len(df)  # počet riadkov pred odstraňovaním duplicit
df = df.drop_duplicates()  # odstráň duplicitné riadky
safe_print(f"\n🧹 Odstránených duplicitných riadkov: {before - len(df)}")  # vypíš, koľko ich bolo

df = detect_datetime_and_expand(df)  # pridaj *_year/_month/_day/_hour ak sú dátumy

# ====== Autodetekcia cieľa a typu úlohy ======
if TARGET_COL is None:              # ak cieľ nie je zadaný ručne
    TARGET_COL = autodetect_target(df)  # skús ho nájsť automaticky

if TARGET_COL is None or TARGET_COL not in df.columns:  # ak sa cieľ nenašiel
    task_type = None  # nevieme, čo trénovať, budeme robiť len EDA a uloženie
    safe_print("\n⚠️ Nebol nájdený cieľový stĺpec – skript vykoná len EDA.")
else:
    safe_print(f"\n🎯 Cieľový stĺpec: {TARGET_COL}")  # vypíš, ktorý stĺpec je cieľ
    task_type = decide_task_type(df[TARGET_COL])  # urči typ úlohy podľa cieľového stĺpca
    safe_print("🧩 Typ úlohy:", task_type)  # vypíš typ (classification/regression)

# ====== Jednoduché doplnenie NaN pre EDA a grafy ======
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()  # zoznam číselných stĺpcov
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()  # zoznam kategórií
if TARGET_COL in numeric_cols:  # ak sa cieľ ocitol medzi numerickými, pre EDA ho vynecháme
    numeric_cols.remove(TARGET_COL)
if TARGET_COL in categorical_cols:  # to isté pre kategórie
    categorical_cols.remove(TARGET_COL)

if numeric_cols:
    # vyplň číselné NaN mediánmi (jednoduché robustné riešenie pre EDA)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
if categorical_cols:
    # kategórie vyplň textom "Unknown" (aby grafy/uloženie nespadli)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# ====== Rýchle grafy pre EDA ======
if numeric_cols:  # ak máme aspoň jeden číselný stĺpec
    ref = numeric_cols[0]  # zober prvý číselný ako referenčný pre histogram
    plt.figure()  # nová figure
    df[ref].hist()  # histogram rozdelenia hodnôt
    plt.title(f"Histogram - {ref}")  # nadpis grafu
    plt.xlabel(ref); plt.ylabel("Frekvencia")  # osi
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{ref}.png"), bbox_inches="tight")  # ulož graf do outputs/
    plt.close()  # zavri figure, nech neprekročíme limity

    if len(numeric_cols) >= 2:  # ak sú aspoň 2 číselné
        plt.figure()
        df.boxplot(column=list(numeric_cols[:2]))  # boxplot pre prvé dva číselné stĺpce
        plt.title("Boxplot – prvé dva číselné stĺpce")
        plt.savefig(os.path.join(OUTPUT_DIR, "boxplot.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])  # scatter vzťahu prvých dvoch číselných stĺpcov
        plt.title(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
        plt.xlabel(numeric_cols[0]); plt.ylabel(numeric_cols[1])
        plt.savefig(os.path.join(OUTPUT_DIR, "scatter.png"), bbox_inches="tight")
        plt.close()

# ====== Uloženie súhrnu a očistených dát (EDA-only fallback) ======
summary_txt = os.path.join(OUTPUT_DIR, "summary.txt")  # cesta k súhrnu
with open(summary_txt, "w", encoding="utf-8") as f:  # otvor súbor na zápis (UTF-8)
    f.write("UNIVERSAL KAGGLE ANALYZER – SUMMARY\n" + "="*40 + "\n\n")  # hlavička
    f.write(f"Source CSV: {DATA_PATH}\n")  # zdroj
    f.write(f"Target: {TARGET_COL if TARGET_COL else 'N/A'}\n\n")  # cieľový stĺpec
    f.write("Data overview:\n")  # nadpis
    f.write(summarize_df(df))  # vlož prehľad o dátach (typy, missingy)

cleaned_csv = os.path.join(OUTPUT_DIR, "cleaned_data.csv")  # cesta pre očistené dáta
df.to_csv(cleaned_csv, index=False)  # ulož DataFrame do CSV bez indexu

if not task_type:  # ak sme nenašli cieľ a teda netrénujeme
    safe_print(f"\n💾 Očistené dáta uložené do: {cleaned_csv}\n📝 Súhrn uložený do: {summary_txt}")
    sys.exit(0)  # ukonči program – EDA je hotové, ale ML časť sa nespustí
# ====== Príprava dát pre model (ak máme TARGET a typ úlohy) ======
y = df[TARGET_COL]                 # y = cieľová premenná (label / target)
X = df.drop(columns=[TARGET_COL])  # X = všetky ostatné stĺpce (features)

# Identifikuj numerické a kategórie po odobratí targetu
num_cols_all = X.select_dtypes(include=["number"]).columns.tolist()            # všetky číselné vstupy
cat_cols_all = X.select_dtypes(include=["object", "category"]).columns.tolist()  # všetky kategórie vstupy

# Obmedz kategórie s príliš veľa unikátmi, aby OneHot neexplodoval
cat_cols_ohe = [c for c in cat_cols_all if X[c].nunique(dropna=True) <= MAX_OHE_UNIQUES]  # ponechaj len rozumné

# Vytvor pre-processing pipelines pre čísla a kategórie
numeric_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # doplň NaN mediánmi
    ("scaler", StandardScaler())                     # škáluj na mean=0, std=1 (vhodné pre LR/SVC/SVR)
])

categorical_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # doplň NaN najčastejšou hodnotou
    ("onehot", OneHotEncoder(handle_unknown="ignore"))     # zakóduj kategórie; ignoruj neznáme pri teste
])

# Poskladaj ColumnTransformer: čísla idú cez numeric_transform, kategórie cez categorical_transform
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, num_cols_all),   # transformuj numerické stĺpce
        ("cat", categorical_transform, cat_cols_ohe)  # transformuj kategórie s rozumnou kardinalitou
    ],
    remainder="drop"  # ostatné stĺpce (napr. veľmi high-cardinality) zahoď
)

# ====== Rozhodnutie vetvy: klasifikácia vs. regresia ======
if task_type == "classification":  # ak ide o klasifikáciu
    # Label-encode y, ak je textový alebo kategória (príp. má priveľa unikátov a nie je int)
    if y.dtype in ["object", "category"] or (y.nunique() > 20 and not pd.api.types.is_integer_dtype(y)):
        le = LabelEncoder()                     # vytvor encoder
        y_enc = le.fit_transform(y.astype(str)) # prevedie triedy na čísla
        class_names = list(le.classes_)         # ulož si mapovanie späť na textové mená
    else:
        le = None                               # netreba kódovať
        y_enc = y.to_numpy()                    # rovno použijeme numerický target
        class_names = sorted(list(pd.Series(y_enc).unique()))  # mená tried sú číselné

    # Stratifikovaný split (ak máme aspoň 2 triedy), inak bez stratifikácie
    strat = y_enc if pd.Series(y_enc).nunique() > 1 else None  # stratifikuj, ak to dáva zmysel
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=RANDOM_STATE, stratify=strat
    )

    # Definuj zoo klasifikačných modelov – každý v pipeline s preprocessorom
    model_zoo = {
        "LogReg": Pipeline([
            ("prep", preprocessor),                                    # preprocessing
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))  # klasifikátor
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
        ]),
        "SVC": Pipeline([
            ("prep", preprocessor),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    }

    # Vyber len tie, ktoré si si nadefinoval v SELECTED_MODELS_CLASS
    selected = [m for m in SELECTED_MODELS_CLASS if m in model_zoo]  # poradie zachované

    results = []             # sem budeme ukladať výsledky modelov (accuracy, F1, CV)
    best_name = None         # meno najlepšieho modelu
    best_model = None        # samotný natrénovaný objekt pipeline
    best_acc = -1.0          # najlepšia accuracy (začneme veľmi nízko)

    # Trénuj a vyhodnoť každý vybraný model
    for name in selected:
        pipe = model_zoo[name]                         # zober pipeline
        pipe.fit(X_train, y_train)                     # natrénuj ju na tréningových dátach
        y_pred = pipe.predict(X_test)                  # predikcie na testovacích dátach

        acc = accuracy_score(y_test, y_pred)          # presnosť
        f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # F1 vážené

        # Krížová validácia pre orientačný obraz o stabilite
        try:
            cv_scores = cross_val_score(pipe, X, y_enc, cv=5, scoring="accuracy")  # 5-fold CV
            cv_line = f"CV acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"     # správa
        except Exception as e:
            cv_line = f"CV error: {e}"  # ak niečo padne (napr. príliš málo vzoriek), zapíš chybu

        # Výstupy do konzoly
        safe_print(f"\n— {name} —\nAccuracy: {acc:.4f}\nF1-weighted: {f1w:.4f}\n{cv_line}\n")
        safe_print(
            "Classification report:\n",
            classification_report(
                y_test, y_pred, zero_division=0, target_names=[str(c) for c in class_names]
            )
        )

        # Ulož aj confusion matrix ako obrázok
        cm = confusion_matrix(y_test, y_pred)  # matica chýb
        plt.figure()
        plt.imshow(cm, interpolation='nearest')  # heatmap-like vizualizácia
        plt.title(f"Confusion matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name}.png"), bbox_inches="tight")
        plt.close()

        # Ak je aktuálny model lepší podľa accuracy, prepis najlepšieho
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipe

        # Ulož si výsledky do prehľadu (budú zapísané aj do summary.txt)
        results.append({
            "model": name,
            "accuracy": float(acc),
            "f1_weighted": float(f1w),
            "cv": cv_line
        })

    # ====== Uloženie artefaktov pre klasifikáciu ======
    # Funkcia na spätné dekódovanie číselných štítkov na textové (ak sme LabelEncoder použili)
    def inv_label(vals):
        return le.inverse_transform(vals) if le is not None else vals  # ak le neexistuje, vráť pôvodné čísla

    # Malý náhľad predikcií (prvých 5 z testu)
    preview_n = min(5, len(y_test))  # vezmi max 5 vzoriek, alebo menej ak test je menší
    preview = pd.DataFrame({
        "true": inv_label(y_test[:preview_n]),                  # skutočné triedy
        "pred": inv_label(best_model.predict(X_test[:preview_n]))  # predikované triedy
    })
    preview_path = os.path.join(OUTPUT_DIR, "preview_predictions.csv")  # kam uložíme náhľad
    preview.to_csv(preview_path, index=False)  # uloženie CSV s náhľadom

    # Ulož aj plné predikcie na celý test
    full_pred = pd.DataFrame({
        "y_true": inv_label(y_test),
        "y_pred": inv_label(best_model.predict(X_test))
    })
    full_pred_path = os.path.join(OUTPUT_DIR, "predictions_test.csv")  # cesta
    full_pred.to_csv(full_pred_path, index=False)  # ulož plné predikcie

    # Ulož najlepší model (vrátane preprocesingu v pipeline)
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{best_name}.joblib")  # názov súboru podľa modelu
    joblib.dump(best_model, model_path)  # dump modelu

    # Dopiš súhrn do summary.txt (append)
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("\n\n=== CLASSIFICATION RESULTS ===\n")  # nadpis
        f.write(json.dumps(results, indent=2, ensure_ascii=False))  # výsledky v JSON formáte
        f.write(f"\n\nBest model: {best_name}\nSaved model: {model_path}\n")  # info o najlepšom modeli
        f.write(f"Preview preds: {preview_path}\nFull preds: {full_pred_path}\n")  # kde nájsť predikcie

    # Výpis dôležitých ciest do konzoly
    safe_print(f"\n🏆 Best model: {best_name} (saved: {model_path})")
    safe_print(f"🔮 Preview predictions -> {preview_path}")
    safe_print(f"💾 Full predictions -> {full_pred_path}")

# ====== Vetva pre REGRESIU ======
else:
    # Ak je úloha regresia, split bez stratifikácie (y je spojitý)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Definuj modely pre regresiu
    model_zoo = {
        "LinReg": Pipeline([
            ("prep", preprocessor),
            ("reg", LinearRegression())
        ]),
        "RandomForestReg": Pipeline([
            ("prep", preprocessor),
            ("reg", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE))
        ]),
        "SVR": Pipeline([
            ("prep", preprocessor),
            ("reg", SVR(kernel="rbf"))
        ])
    }

    # Podľa konfigurácie vyber ktoré trénovať
    selected = [m for m in SELECTED_MODELS_REGR if m in model_zoo]

    results = []           # výsledky modelov (R2, MAE, RMSE, CV)
    best_name = None       # najlepší názov
    best_model = None      # najlepší pipeline objekt
    best_r2 = -1e9         # najlepší R^2 (začíname veľmi nízko)

    # Trénuj a vyhodnocuj každý model
    for name in selected:
        pipe = model_zoo[name]          # pipeline s preprocesingom
        pipe.fit(X_train, y_train)      # tréning na tréningovej množine
        y_pred = pipe.predict(X_test)   # predikcia na testovacej množine

        # Metriky regresie
        r2  = r2_score(y_test, y_pred)                          # R^2 (vyššie = lepšie)
        mae = mean_absolute_error(y_test, y_pred)               # MAE (nižšie = lepšie)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE (nižšie = lepšie)

        # Krížová validácia (R2)
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")  # 5-fold CV pre R2
            cv_line = f"CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        except Exception as e:
            cv_line = f"CV error: {e}"

        # Výpis do konzoly
        safe_print(f"\n— {name} —\nR2: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n{cv_line}\n")

        # Sleduj najlepší podľa R2
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = pipe

        # Ulož výsledky do zoznamu
        results.append({
            "model": name,
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "cv": cv_line
        })

    # Uloženie najlepšieho modelu
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{best_name}.joblib")
    joblib.dump(best_model, model_path)

    # Ulož predikcie na test
    full_pred = pd.DataFrame({"y_true": y_test, "y_pred": best_model.predict(X_test)})
    full_pred_path = os.path.join(OUTPUT_DIR, "predictions_test.csv")
    full_pred.to_csv(full_pred_path, index=False)

    # Dopiš do summary.txt (append)
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("\n\n=== REGRESSION RESULTS ===\n")
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
        f.write(f"\n\nBest model: {best_name}\nSaved model: {model_path}\n")
        f.write(f"Predictions: {full_pred_path}\n")

    # Výpis ciest do konzoly
    safe_print(f"\n🏆 Best model: {best_name} (saved: {model_path})")
    safe_print(f"💾 Predictions -> {full_pred_path}")

# ====== Uloženie očistených dát a summary (záverečný krok) ======
# Pozn.: cleaned_data.csv sme už uložili skôr (pre EDA-only vetvu aj pre ML vetvu),
# ale znovu uloženie je idempotentné – nevadí, ak sa prepíše novšou verziou.

df.to_csv(cleaned_csv, index=False)  # ulož DataFrame po základných EDA úpravách
safe_print(f"\n💾 Cleaned -> {cleaned_csv}")  # informácia o ceste k očisteným dátam
safe_print(f"📝 Summary -> {summary_txt}")    # informácia o ceste k súhrnu

# ====== FINÁLNA SPRÁVA ======
safe_print("\n✅ DONE.")  # jasné ukončenie skriptu pre používateľa

# ====== Poznámky k používaniu ======
# 1) CSV si daj k tomuto skriptu (alebo nastav env premennú CSV_PATH).
# 2) Inštalácia závislostí:
#      pip install pandas numpy matplotlib scikit-learn joblib
# 3) Spustenie:
#      python universal_kaggle_analysis_commented.py
# 4) Výstupy hľadaj v priečinku outputs/ :
#      - cleaned_data.csv        (očistené dáta)
#      - summary.txt             (prehľad a metriky)
#      - cm_*.png                (confusion matrices pri klasifikácii)
#      - hist_*.png, boxplot.png, scatter.png (základné grafy)
#      - predictions_test.csv    (predikcie na testovacej množine)
#      - best_model_*.joblib     (najlepší model s preprocesingom)
#
# 5) Úpravy:
#    - Ak chceš vynútiť typ úlohy, nastav FORCE_TASK = "classification" alebo "regression".
#    - Ak poznáš cieľový stĺpec, nastav TARGET_COL = "tvoj_label".
#    - Ak chceš pridať/odobrať modely, uprav SELECTED_MODELS_CLASS / SELECTED_MODELS_REGR.
#    - MAX_OHE_UNIQUES uprav podľa potreby, ak máš veľa kategórií (napr. mestá, názvy produktov).
#
# 6) Bezpečnostné poznámky:
#    - Tento skript je určený na rýchlu exploráciu; pre produkciu zvaž validáciu vstupov,
#      lepší tuning modelov (GridSearchCV/RandomizedSearchCV), pipeline persistenciu verzií, atď.
#
# 7) Rozšírenia (nápady):
#    - Pridať automatické ukladanie feature importances pri RandomForeste (klasifikácia aj regresia).
#    - Export confusion matrix aj ako textová tabuľka (nie len obrázok).
#    - Automatická detekcia extrémov/outlierov a report.
#    - Automatický profilovací report (pandas-profiling/ydata-profiling), ak je povolené.
#
# 8) Najčastejšie problémy:
#    - CSV sa nenačíta: skontroluj cestu (DATA_PATH alebo env CSV_PATH).
#    - Žiadny target: nastav TARGET_COL ručne alebo uprav autodetekciu.
#    - Priveľa kategórií: zníž MAX_OHE_UNIQUES alebo urob target encoding (pokročilé).
#    - Malý dataset: CV môže zlyhať; skript chybu zaloguje a pokračuje.
