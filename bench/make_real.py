"""Build benchmark datasets from public tables (the five Kaggle examples and others).

    python bench/make_real.py RAW_DIR OUT_DIR

RAW_DIR must contain the downloaded CSVs (see bench/README section of the main
README for sources); sklearn's bundled datasets need no download. Datasets
whose raw file is missing are skipped. Raw data is not committed to the repo.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_dataset  # noqa: E402


def _read(raw, rel, **kw):
    path = os.path.join(raw, rel)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kw)


def build(raw: str, out: str) -> list:
    made = []

    def add(name, df, **kw):
        if df is None:
            print(f"skip {name} (raw file missing)")
            return
        made.append(write_dataset(out, name, df, **kw))
        print(f"wrote {name}: {df.shape}")

    # --- the five examples from the brief -------------------------------------------
    telco = _read(raw, "telco/telco.csv")
    if telco is not None:
        ft = {c: "categorical" for c in telco.columns if c not in ("customerID", "Churn")}
        ft.update({"tenure": "numeric", "MonthlyCharges": "numeric", "TotalCharges": "numeric"})
        add("real_telco_churn", telco, task="classification", target="Churn", id_column="customerID",
            feature_types=ft, metric="f1_macro")

    ai4i = _read(raw, "ai4i/ai4i2020.csv")
    if ai4i is not None:
        ai4i = ai4i.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])  # failure-mode flags leak the target
        add("real_ai4i_maintenance", ai4i, task="classification", target="Machine failure", metric="f1_macro")

    house = _read(raw, "house/train.csv")
    if house is not None:
        ft = {c: ("numeric" if pd.api.types.is_numeric_dtype(house[c]) else "categorical")
              for c in house.columns if c not in ("Id", "SalePrice")}
        ft["MSSubClass"] = "categorical"
        add("real_house_prices", house, task="regression", target="SalePrice", id_column="Id",
            feature_types=ft, metric="rmse")

    bike = _read(raw, "bike/train.csv")
    if bike is not None:
        bike = bike.drop(columns=["casual", "registered"])  # they sum to the target
        ft = {"datetime": "datetime", "season": "categorical", "holiday": "categorical",
              "workingday": "categorical", "weather": "categorical", "temp": "numeric", "atemp": "numeric",
              "humidity": "numeric", "windspeed": "numeric"}
        add("real_bike_sharing", bike, task="regression", target="count", feature_types=ft, metric="rmsle")

    mall = _read(raw, "mall/mall.csv")
    if mall is not None:
        z = StandardScaler().fit_transform(mall[["Annual Income (k$)", "Spending Score (1-100)"]])
        ref = KMeans(5, n_init=50, random_state=0).fit_predict(z)
        add("real_mall_customers", mall, task="clustering", labels=ref, n_clusters=5, metric="ari",
            truth={"note": "reference labels = KMeans(5) on standardised income x spending score"})

    # --- other public tables ------------------------------------------------------
    peng = _read(raw, "penguins/penguins.csv")
    if peng is not None:
        lab = peng["species"].astype("category").cat.codes.to_numpy()
        add("real_penguins_clu", peng.drop(columns=["species"]), task="clustering", labels=lab, metric="ari")
        add("real_penguins_cls", peng, task="classification", target="species", metric="f1_macro")

    tit = _read(raw, "titanic/titanic.csv")
    if tit is not None:
        ft = {"Pclass": "categorical", "Name": "text", "Sex": "categorical", "Age": "numeric", "SibSp": "numeric",
              "Parch": "numeric", "Ticket": "categorical", "Fare": "numeric", "Cabin": "categorical",
              "Embarked": "categorical"}
        add("real_titanic", tit, task="classification", target="Survived", id_column="PassengerId",
            feature_types=ft, metric="accuracy")

    cal = _read(raw, "calhousing/housing.csv")
    if cal is not None:
        add("real_california_housing", cal, task="regression", target="median_house_value", metric="rmse")

    adult = _read(raw, "adult/adult-all.csv", header=None, skipinitialspace=True,
                  names=["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                         "hours_per_week", "native_country", "income"])
    if adult is not None:
        add("real_adult_income", adult, task="classification", target="income", metric="f1_macro")

    abal = _read(raw, "abalone/abalone.csv", header=None,
                 names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
                        "viscera_weight", "shell_weight", "rings"])
    if abal is not None:
        add("real_abalone", abal, task="regression", target="rings", metric="mae")

    glass = _read(raw, "glass/glass.csv", header=None,
                  names=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"])
    if glass is not None:
        add("real_glass", glass, task="classification", target="type", metric="f1_macro")

    wine_red = _read(raw, "winered/winequality-red.csv", header=None,
                     names=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                            "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates",
                            "alcohol", "quality"])
    if wine_red is not None:
        add("real_wine_quality", wine_red, task="regression", target="quality", metric="rmse")

    pima = _read(raw, "pima/pima.csv", header=None,
                 names=["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
                        "pedigree", "age", "outcome"])
    if pima is not None:
        add("real_pima_diabetes", pima, task="classification", target="outcome", metric="balanced_accuracy")

    iono = _read(raw, "ionosphere/ionosphere.csv", header=None)
    if iono is not None:
        iono.columns = [f"a{i:02d}" for i in range(iono.shape[1] - 1)] + ["radar"]
        add("real_ionosphere", iono, task="classification", target="radar", metric="accuracy")

    seeds = _read(raw, "seeds/seeds.csv", header=None,
                  names=["area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry",
                         "groove_length", "variety"])
    if seeds is not None:
        add("real_seeds_clu", seeds.drop(columns=["variety"]), task="clustering",
            labels=seeds["variety"].to_numpy(), metric="ari")

    ins = _read(raw, "autoins/auto-insurance.csv", header=None, names=["claims", "payment"])
    if ins is not None:
        add("real_auto_insurance_tiny", ins, task="regression", target="payment", metric="rmse")

    # --- sklearn bundled datasets (no download) -----------------------------------
    iris = skd.load_iris(as_frame=True)
    add("real_iris_clu", iris.data, task="clustering", labels=iris.target.to_numpy(), metric="ari")
    wine = skd.load_wine(as_frame=True)
    add("real_wine_clu", wine.data, task="clustering", labels=wine.target.to_numpy(), metric="ari")
    bc = skd.load_breast_cancer(as_frame=True)
    df = bc.data.copy()
    df["diagnosis"] = np.where(bc.target == 0, "malignant", "benign")
    add("real_breast_cancer", df, task="classification", target="diagnosis", metric="f1_macro")
    dig = skd.load_digits(as_frame=True)
    df = dig.data.copy()
    df["digit"] = dig.target
    add("real_digits_cls", df, task="classification", target="digit", metric="accuracy")
    add("real_digits_clu", dig.data, task="clustering", labels=dig.target.to_numpy(), metric="ari")
    # extra clustering tables with real labels (harder; used to validate the selection rule)
    add("realv_breast_cancer_clu", bc.data, task="clustering", labels=bc.target.to_numpy(), metric="ari")
    if glass is not None:
        add("realv_glass_clu", glass.drop(columns=["type"]), task="clustering", labels=glass["type"].to_numpy(),
            metric="ari")
    if abal is not None:
        add("realv_abalone_sex_clu", abal.drop(columns=["sex"]), task="clustering",
            labels=abal["sex"].astype("category").cat.codes.to_numpy(), metric="ari")
    dia = skd.load_diabetes(as_frame=True, scaled=False)
    df = dia.data.copy()
    df["progression"] = dia.target
    add("real_diabetes", df, task="regression", target="progression", metric="r2")
    return made


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])
