"""
Modelo: Satisfacción con la vida vs. PIB per cápita.

Uso:
    python modelo.py                     # usa ../examenidrovo1p.csv
    python modelo.py ruta/al/archivo.csv
    python modelo.py --predecir 30000    # predice la satisfacción para un PIB dado
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

AQUI = Path(__file__).resolve().parent
CSV_POR_DEFECTO = AQUI.parent / "examenidrovo1p.csv"
COL_PIB = "GDP per capita (USD)"
COL_SAT = "Life satisfaction"


def limpiar_numero(valor):
    """Convierte '19.705.602' o '19705.602' en 19705.602.

    Si hay más de un punto, los primeros son separadores de miles
    y solo el último es el decimal.
    """
    texto = str(valor).strip()
    if texto.count(".") > 1:
        entero, decimal = texto.rsplit(".", 1)
        texto = entero.replace(".", "") + "." + decimal
    return float(texto)


def cargar_datos(ruta):
    df = pd.read_csv(ruta, dtype=str)
    df[COL_PIB] = df[COL_PIB].map(limpiar_numero)
    df[COL_SAT] = df[COL_SAT].map(limpiar_numero)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=CSV_POR_DEFECTO)
    parser.add_argument("--predecir", type=float, help="PIB per cápita a predecir")
    args = parser.parse_args()

    df = cargar_datos(args.csv)
    X = df[[COL_PIB]].values
    y = df[COL_SAT].values

    print(f"Datos: {len(df)} países  ({args.csv})")
    print(df.describe().round(2), "\n")
    print(f"Correlación PIB vs. satisfacción: {df[COL_PIB].corr(df[COL_SAT]):.3f}\n")

    modelos = {
        "Regresión lineal": LinearRegression(),
        "KNN (k=3)": KNeighborsRegressor(n_neighbors=3),
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        pred = modelo.predict(X)
        print(f"{nombre}: R² = {r2_score(y, pred):.3f}   "
              f"RMSE = {np.sqrt(mean_squared_error(y, pred)):.2f}")

    lin = modelos["Regresión lineal"]
    print(f"\nEcuación: satisfacción = {lin.intercept_:.3f} + {lin.coef_[0]:.6f} × PIB")

    if args.predecir is not None:
        print(f"\nPredicción para PIB = {args.predecir:,.0f} USD:")
        for nombre, modelo in modelos.items():
            print(f"  {nombre}: {modelo.predict([[args.predecir]])[0]:.2f}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[COL_PIB], df[COL_SAT], color="#d97757")
    for _, fila in df.iterrows():
        ax.annotate(fila["Country"], (fila[COL_PIB], fila[COL_SAT]),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(df[COL_PIB].min(), df[COL_PIB].max(), 100).reshape(-1, 1)
    ax.plot(xs, lin.predict(xs), color="#333", label="Regresión lineal")
    ax.set_xlabel("PIB per cápita (USD)")
    ax.set_ylabel("Satisfacción con la vida")
    ax.set_title("Satisfacción con la vida vs. PIB per cápita")
    ax.legend()
    fig.tight_layout()
    salida = AQUI / "grafico.png"
    fig.savefig(salida, dpi=120)
    print(f"\nGráfico guardado en {salida.name}")


if __name__ == "__main__":
    main()
