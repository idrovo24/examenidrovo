# Carpeta `claude`

Modelo que predice la **satisfacción con la vida** a partir del **PIB per cápita** de 40 países.

## Archivos
- `modelo.py`: carga y limpia los datos, entrena dos modelos y genera el gráfico.
- `requirements.txt`: librerías necesarias.
- `grafico.png`: gráfico de dispersión con la recta de regresión.

## Cómo usarlo
```bash
pip install -r requirements.txt
python modelo.py                              # usa ../examenidrovo1p.csv
python modelo.py ../examenidrovomadero.csv    # también limpia los números con puntos de miles
python modelo.py --predecir 30000             # predicción para un PIB de 30 000 USD
```

## Resultados
| Modelo | R² | RMSE |
|---|---|---|
| Regresión lineal | 0.031 | 10.71 |
| KNN (k=3) | 0.410 | 8.36 |

Ecuación lineal: `satisfacción = 50.342 + 0.000091 × PIB`  
Correlación PIB vs. satisfacción: **0.175** (relación débil).

> Nota: algunos valores de PIB no son realistas (Bolivia, Burundi, Eswatini, El Salvador, India),
> lo que explica la baja correlación.
