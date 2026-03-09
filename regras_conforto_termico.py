import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1 - SISTEMA BASEADO EM REGRAS
# -------------------------------

def sistema_regras(temp, umid):

    if temp <= 5:
        return "Muito Frio"

    if temp >= 35:
        return "Muito Quente"

    if umid <= 30:
        return "Muito Seco"

    if umid >= 80:
        return "Muito Úmido"

    if 10 <= temp < 20 and 30 < umid < 80:
        return "Precisa de Sol para Conforto"

    if 20 <= temp <= 25 and 30 < umid < 80:
        return "Confortável"

    if 30 < temp < 35 and 30 < umid < 80:
        return "Precisa de Vento para Conforto"

    return "Confortável"


# -------------------------------
# 2 - DATASET DE TESTE
# -------------------------------

dados = pd.DataFrame({
    "temperatura": [3, 15, 23, 29, 38, 25],
    "umidade": [50, 50, 50, 50, 60, 85]
})

# Classificação usando regras
dados["classificacao_regras"] = dados.apply(
    lambda x: sistema_regras(x["temperatura"], x["umidade"]),
    axis=1
)

print("Resultado com Sistema de Regras:")
print(dados)


# -------------------------------
# 3 - MACHINE LEARNING
# -------------------------------

# Entradas (features)
X = dados[["temperatura", "umidade"]]

# Saída (target)
y = dados["classificacao_regras"]

# Criando modelo
modelo = DecisionTreeClassifier()

# Treinamento
modelo.fit(X, y)

# Previsão
dados["classificacao_ml"] = modelo.predict(X)

print("\nResultado com Machine Learning:")
print(dados)


# -------------------------------
# 4 - COMPARAÇÃO
# -------------------------------

acuracia = accuracy_score(
    dados["classificacao_regras"],
    dados["classificacao_ml"]
)

print("\nAcurácia do modelo de Machine Learning:", acuracia)
