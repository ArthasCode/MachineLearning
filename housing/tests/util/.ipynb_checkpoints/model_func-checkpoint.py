import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib
import os
# Aqui estamos separando oq será conjunto de treinamento e conjunto de testes,
# de maneira completamente aleátoria.

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Este código abaixo é uma separação dos dados para qual irá ser direcionado ao
# conjunto de testes e qual será de treinamento.
# Aqui estamos usando o crc32 para um sistema de hash baseado no id de cada
# linha da base de dados que sempre separe oq for 
# dados de treinamento e dados de teste, da mesma forma, todas as vezes.

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Uma função que utiliza que retorna um conjunto de treinamento
# e outro de teste a partir de uma separação estratificada no
# conjunto de dados housing, baseando-se da coluna "income_cat".

def stratified_shuffle_split(housing, n_splits=1, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return (strat_train_set, strat_test_set)

# Função que ensina um modelo e retorna o RMSE de suas previsões


def model_score_rmse(learning_model, X, y=None, path_model=None):
    if path_model and os.path.exists(path_model):
        learning_model = joblib.load(path_model)
    else:
        learning_model.fit(X, y)

    X_predictions = learning_model.predict(X)
    model_mse = mean_squared_error(y, X_predictions)
    model_rmse = np.sqrt(model_mse)
    return model_rmse


def model_cross_rmse(learning_model, X, y=None, path_cross=None):
    if path_cross and os.path.exists(path_cross):
        return joblib.load(path_cross)
    score = cross_val_score(learning_model, X, y, cv=10,
                            scoring="neg_mean_squared_error")

    return np.sqrt(-score)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())