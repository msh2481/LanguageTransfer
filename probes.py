import pandas as pd
import torch as t
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT


@typed
def linear_regression_probe(
    embeddings: Float[TT, "n d"], features: Float[TT, "n"]
) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, features, test_size=0.2
    )
    model = Ridge().fit(X_train, y_train)
    y_pred = t.tensor(model.predict(X_test))
    return r2_score(y_test, y_pred)


@typed
def linear_classification_probe(
    embeddings: Float[TT, "n d"], features: Bool[TT, "n"]
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        features,
        test_size=0.2,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


@typed
def mixed_probe(
    embeddings: Float[TT, "n d"], features: pd.DataFrame
) -> dict[str, float]:
    result = {}
    for column in features.columns:
        dtype = str(features[column].dtype)
        y = t.tensor(features[column])
        if "float" in dtype:
            result[column] = linear_regression_probe(embeddings, y)
        else:
            result[column] = linear_classification_probe(embeddings, y)
    return result
