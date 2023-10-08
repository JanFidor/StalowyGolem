import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset
from sktime.datasets import (
    load_japanese_vowels,  # multivariate dataset with unequal length
)
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)
from sktime.forecasting.model_selection import SlidingWindowSplitter
from tqdm import tqdm

cv = SlidingWindowSplitter(window_length=15, fh=1)

minirocket_pipeline = make_pipeline(
    MiniRocket(),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)


X_train, y_train = load_arrow_head(split="train")

TimeSeriesSplit(n_splits=3)


minirocket_pipeline.fit(X_train, y_train)

X_test, y_test = load_arrow_head(split="test")

minirocket_pipeline.score(X_test, y_test)