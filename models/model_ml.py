"""
Currently NOT used, NOT tested.
"""

import pickle
import json
from copy import deepcopy
from pathlib import Path
from random import shuffle
from typing import Any, NoReturn, Dict, Optional, List, Union, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier
from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import ClassificationOutput  # noqa: F401
from torch_ecg.utils.utils_data import stratified_train_test_split

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from cfg import BaseCfg
from data_reader import CINC2022Reader
from utils.scoring_metrics import compute_challenge_metrics  # noqa: F401


__all__ = [
    "OutComeClassifier",
]


class OutComeClassifier(object):
    """ """

    __name__ = "OutComeClassifier"

    def __init__(
        self,
        config: Optional[CFG] = None,
        db_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters:
        -----------
        to write

        """
        self.config = deepcopy(config)

        self.db_dir = db_dir
        self.reader = None
        self.__df_features = None
        self._prepare_training_data()

        self.__cache = {}
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.best_clf, self.best_params, self.best_score = None, None, None
        self.test_score = None
        self._no = 1

    @property
    def y_col(self) -> str:
        return "Outcome"

    @property
    def feature_list(self) -> List[str]:
        return ["Age", "Sex", "Height", "Weight", "Pregnancy status"] + [
            f"Location-{loc}" for loc in ["PV", "AV", "MV", "TV", "Phc"]
        ]

    def _prepare_training_data(
        self, db_dir: Optional[Union[str, Path]] = None
    ) -> NoReturn:
        """ """
        if db_dir is not None:
            self.db_dir = db_dir
        if self.db_dir is None:
            return
        self.db_dir = Path(self.db_dir).resolve().absolute()
        self.reader = CINC2022Reader(self.db_dir)
        self.__df_features = self.reader._df_records.copy()
        raise NotImplementedError

    def get_model(self, model_name: str, params: Optional[dict] = None) -> object:
        """
        Returns a model instance.
        """
        return self.model_map[model_name](**(params or {}))

    def load_model(self, model_path: Union[str, Path]) -> object:
        """
        Loads a model from a file.
        """
        return pickle.loads(Path(model_path).read_bytes())

    def save_model(self, model: object, model_path: Union[str, Path]) -> NoReturn:
        """
        Saves a model to a file.
        """
        Path(model_path).write_bytes(pickle.dumps(model))

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]) -> "OutComeClassifier":
        """ """
        clf = cls()
        clf.best_clf = clf.load_model(path)
        return clf

    def search(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
    ) -> tuple:
        """ """

        cache_key = self._get_cache_key(model_name, cv)

        if cv is None:
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = self._perform_grid_search_no_cv(
                model_name,
                self.grid_search_config[model_name],
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
            )

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score
        else:
            (
                self.best_clf,
                self.best_params,
                self.best_score,
                self.test_score,
            ) = self._perform_grid_search_cv(
                model_name,
                self.grid_search_config[model_name],
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                cv,
            )

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
                test_score=self.test_score,
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score, self.test_score

    def _perform_grid_search_no_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[object, dict, float]:
        """
        Performs a grid search on the given model and parameters without cross validation.
        """
        best_score = 0
        best_clf = None
        best_params = None
        with tqdm(param_grid) as pbar:
            for params in pbar:
                try:
                    clf_gs = self.get_model(model_name, params)
                    clf_gs.fit(X_train, y_train)
                except Exception:
                    continue
        raise NotImplementedError
        #         y_prob = clf_gs.predict_proba(X_val)
        #         output = ClassificationOutput(
        #             classes=self.outcomes,
        #             prob=y_prob,
        #             pred=pred,
        #             bin_pred=bin_pred,
        #         )
        #         metric_score = compute_challenge_metrics(output, y_val)
        #         if metric_score > best_score:
        #             best_score = metric_score
        #             best_params = params
        #             best_clf = clf_gs
        # return best_clf, best_params, best_score

    def _perform_grid_search_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv: int = 5,
    ) -> Tuple[object, dict, float, float]:
        """
        Performs a grid search on the given model and parameters with cross validation.
        """
        gscv = GridSearchCV(
            estimator=self.get_model(model_name),
            param_grid=param_grid.param_grid,
            scoring="roc_auc",
            cv=cv,
            verbose=1,
        )
        gscv.fit(X_train, y_train)
        best_clf = gscv.best_estimator_
        best_params = gscv.best_params_
        best_score = gscv.best_score_
        # test_score = roc_auc_score(y_val, best_clf.predict_proba(X_val)[:, 1])
        raise NotImplementedError

        # return best_clf, best_params, best_score, test_score

    def get_cache(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> dict:
        """ """
        key = self._get_cache_key(model_name, cv, name)
        return self.__cache[key]

    def _get_cache_key(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """ """
        key = model_name
        if cv is not None:
            key += f"_{cv}"
        if name is None:
            name = f"ex{self._no}"
        key += f"_{name}"
        return key

    def list_cache(self) -> List[str]:
        return list(self.__cache)

    @property
    def df_features(self) -> pd.DataFrame:
        return self.__df_features

    @property
    def model_map(self) -> Dict[str, object]:
        """
        Returns a map of model name to model class.
        """
        return {
            "svm": SVC,
            "svc": SVC,
            "random_forest": RandomForestClassifier,
            "rf": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "gdbt": GradientBoostingClassifier,
            "gb": GradientBoostingClassifier,
            "bagging": BaggingClassifier,
            "xgboost": XGBClassifier,
            "xgb": XGBClassifier,
        }

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """ """
        if self.reader is None:
            print(
                "No training data available. Please call `_prepare_training_data` first."
            )
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        aux_train_file = (
            BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        )
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())

        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            if self.training:
                return json.loads(aux_train_file.read_text())
            else:
                return json.loads(aux_test_file.read_text())

        df_train, df_test = stratified_train_test_split(
            self.reader.df_stats,
            [
                "Murmur",
                "Age",
                "Sex",
                "Pregnancy status",
                "Outcome",
            ],
            test_ratio=1 - train_ratio,
        )

        train_set = df_train["Patient ID"].tolist()
        test_set = df_test["Patient ID"].tolist()

        train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        test_file.write_text(json.dumps(test_set, ensure_ascii=False))
        aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        shuffle(train_set)
        shuffle(test_set)

        return train_set, test_set
