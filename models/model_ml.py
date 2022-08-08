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
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier
from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import ClassificationOutput
from torch_ecg.components.loggers import LoggerManager
from torch_ecg.utils.utils_data import stratified_train_test_split
from torch_ecg.utils.utils_metrics import _cls_to_bin

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from cfg import BaseCfg
from data_reader import CINC2022Reader
from utils.scoring_metrics import compute_challenge_metrics
from models.outputs import CINC2022Outputs


__all__ = [
    "OutComeClassifier_CINC2022",
]


class OutComeClassifier_CINC2022(object):
    """ """

    __name__ = "OutComeClassifier_CINC2022"

    def __init__(
        self,
        config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters:
        -----------
        config: CFG, optional,

        """
        self.config = deepcopy(config)
        self.__imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.__scaler = StandardScaler()

        self.logger_manager = None
        self.reader = None
        self.__df_features = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self._prepare_training_data()

        self.__cache = {}
        self.__pipeline = None
        self.best_clf, self.best_params, self.best_score = None, None, None
        self._no = 1

    @property
    def y_col(self) -> str:
        return "Outcome"

    @property
    def feature_list(self) -> List[str]:
        return deepcopy(self.config.feature_list)

    def _prepare_training_data(
        self, db_dir: Optional[Union[str, Path]] = None
    ) -> NoReturn:
        """ """
        if db_dir is not None:
            self.config.db_dir = db_dir
        self.config.db_dir = self.config.get("db_dir", None)
        if self.config.db_dir is None:
            return

        if self.config.cont_scaler.lower() == "minmax":
            self.__scaler = MinMaxScaler()
        elif self.config.cont_scaler.lower() == "standard":
            self.__scaler = StandardScaler()
        else:
            raise ValueError(f"Scaler: {self.config.cont_scaler} not supported.")
        if self.logger_manager is None:
            logger_config = dict(
                log_dir=self.config.get("log_dir", None),
                log_suffix="OutcomeGridSearch",
                tensorboardx_logger=False,
            )
            self.logger_manager = LoggerManager.from_config(logger_config)

        self.config.db_dir = Path(self.config.db_dir).resolve().absolute()
        self.reader = CINC2022Reader(self.config.db_dir)

        self.__df_features = self.reader.df_stats[
            [self.config.split_col, self.config.y_col] + self.config.x_cols
        ]
        # to ordinal
        self.__df_features.loc[:, self.config.y_col] = self.__df_features[
            self.config.y_col
        ].map(self.config.class_map)
        for col in self.config.x_cols_cate:
            if self.__df_features[col].dtype == "bool":
                self.__df_features.loc[:, col] = self.__df_features[col].astype(int)
            elif col in self.config.ordinal_mappings:
                mapping = self.config.ordinal_mappings[col]
                self.__df_features.loc[:, col] = self.__df_features[col].map(mapping)
        for col in self.config.x_cols_cont:
            self.__df_features.loc[:, col] = self.__df_features.loc[:, col].apply(
                lambda x: np.nan if x == self.reader.stats_fillna_val else x
            )
        self.__df_features.loc[
            :, self.config.x_cols_cont
        ] = self.__imputer.fit_transform(self.__df_features[self.config.x_cols_cont])
        self.__df_features.loc[
            :, self.config.x_cols_cont
        ] = self.__scaler.fit_transform(self.__df_features[self.config.x_cols_cont])

        for loc in self.config.location_list:
            self.__df_features.loc[:, f"Location-{loc}"] = self.__df_features.apply(
                lambda row: -1
                if loc not in row["Locations"]
                else 0
                if loc not in row["Murmur locations"]
                else 1,
                axis=1,
            )

        self.__df_features.set_index(self.config.split_col, inplace=True)
        self.__df_features = self.__df_features[
            [self.config.y_col] + self.config.feature_list
        ]

        train_set, test_set = self._train_test_split()
        df_train = self.__df_features.loc[train_set]
        df_test = self.__df_features.loc[test_set]
        self.X_train = df_train[self.config.feature_list].values
        self.y_train = df_train[self.config.y_col].values
        self.X_test = df_test[self.config.feature_list].values
        self.y_test = df_test[self.config.y_col].values

    def get_model(
        self, model_name: str, params: Optional[dict] = None
    ) -> BaseEstimator:
        """
        Returns a model instance.
        """
        return self.model_map[model_name](**(params or {}))

    # def load_model(self, model_path: Union[str, Path]) -> dict:
    #     """
    #     Loads a model from a file.
    #     """
    #     return pickle.loads(Path(model_path).read_bytes())

    def save_model(
        self,
        model: BaseEstimator,
        imputer: SimpleImputer,
        scaler: BaseEstimator,
        config: CFG,
        model_path: Union[str, Path],
    ) -> NoReturn:
        """
        Saves a model to a file.
        """
        _config = deepcopy(config)
        _config.pop("db_dir", None)
        Path(model_path).write_bytes(
            pickle.dumps(
                {
                    "config": _config,
                    "imputer": imputer,
                    "scaler": scaler,
                    "outcome_classifier": model,
                }
            )
        )

    def format_pipeline(self) -> Pipeline:
        """ """
        assert all(
            [
                self.imputer is not None,
                self.scaler is not None,
                self.best_clf is not None,
            ]
        )
        self.__pipeline = Pipeline(
            [
                ("imputer", self.__imputer),
                ("scaler", self.__scaler),
                ("outcome_classifier", self.best_clf),
            ]
        )
        return self.__pipeline

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "OutComeClassifier":
        """ """
        loaded = pickle.loads(Path(path).read_bytes())
        config = loaded["config"]
        clf = cls(config)
        clf.__imputer = loaded["imputer"]
        clf.__scaler = loaded["scaler"]
        clf.best_clf = loaded["outcome_classifier"]
        clf.format_pipeline()
        return clf

    def search(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
    ) -> Tuple[BaseEstimator, dict, float]:
        """ """

        cache_key = self._get_cache_key(model_name, cv)

        if cv is None:
            msg = "Performing grid search with no cross validation."
            self.logger_manager.log_message(msg)
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = self._perform_grid_search_no_cv(
                model_name,
                ParameterGrid(self.config.grids[model_name]),
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
            msg = f"Performing grid search with {cv}-fold cross validation."
            self.logger_manager.log_message(msg)
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = self._perform_grid_search_cv(
                model_name,
                ParameterGrid(self.config.grids[model_name]),
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
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score

    def _perform_grid_search_no_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[BaseEstimator, dict, float]:
        """
        Performs a grid search on the given model and parameters without cross validation.
        """
        best_score = np.inf
        best_clf = None
        best_params = None
        with tqdm(enumerate(param_grid)) as pbar:
            for idx, params in pbar:
                try:
                    clf_gs = self.get_model(model_name, params)
                    clf_gs.fit(X_train, y_train)
                except Exception:
                    continue

                y_prob = clf_gs.predict_proba(X_val)
                y_pred = clf_gs.predict(X_val)
                bin_pred = _cls_to_bin(
                    y_pred, shape=(y_pred.shape[0], len(self.config.classes))
                )
                outputs = CINC2022Outputs(
                    outcome_output=ClassificationOutput(
                        classes=self.config.classes,
                        prob=y_prob,
                        pred=y_pred,
                        bin_pred=bin_pred,
                    ),
                    murmur_output=None,
                    segmentation_output=None,
                )

                val_metrics = compute_challenge_metrics(
                    labels=[{"outcome": y_val}],
                    outputs=[outputs],
                )

                msg = (
                    f"""Model - {self.model_map[model_name].__name__}\nParameters:\n"""
                )
                for k, v in params.items():
                    msg += f"""{k} = {v}\n"""
                self.logger_manager.log_message(msg)

                self.logger_manager.log_metrics(
                    metrics=val_metrics,
                    step=idx,
                    epoch=self._no,
                    part="val",
                )

                if val_metrics[self.config.monitor] < best_score:
                    best_score = val_metrics[self.config.monitor]
                    best_clf = clf_gs
                    best_params = params

        return best_clf, best_params, best_score

    def _perform_grid_search_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv: int = 5,
    ) -> Tuple[BaseEstimator, dict, float]:
        """
        Performs a grid search on the given model and parameters with cross validation.
        """
        gscv = GridSearchCV(
            estimator=self.get_model(model_name),
            param_grid=param_grid.param_grid,
            cv=cv,
            verbose=1,
        )
        gscv.fit(X_train, y_train)
        best_clf = gscv.best_estimator_
        best_params = gscv.best_params_
        # best_score = gscv.best_score_
        y_prob = best_clf.predict_proba(X_val)
        y_pred = best_clf.predict(X_val)
        bin_pred = _cls_to_bin(
            y_pred, shape=(y_pred.shape[0], len(self.config.classes))
        )
        outputs = CINC2022Outputs(
            outcome_output=ClassificationOutput(
                classes=self.config.classes,
                prob=y_prob,
                pred=y_pred,
                bin_pred=bin_pred,
            ),
            murmur_output=None,
            segmentation_output=None,
        )

        val_metrics = compute_challenge_metrics(
            labels={
                "outcome": y_val,
            },
            outputs=outputs,
        )
        best_score = val_metrics[self.config.monitor]

        msg = f"""Model - {self.model_map[model_name].__name__}\nParameters:\n"""
        for k, v in best_params.items():
            msg += f"""{k} = {v}\n"""
        self.logger_manager.log_message(msg)

        self.logger_manager.log_metrics(
            metrics=val_metrics,
            step=self._no,
            epoch=self._no,
            part="val",
        )

        return best_clf, best_params, best_score

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
    def imputer(self) -> SimpleImputer:
        return self.__imputer

    @property
    def scaler(self) -> BaseEstimator:
        return self.__scaler

    @property
    def pipeline(self) -> Pipeline:
        return self.__pipeline

    @property
    def model_map(self) -> Dict[str, BaseEstimator]:
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
            return json.loads(train_file.read_text()), json.loads(test_file.read_text())
        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            return json.loads(aux_train_file.read_text()), json.loads(
                aux_test_file.read_text()
            )

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
