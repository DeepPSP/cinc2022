"""
"""

import pickle  # noqa: F401
import time
from copy import deepcopy
from typing import Any, NoReturn

import torch
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight  # noqa: F401
from sklearn.model_selection import GridSearchCV  # noqa: F401

from sklearn.metrics import (  # noqa: F401
    make_scorer,
    accuracy_score,
    fbeta_score,
    jaccard_score,
    plot_confusion_matrix,
)
from xgboost import XGBClassifier
from torch_ecg.cfg import CFG


__all__ = [
    "OutComeXGB",
]


class OutComeXGB(object):
    """

    NOT finished yet!

    """

    __name__ = "OutComeXGB"

    def __init__(
        self,
        config: CFG,
        db_dir: str,
        feature_extractor: torch.nn.Module,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters:
        -----------
        to write

        """
        self.config = CFG(deepcopy(config))
        self.db_dir = db_dir

        self.model = XGBClassifier(**self.config.init_params)
        self.model_name = type(self.model).__name__

        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.fit_params = CFG()

    def train(self) -> NoReturn:
        """NOT finished"""
        # TODO: load feature using `self.feature_extractor`

        if not all(
            [
                len(self.x_train),
                len(self.y_train),
                len(self.x_test),
                len(self.y_test),
            ]
        ):
            raise ValueError("load data first!")

        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        params = {
            "tree_method": "gpu_hist",
        }
        params.update(self.config.train_params)

        evals_result = dict()

        start = time.time()
        booster = xgb.train(
            params,
            dtrain,
            evals=[(dtest, "Test")],
            evals_result=evals_result,
            **self.config.train_kw,
        )
        print(f"XGB training costs {(time.time()-start)/60:.2f} minutes")
        # print(f"evals_result = {dict_to_str(evals_result)}")

        save_path_params = "_".join(
            [
                str(k) + "-" + str(v)
                for k, v in params.items()
                if k
                not in [
                    "objective",
                    "num_class",
                    "verbosity",
                    "eval_metric",
                ]
            ]
        )
        # eval_name = f'Test_merror_{np.min(evals_result["Test"]["merror"]):.4f}'
        # save_path = cfg.model_path["ml"].format(
        #     model_name=self.model_name,
        #     params=save_path_params,
        #     eval=eval_name,
        #     ext="pkl",
        # )
        # booster.save_model(save_path)
        save_dict = {
            "model": booster,
        }

        # with open(save_path, "wb") as f:
        #     pickle.dump(save_dict, f)

    def _cv_xgb(self, params: dict):
        """not finished,

        Parameters:
        -----------
        to write

        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        cv_results = xgb.cv(
            params,
            dtrain,
            **self.config.cv_kw,
        )
        return cv_results
