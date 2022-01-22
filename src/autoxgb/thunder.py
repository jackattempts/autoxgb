import copy
import os
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass

import joblib
import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

from .enums import ProblemType
from .logger import logger
from .metrics import Metrics
from .params import get_params
from .utils import dict_mean, save_test_predictions, save_valid_predictions
from .autoxgb import AutoXGB

optuna.logging.set_verbosity(optuna.logging.INFO)

@dataclass
class Thunder_ML(AutoXGB):
    classificaton_tasks = [
        ProblemType.binary_classification,
        ProblemType.multi_class_classification,
        ProblemType.multi_label_classification
    ]

    def load_fold_data(self, fold, aug=False):
        train_feather = pd.read_feather(os.path.join(self.model_config.output, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(self.model_config.output, f"valid_fold_{fold}.feather"))

        xtest = None
        if self.model_config.test_filename is not None:
            test_feather = pd.read_feather(os.path.join(self.model_config.output, f"test_fold_{fold}.feather"))
            
        if aug and self.model_config.data_aug_func is not None:
            train_feather, valid_feather, test_feather = self.model_config.data_aug_func(
                train_feather, valid_feather, test_feather, self.model_config, fold
            )

        xtrain = train_feather[self.model_config.features]
        xvalid = valid_feather[self.model_config.features]

        ytrain = train_feather[self.model_config.targets].values
        yvalid = valid_feather[self.model_config.targets].values
        valid_ids = valid_feather[self.model_config.idx].values

        test_ids = None
        if self.model_config.test_filename is not None:
            xtest = test_feather[self.model_config.features]
            test_ids = test_feather[self.model_config.idx].values

        return (xtrain, ytrain), (xvalid, yvalid, valid_ids), (xtest, test_ids)
    
    def train_step(self, model, xtrain, ytrain, xvalid, yvalid, fold_idx):
        self.fit(model, xtrain, ytrain, xvalid, yvalid, fold_idx)
        
    def test_step(self, model, xtest, fold_idx):
        return self.predict(model, xtest, fold_idx)

    def multi_step(self, model, xtrain, ytrain, xvalid, yvalid, xtest, fold_idx, save_model=True):
        ypred = []
        test_pred = []
        trained_models = []
        for idx in range(len(self.model_config.targets)):
            _m = copy.deepcopy(model)
            self.train_step(_m, xtrain, ytrain[:, idx], xvalid, yvalid[:, idx], fold_idx)
            trained_models.append(_m)
                
            if self.model_config.problem_type == ProblemType.multi_column_regression:
                ypred_temp = self.predict(_m, xvalid, fold_idx, is_proba=False)
                if xtest is not None and self.model_config.test_filename is not None:
                    test_pred_temp = self.predict(_m, xtest, fold_idx, is_proba=False)
            else:
                ypred_temp = self.predict(_m, xvalid, is_proba=True)[:, 1]
                if xtest is not None and self.model_config.test_filename is not None:
                    test_pred_temp = self.predict(_m, xtest, fold_idx, is_proba=True)[:, 1]

            ypred.append(ypred_temp)
            if xtest is not None and self.model_config.test_filename is not None:
                test_pred.append(test_pred_temp)

        ypred = np.column_stack(ypred)
        if xtest is not None and self.model_config.test_filename is not None:
            test_pred = np.column_stack(test_pred)
        if save_model:
            try:
                joblib.dump(trained_models, os.path.join(self.model_config.output, f"axgb_model.{fold_idx}"))
            except:
                pass
        return ypred, test_pred
        
    def single_step(self, model, xtrain, ytrain, xvalid, yvalid, xtest, fold_idx, save_model=True):
        self.train_step(model, xtrain, ytrain, xvalid, yvalid, fold_idx)
        ypred = self.test_step(model, xvalid, fold_idx)

        test_pred = None
        if xtest is not None and self.model_config.test_filename is not None:
            test_pred = self.test_step(model, xtest, fold_idx)
        if save_model:
            try:
                joblib.dump(model, os.path.join(self.model_config.output, f"axgb_model.{fold_idx}"))
            except:
                pass
        return ypred, test_pred

    def fetch_problem_params(self):
        if self.model_config.problem_type == ProblemType.binary_classification:
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
        elif self.model_config.problem_type == ProblemType.multi_class_classification:
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "mlogloss"
        elif self.model_config.problem_type == ProblemType.multi_label_classification:
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
        elif self.model_config.problem_type == ProblemType.single_column_regression:
            use_predict_proba = False
            direction = "minimize"
            eval_metric = "rmse"
        elif self.model_config.problem_type == ProblemType.multi_column_regression:
            use_predict_proba = False
            direction = "minimize"
            eval_metric = "rmse"
        else:
            raise NotImplementedError
        return use_predict_proba, eval_metric, direction

    def process(self, training_params=None, show_progress=False):
        self._process_data()
        if training_params is None:
            training_params = self.get_tuned_params()
        self.training_params = training_params

        metrics = Metrics(self.model_config.problem_type)
        scores, final_test_predictions, final_valid_predictions = [], [], {}
        target_encoder = joblib.load(f"{self.model_config.output}/axgb.target_encoder")

        dl = tqdm(range(self.model_config.num_folds)) if show_progress else range(self.model_config.num_folds)
        for fold_idx in dl:
            model = self.get_model(params=training_params)

            logger.info(f"Training and predicting for fold {fold_idx}")
            (xtrain, ytrain), (xvalid, yvalid, valid_ids), (xtest, test_ids) = self.load_fold_data(fold_idx, aug=True)

            if self.model_config.problem_type in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
                ypred, test_pred = self.multi_step(model, xtrain, ytrain, xvalid, yvalid, xtest, fold_idx)
            else:
                ypred, test_pred = self.single_step(model, xtrain, ytrain, xvalid, yvalid, xtest, fold_idx)
                
            final_valid_predictions.update(dict(zip(valid_ids, ypred)))
            if self.model_config.test_filename is not None:
                final_test_predictions.append(test_pred)

            # calculate metric
            metric_dict = metrics.calculate(yvalid, ypred)
            scores.append(metric_dict)
            logger.info(f"Fold {fold_idx} done!")

        mean_metrics = dict_mean(scores)
        logger.info(f"Metrics: {mean_metrics}")
        save_valid_predictions(final_valid_predictions, self.model_config, target_encoder, "oof_predictions.csv")

        if self.model_config.test_filename is not None:
            save_test_predictions(final_test_predictions, self.model_config, target_encoder, test_ids, "test_predictions.csv")
        else:
            logger.info("No test data supplied. Only OOF predictions were generated")

    def objective(self, trial):
        tuning_params = self.params_for_optuna(trial)
        metrics = Metrics(self.model_config.problem_type)
        scores = []
        _, eval_metric, _ = self.fetch_problem_params()
        try:
            for fold in range(self.model_config.num_folds):
                (xtrain, ytrain), (xvalid, yvalid, _), _ = self.load_fold_data(fold, aug=False)
                model = self.get_model(params=tuning_params)
                if self.model_config.problem_type in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
                    ypred, _ = self.multi_step(model, xtrain, ytrain, xvalid, yvalid, xtest=None, fold_idx=fold, save_model=False)
                else:
                    ypred, _ = self.single_step(model, xtrain, ytrain, xvalid, yvalid, xtest=None, fold_idx=fold, save_model=False)
                # calculate metric
                metric_dict = metrics.calculate(yvalid, ypred)
                scores.append(metric_dict)
                if self.model_config.fast is True:
                    break
            mean_metrics = dict_mean(scores)
            logger.info(f"Metrics: {mean_metrics}")
            return mean_metrics[eval_metric]            
        except Exception as e:
            print('Error:', e, tuning_params)
            return 1.
        
    def get_tuned_params(self, fold_idx=0, n_trials=200):
        if self.params_for_optuna is None:
            raise NotImplementedError
        
        print(f'Tuning {self.__class__.__name__} started')

        _, _, direction = self.fetch_problem_params()
        optimize_func = partial(
            self.objective,
        )
        db_path = os.path.join(self.model_config.output, "params.db")
        study = optuna.create_study(
            direction=direction,
            study_name="autoxgb",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )
        study.optimize(optimize_func, n_trials=self.model_config.num_trials, timeout=self.model_config.time_limit)
        return study.best_params
    
    def fit(self, model, xtrain, ytrain, xvalid, yvalid, fold_idx):
        model.fit(xtrain, ytrain)
    
    def predict(self, model, X, fold_idx, is_proba=None):
        use_predict_proba, _, _ = self.fetch_problem_params()
        if is_proba is None:
            is_proba = use_predict_proba
        if is_proba:
            return model.predict_proba(X)
        else:
            return model.predict(X)

    def get_best_params(self):
        params_path = os.path.join(self.model_config.output, "params.db")
        if not os.path.exists(params_path):
            print(f"params doesn't exist. Invalid path: {params_path}")
            return None 

        best_params = None
        try:
            study = optuna.load_study(study_name="autoxgb", storage=f"sqlite:///{params_path}")
        except Exception as exc:
            print("Error while loading optuna study from database", exc_info=exc)
        else:
            best_params = study.best_params
        finally:
            return best_params

    @abstractmethod
    def get_model(self, params=None):
        ...
        
    def before_train_fold_cb(self, fold_idx):
        ...
    
    def after_train_fold_cb(self, fold_idx):
        ...

