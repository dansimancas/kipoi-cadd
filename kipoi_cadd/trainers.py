from gin_train.utils import write_json
import os
import numpy as np
from numpy import inf
from tqdm import tqdm
from kipoi.data_utils import numpy_collate_concat
from kipoi.external.flatten_json import flatten
import gin
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class SklearnTrainer:
    """Simple Scikit Learn model trainer
    """

    def __init__(self, model, train_dataset, valid_dataset, output_dir, cometml_experiment=None):
        """
        Args:
          model: 
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self,
              coef_init=None,
              intercept_init=None,
              sample_weight=None,
              batch_size=64,
              shuffle=True,
              num_workers=10):
        """Train the model
        Args:
          batch_size:
          num_workers: how many workers to use in parallel
        """
        from sklearn.externals import joblib

        print("Started loading training dataset")
        
        X_train, y_train = self.train_dataset.load_all(batch_size=batch_size,
                                               num_workers=num_workers)
        """
        it = self.train_dataset.batch_train_iter(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        X_train, y_train = next(it)
        """

        print("Finished loading training dataset")
        # train the model
        if len(self.valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        self.model.fit(X_train,
                       y_train,
                       coef_init=coef_init,
                       intercept_init=intercept_init,
                       sample_weight=sample_weight)

        joblib.dump(self.model, self.ckp_file)

    #     def load_best(self):
    #         """Load the best model from the Checkpoint file
    #         """
    #         self.model = load_model(self.ckp_file)

    def evaluate(self, batch_size=256, shuffle=True, num_workers=8, save=True):
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
        """
        print("Started loading validation dataset")
        
        X_valid, y_valid = self.valid_dataset.load_all(batch_size=batch_size,
                                               num_workers=num_workers)
        """
        it = self.valid_dataset.batch_train_iter(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        X_valid, y_valid = next(it)
        """
        print("Finished loading validation dataset")
        metric_res = self.model.score(X_valid, y_valid)

        if save:
            write_json(metric_res, self.evaluation_path, indent=2)

        if self.cometml_experiment:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res), prefix="best/")

        return metric_res


@gin.configurable
class SklearnLogisticRegressionTrainer:
    """Simple Scikit Learn model trainer
    """

    def __init__(self, model, train_dataset, valid_dataset, output_dir, cometml_experiment=None):
        """
        Args:
          model: 
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self, sample_weight=None):
        """Train the model
        Args:
          batch_size:
          num_workers: how many workers to use in parallel
        """
        from sklearn.externals import joblib

        print("Started loading training dataset")
        
        X_train, y_train = self.train_dataset.load_all()

        print("Finished loading training dataset. Shape: ", X_train.shape, "True values:", sum(y_train)/len(y_train))
        # train the model
        if len(self.valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        self.model.fit(X_train,
                       y_train,
                       sample_weight=sample_weight)

        joblib.dump(self.model, self.ckp_file)

    #     def load_best(self):
    #         """Load the best model from the Checkpoint file
    #         """
    #         self.model = load_model(self.ckp_file)

    def evaluate(self, sample_weight=None, save=True):
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
        """
        print("Started loading validation dataset")
        
        X_valid, y_valid = self.valid_dataset.load_all()

        print("Finished loading validation dataset Shape: ", X_valid.shape, "True values:", sum(y_valid)/len(y_valid))
        X_valid[X_valid == inf] = 65500
        y_valid[y_valid == inf] = 65500
        
        print(X_valid.max(), max(y_valid))

        metric_res = self.model.score(X_valid, y_valid, sample_weight=sample_weight)
        print("metric_res", metric_res, np.amax(X_valid))

        if save:
            write_json(metric_res, self.evaluation_path, indent=2)

        if self.cometml_experiment:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res), prefix="best/")

        return metric_res
