import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization

logging.basicConfig(level=logging.INFO)


def load_data(train_path: str, test_path: str, target_column: str) -> tuple:
    train_data = pd.read_csv(train_path)
    if target_column not in train_data.columns:
        raise ValueError(f"Column '{target_column}' is missing from the training data.")
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(train_data: pd.DataFrame) -> tuple:
    """Create a preprocessing pipeline for input data and process the training data."""
    label_encoder = LabelEncoder()
    train_labels = torch.tensor(label_encoder.fit_transform(train_data.pop('NObeyesdad')), dtype=torch.long)

    numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns

    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])

    train_features = torch.tensor(preprocessor.fit_transform(train_data), dtype=torch.float)
    return train_features, train_labels, preprocessor, label_encoder


def create_model(input_size, hidden_size, output_size):
    model = BayesianNN(input_size, hidden_size, output_size)
    return model


class BayesianNN(PyroModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](hidden_size, output_size)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([output_size, hidden_size]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([output_size]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def train_and_optimize(train_features, train_labels, pbounds, random_state):
    optimizer = BayesianOptimization(
        f=lambda hidden_size, lr: train(hidden_size, lr, train_features, train_labels),
        pbounds=pbounds,
        random_state=random_state)
    optimizer.maximize(init_points=2, n_iter=3)
    return optimizer.max['params']


def train(hidden_size, lr, train_features, train_labels):
    hidden_size = int(hidden_size)
    lr = float(lr)
    model = create_model(train_features.shape[1], hidden_size, len(torch.unique(train_labels)))
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    kf = KFold(n_splits=5)
    accuracies = []
    for train_index, val_index in kf.split(train_features):
        train_feat, val_feat = train_features[train_index], train_features[val_index]
        train_lab, val_lab = train_labels[train_index], train_labels[val_index]
        pyro.clear_param_store()
        for _ in range(1000):
            svi.step(train_feat, train_lab)
        predictive = Predictive(model, guide=guide, num_samples=1000)
        samples = predictive(val_feat)
        predictions = torch.mode(samples['obs'], 0).values
        accuracy = (predictions == val_lab).float().mean().item()
        accuracies.append(accuracy)
    return np.mean(accuracies)


def train_final_model(train_features: torch.Tensor, train_labels: torch.Tensor, input_size: int, hidden_size: int,
                      lr: float):
    """Train the final Bayesian Neural Network model using optimized hyperparameters."""
    final_model = BayesianNN(input_size, hidden_size, len(torch.unique(train_labels)))
    final_guide = AutoDiagonalNormal(final_model)
    final_optimizer = Adam({"lr": lr})
    final_svi = SVI(final_model, final_guide, final_optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    for j in range(5000):  # Full training cycle
        loss = final_svi.step(train_features, train_labels)
        if j % 100 == 0:
            logging.info(f"Iteration {j} : loss = {loss}")

    return final_model, final_guide


def predict(test_features, model, guide):
    predictive = Predictive(model, guide=guide, num_samples=1000)
    samples = predictive(test_features)
    predictions = torch.mode(samples['obs'], 0).values
    return predictions.numpy()


def save_predictions(test_data, predictions, label_encoder, filename):
    decoded_predictions = label_encoder.inverse_transform(predictions)
    prediction_df = pd.DataFrame({
        'id': test_data['id'],
        'NObeyesdad': decoded_predictions
    })
    prediction_df.to_csv(filename, index=False)
    logging.info(f"Predictions saved to {filename}")


def main():
    train_data, test_data = load_data('data/train.csv', 'data/test.csv', 'NObeyesdad')
    train_features, train_labels, preprocessor, label_encoder = preprocess_data(train_data)
    test_features = torch.tensor(preprocessor.transform(test_data), dtype=torch.float)
    best_params = train_and_optimize(train_features, train_labels, {'hidden_size': (50, 200), 'lr': (0.001, 0.1)},
                                     random_state=1)
    final_model, final_guide = train_final_model(train_features, train_labels, train_features.shape[1],
                                                 int(best_params['hidden_size']), best_params['lr'])
    predictions = predict(test_features, final_model, final_guide)
    save_predictions(test_data, predictions, label_encoder, 'predictions_bnn.csv')


if __name__ == "__main__":
    main()
