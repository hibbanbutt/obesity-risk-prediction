import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from bayes_opt import BayesianOptimization

# Setting up logging
logging.basicConfig(level=logging.INFO)


def load_data(file_path: str, target_column: str = None):
    """Load data from a file and check for target column if specified."""
    data = pd.read_csv(file_path)
    if target_column and target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' is missing from the data.")
    return data


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Preprocess the data and separate features and target labels."""
    label_encoder = LabelEncoder()
    train_labels = torch.tensor(label_encoder.fit_transform(train_data.pop('NObeyesdad')), dtype=torch.long)

    categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]),
         numerical_columns),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)
    ])

    train_features = torch.tensor(preprocessor.fit_transform(train_data), dtype=torch.float)
    test_features = torch.tensor(preprocessor.transform(test_data), dtype=torch.float)

    return train_features, train_labels, test_features, label_encoder


class NeuralNetwork(nn.Module):
    """A standard Neural Network with Dropout."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def bayesian_optimization(train_features, train_labels):
    """Perform Bayesian Optimization to find the best model hyperparameters."""

    def train_model(hidden_size, lr):
        model = NeuralNetwork(train_features.shape[1], int(hidden_size), len(torch.unique(train_labels)))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        kf = KFold(n_splits=5)
        accuracies = []

        for train_index, val_index in kf.split(train_features):
            train_feat, val_feat = train_features[train_index], train_features[val_index]
            train_lab, val_lab = train_labels[train_index], train_labels[val_index]
            for _ in range(20):  # Training loop
                optimizer.zero_grad()
                outputs = model(train_feat)
                loss = criterion(outputs, train_lab)
                loss.backward()
                optimizer.step()
            outputs = model(val_feat)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == val_lab).float().mean().item()
            accuracies.append(accuracy)
        return np.mean(accuracies)

    optimizer = BayesianOptimization(f=train_model, pbounds={'hidden_size': (50, 200), 'lr': (0.001, 0.1)},
                                     random_state=1)
    optimizer.maximize(init_points=2, n_iter=3)
    return optimizer.max['params']


def train_final_model(train_features, train_labels, best_params):
    """Train the final model using the optimized parameters."""
    model = NeuralNetwork(train_features.shape[1], int(best_params['hidden_size']), len(torch.unique(train_labels)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    for epoch in range(50):  # Train for 50 epochs
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')
    return model


def predict_with_model(model, test_features, samples=100):
    """Generate predictions using the trained model with dropout during inference."""
    model.train()
    predictions = []
    for _ in range(samples):
        outputs = model(test_features)
        predicted = torch.max(outputs, 1)[1]
        predictions.append(predicted.numpy())
    return np.array(predictions).T


def save_predictions(predictions, test_data, label_encoder, filename):
    """Save predictions to a CSV file."""
    mode_predictions = [np.bincount(pred).argmax() for pred in predictions]
    decoded_predictions = label_encoder.inverse_transform(mode_predictions)
    prediction_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': decoded_predictions})
    prediction_df.to_csv(filename, index=False)
    logging.info(f"Predictions saved to {filename}")

def main():
    train_data = load_data('data/train.csv', 'NObeyesdad')  # Expect target column
    test_data = load_data('data/test.csv')  # Do not expect target column
    train_features, train_labels, test_features, label_encoder = preprocess_data(train_data, test_data)
    best_params = bayesian_optimization(train_features, train_labels)
    final_model = train_final_model(train_features, train_labels, best_params)
    predictions = predict_with_model(final_model, test_features)
    save_predictions(predictions, test_data, label_encoder, 'predictions_mcd.csv')


# Main execution block
if __name__ == "__main__":
    main()
