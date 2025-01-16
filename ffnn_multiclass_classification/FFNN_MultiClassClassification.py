import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import copy

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
torch.set_printoptions(precision=5, sci_mode=False)

# We load the data from a csv file
path = 'kaggle_diagnosed_cbc_data_v4.csv'
dataset = pd.read_csv(path)

# ---------------------------
# ----- DATA STATISTICS -----
# ---------------------------

print("\n\n\nThe first 10 rows from dataset are: \n")
print(dataset.head(10))

print("\n\n\nThe Statistics of all columns (DFs and TFs) are: \n")
print(dataset.describe())

print("\n\n\nThe number of NaN's corresponding to each column is: \n")
print(dataset.isnull().sum())

print("\n\n\nThe Column's DataTypes are: \n")
print(dataset.dtypes)

print("\n\n\nThe Target Levels of the dataset are: \n")
print(dataset['Diagnosis'].unique())

print("\n\n\nThe number of Target Levels Instances are: \n")
print(dataset['Diagnosis'].value_counts())

# -------------------------------
# ----- DATA PRE-PROCESSING -----
# -------------------------------

# We split the Independent Features from the Dependent Features
X = dataset.iloc[:, 0: -1].values
y = dataset.iloc[:, -1].values

y = y.reshape(-1, 1)

# We convert the TFs to One-Hot-Encoding representation
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# We split tha Dataset into Training Set and Test Set, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Feature Scaling ----- Normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data into (PyTorch) Tensors
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

print("\n\n\nThe shapes are:")
print(f"X_train: {X_train.size()}")
print(f"X_Test: {X_test.size()}")
print(f"y_train: {y_train.size()}")
print(f"y_test has: {y_test.size()}")


# -----------------------------
# ----- THE ARCHITECTURE ------
# -----------------------------

class FFNN_MultiClassClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearLayer1 = nn.Linear(in_features=14, out_features=10)
        self.activation1 = nn.ReLU()

        self.linearLayer2 = nn.Linear(in_features=10, out_features=50)
        self.activation2 = nn.ReLU()

        self.linearLayer3 = nn.Linear(in_features=50, out_features=100)
        self.activation3 = nn.ReLU()

        self.linearLayer4 = nn.Linear(in_features=100, out_features=150)
        self.activation4 = nn.ReLU()

        self.outputLayer = nn.Linear(in_features=150, out_features=9)
        self.activationOutput = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linearLayer1(x)
        x = self.activation1(x)

        x = self.linearLayer2(x)
        x = self.activation2(x)

        x = self.linearLayer3(x)
        x = self.activation3(x)

        x = self.linearLayer4(x)
        x = self.activation4(x)

        x = self.outputLayer(x)
        x = self.activationOutput(x)

        return x


# ------------------------------
# ----- TRAINING PROCEDURE -----
# ------------------------------

# Set the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_accuracy(model, inputs, labels):
    # It computes the accuracy of the <model> on the dataset <inputs,labels>

    # Set the model/data to GPU is available
    model = model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)

    model.eval()  # Set the model to evaluation mode

    outputs = model(inputs)
    acc = (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).float().mean()
    return acc


def train(model, train_inputs, train_labels, test_inputs, test_labels, batch_size, num_epochs, criterion, optimizer):
    print("\n\n\n ... The training process ...\n")

    # We split the Train Set into Training and Validation Sets, respectively
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_inputs, train_labels, test_size=0.2, random_state=0)
    print(f"Training Set size = {(train_inputs.size())[0]} --- Validation Set size = {(val_inputs.size())[0]}\n")

    # Set the model/data to GPU if available
    model = model.to(device)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

    train_size = (train_inputs.size())[0]
    batches = torch.arange(0, train_size, batch_size)

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        model.train()

        total_loss = 0
        for start_batch in batches:
            # Take a batch
            X_batch = train_inputs[start_batch: start_batch + batch_size]
            y_batch = train_labels[start_batch: start_batch + batch_size]

            # Forward and Backward Passes
            optimizer.zero_grad()
            batch_outputs = model(X_batch)
            batch_loss = criterion(batch_outputs, y_batch)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss

        # Compute the model's accuracy on validation set and record the model's weights corresponding to
        # the best validation accuracy
        validation_accuracy = compute_accuracy(model, val_inputs, val_labels)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        epoch_time = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {epoch_time: .3f} ===> "
              f"Validation Accuracy = {validation_accuracy: .4f} ===> Best Accuracy = {best_accuracy} at "
              f"epoch {best_epoch}")

    # Set the model('s weights) with the best accuracy, based on the Validation Set
    model.load_state_dict(best_weights)

    print("\n\n\nJust checking ...")
    final_accuracy = compute_accuracy(model, test_inputs, test_labels)
    print(f"\nThe Test Accuracy of the Final Models is: {final_accuracy: .4f}")


# ---------------------
# -----  MAIN() -------
# ---------------------

ffnn = FFNN_MultiClassClassification()

batch_size = 10
number_of_epochs = 50
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.001)

train(model=ffnn, train_inputs=X_train, train_labels=y_train, test_inputs=X_test, test_labels=y_test,
      batch_size=batch_size, num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)

# --------------------------------
# ----- PREDICTION/INFERENCE -----
# --------------------------------

print("\n\n\n----- Now we do some Inference --- some Prediction -----")

ffnn.eval()

prediction_01 = ffnn(torch.FloatTensor(scaler.transform(
    [[10, 43.2, 50.1, 4.3, 5, 2.77, 7.3, 24.2, 87.7, 26.3, 30.1, 189, 12.5, 0.17]])))
print("\nThe probabilities are: " + str(prediction_01))
print("The answer is: " + str((torch.argmax(prediction_01)).item() + 1))
print("The true label is: <index class = 6>")

prediction_02 = ffnn(torch.FloatTensor(scaler.transform(
    [[7.2, 30.7, 60.7, 2.2, 4.4, 3.97, 9, 30.5, 77, 22.6, 29.5, 148, 14.3, 0.14]])))
print("\nThe probabilities are: " + str(prediction_02))
print("The answer is: " + str((torch.argmax(prediction_02)).item() + 1))
print("The true label is: <index class = 2>")

prediction_03 = ffnn(torch.FloatTensor(scaler.transform(
    [[5.2, 19.7, 72.4, 1, 3.8, 4.85, 13.2, 41, 84.7, 27.2, 32.1, 181, 10, 0.15]])))
print("\nThe probabilities are: " + str(prediction_03))
print("The answer is: " + str((torch.argmax(prediction_03)).item() + 1))
print("The true label is: <index class = 1>")
