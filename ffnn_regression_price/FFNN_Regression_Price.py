import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)
torch.set_printoptions(precision=5, sci_mode=False)

# We load the data from a csv file
path = 'insurance.csv'
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

num_zeros = (dataset[['age', 'bmi', 'children', 'charges']] == 0).sum()
print("\n\n\nThe number of zeros from each column: \n")
print(num_zeros)

# -------------------------------
# ----- DATA PRE-PROCESSING -----
# -------------------------------

# We split the Independent Features from the Dependent Features
X = dataset.iloc[:, 0: -1].values
y = dataset.iloc[:, -1].values

# We convert the string <sex> = <male/female> and <smoker> = <yes/no> DFs to integer values (0/1), using LabelEncoder()
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 4] = le.fit_transform(X[:, 4])

# We convert the <region> DF to One-Hot-Encoding representation
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X))

# We split tha Dataset into Training Set and Test Set, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Feature Scaling ----- Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sc = StandardScaler()
y_train = sc.fit_transform(y_train.reshape(-1, 1))
y_test = sc.transform(y_test.reshape(-1, 1))

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

class FFNN_Regression(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearLayer1 = nn.Linear(in_features=9, out_features=20)
        self.activation1 = nn.ReLU()

        self.linearLayer2 = nn.Linear(in_features=20, out_features=50)
        self.activation2 = nn.ReLU()

        self.linearLayer3 = nn.Linear(in_features=50, out_features=100)
        self.activation3 = nn.ReLU()

        self.linearLayer4 = nn.Linear(in_features=100, out_features=150)
        self.activation4 = nn.ReLU()

        self.outputLayer = nn.Linear(in_features=150, out_features=1)
        self.activationOutput = nn.Linear(in_features=1, out_features=1)

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set GPU if available


def train(model, train_inputs, train_labels, test_inputs, test_labels, batch_size, num_epochs, criterion, optimizer):
    print("\n\n\n ... The training process ...\n")

    # Set the model/data to GPU if available
    model = model.to(device)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

    train_size = (train_inputs.size())[0]
    batches = torch.arange(0, train_size, batch_size)

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

        end_time = time.perf_counter()
        epoch_time = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {epoch_time: .3f}")


# -------------------------------
# ----- MAIN() ------------------
# -------------------------------

ffnn = FFNN_Regression()

batch_size = 10
number_of_epochs = 150
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.001)

train(model=ffnn, train_inputs=X_train, train_labels=y_train, test_inputs=X_test, test_labels=y_test,
      batch_size=batch_size, num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)

# --------------------------------
# ----- PREDICTION/INFERENCE -----
# --------------------------------

print("\n\n\n----- Now we do some Inference --- some Prediction -----")

ffnn.eval()

prediction_01 = ffnn(torch.FloatTensor(scaler.transform([[0, 0, 0, 1, 19, 0, 27.9, 0, 1]])))
print("\nThe prediction is: " + str(sc.inverse_transform(prediction_01.detach()).item()) +
      " --- The true value/label is: 16884.92400" + " --- So the difference is: " +
      str(sc.inverse_transform(prediction_01.detach()).item() - 16884.92400))

prediction_02 = ffnn(torch.FloatTensor(scaler.transform([[0, 1, 0, 0, 60, 0, 25.84, 0, 0]])))
print("\nThe prediction is: " + str(sc.inverse_transform(prediction_02.detach()).item()) +
      " --- The true value/label is: 28923.13692" + " --- So the difference is: " +
      str(sc.inverse_transform(prediction_02.detach()).item() - 28923.13692))

prediction_03 = ffnn(torch.FloatTensor(scaler.transform([[1, 0, 0, 0, 37, 1, 29.83, 2, 0]])))
print("\nThe prediction is: " + str(sc.inverse_transform(prediction_03.detach()).item()) +
      " --- The true value/label is: 6406.41070" + " --- So the difference is: " +
      str(sc.inverse_transform(prediction_03.detach()).item() - 6406.41070))
