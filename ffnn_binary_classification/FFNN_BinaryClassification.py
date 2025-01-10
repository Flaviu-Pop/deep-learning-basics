import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', None)


path = 'Churn_Modelling.csv'
dataset = pd.read_csv(path)


########################################### ----- DATA STATISTICS ----- ################################################
print('\n\n\nThe first ten rows\observations are:\n')
print(dataset.head(10))

print("\n\n\nThe statistics summary of the dataset is:\n")
print(dataset.describe())

print("\n\n\nThe Object Datatypes are:\n")
print(dataset.dtypes)

num_zeros = (dataset[['RowNumber', 'CustomerId', 'CreditScore', 'Age', 'Tenure',
                      'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']] == 0).sum()
print("\n\n\nThe number of zeros from each column:\n")
print(num_zeros)

print('\n\n\nThe number of NaN for each column is:\n')
print(dataset.isnull().sum())


####################################### ----- DATA PRE-PRPCESSING ----- ################################################
# We teke the Descriptive Features and Target Feature, respectively
X = dataset.iloc[:, 3:-1].values  #Descriptive Feature == Independent Variable
y = dataset.iloc[:, -1].values    #Target Feature == Dependent Variable


# We make the corresponding encodings
# Encoding the "Gender" column ----- LabelEncoder()
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encoding the "Geography" column ----- OneHotEncoder()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# We split the Dataset into Training Set and Test Set, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# We do Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# We convert the datasets into (PyTorch) Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)


print("\n\n\nThe shapes are:")
print(f"X_train: {X_train.size()}")
print(f"X_Test: {X_test.size()}")
print(f"y_train: {y_train.size()}")
print(f"y_test has: {y_test.size()}")


#########################################  THE ARCHITECTURE --- FCNN ###################################################
class FFNN_BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearLayer1 = torch.nn.Linear(in_features=12, out_features=20)
        self.relu1 = torch.nn.ReLU()
        self.linearLayer2 = torch.nn.Linear(in_features=20, out_features=30)
        self.relu2 = torch.nn.ReLU()
        self.linearLayer3 = torch.nn.Linear(in_features=30, out_features=50)
        self.relu3 = torch.nn.ReLU()
        self.linearLayer4 = torch.nn.Linear(in_features=50, out_features=100)
        self.relu4 = torch.nn.ReLU()
        self.linearLayer5 = torch.nn.Linear(in_features=100, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.linearLayer1(x)
        x = self.relu1(x)

        x = self.linearLayer2(x)
        x = self.relu2(x)

        x = self.linearLayer3(x)
        x = self.relu3(x)

        x = self.linearLayer4(x)
        x = self.relu4(x)

        x = self.linearLayer5(x)
        x = self.sigmoid(x)

        return x


###################################### THE TRAINING PROCEDURE ##########################################################
# Set the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_accuracy(model, inputs, labels):
    # It computes the accuracy of the <model> on the dataset <inputs,labels>

    total_correct = 0

    # Set the model/data to GPU is available
    model = model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.eval() # Set the model to evaluation mode

    outputs = model(inputs)
    outputs = torch.round(outputs)

    outputs_shape = list(outputs.size())
    reshape_index_0 = outputs_shape.__getitem__(0)
    outputs = outputs.reshape([reshape_index_0])

    total_correct = (outputs == labels).sum()

    return total_correct/reshape_index_0


def train(model, train_inputs, train_labels, test_inputs, test_labels, batch_size, num_epochs, criterion, optimizer):
    print("\n\n\n--- Now: The Training Process ---\n")

    # We split the Train Set into Training and Validation Sets, respectively
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=0.2, random_state=0)
    print(f"Training Set size = {(train_inputs.size())[0]} --- Validation Set size = {(val_inputs.size())[0]}")


    # Set the model/data to GPU if available
    model = model.to(device)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

    train_size = (train_inputs.size())[0]
    batches = torch.arange(0, train_size, batch_size)

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    # The Training-Loop
    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        model.train()   # Set the model to training mode

        total_loss = 0
        for start_batch in batches:
            # Take a batch
            batch_inputs = train_inputs[start_batch : start_batch + batch_size]
            batch_labels = train_labels[start_batch : start_batch + batch_size]

            # Forward and Backward Passes
            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            batch_outputs = torch.squeeze(batch_outputs, 1)
            batch_loss = criterion(batch_outputs, batch_labels)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss

        # Take the model('s weights) with the best accuracy
        validation_accuracy = compute_accuracy(model, val_inputs, val_labels)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        epoch_time = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {epoch_time: .3f} ===> Validation Accuracy = {validation_accuracy: .4f}  ===> Best Accuracy = {best_accuracy: .4f} at Epoch {best_epoch}")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    print("\n\n\nJust checking ...")
    final_accuracy = compute_accuracy(model, test_inputs, test_labels)
    print(f"\nThe Test Accuracy of the Final Models is: {final_accuracy: .4f}")


################################################# ---  MAIN() ---  #####################################################
ffnn = FFNN_BinaryClassification()

batch_size = 10
number_of_epochs = 50
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.001)

train(model=ffnn, train_inputs=X_train, train_labels=y_train, test_inputs=X_test, test_labels=y_test, batch_size=batch_size, num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)


##################################### -----  PREDICTION/INFERENCE ----- ################################################
print("\n\n\n----- Now: We do some predictions, some inference -----")

ffnn.eval() # Set the model to evaluation mode

prediction_01 = ffnn(torch.FloatTensor(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print("\nThe probability is: " + str(prediction_01.item()))
print("The answer is: " + str((prediction_01 > 0.5).item()))

prediction_02 = ffnn(torch.FloatTensor(sc.transform([[1, 0, 0, 619, 0, 42, 2, 0, 1, 1, 1, 101348.88]])))
print("\nThe probability is: " + str(prediction_02.item()))
print("The answer is: " + str((prediction_02 > 0.5).item()))
print("The true label is: <exit = 1>")

prediction_03 = ffnn(torch.FloatTensor(sc.transform([[0, 0, 1, 608, 0, 41, 1, 83807.86, 1, 0, 1, 112542.58]])))
print("\nThe probability is: " + str(prediction_03.item()))
print("The answer is: " + str((prediction_03 > 0.5).item()))
print("The true label is: <exit = 0>")

prediction_04 = ffnn(torch.FloatTensor(sc.transform([[1, 0, 0, 502, 0, 42, 8, 159660.80, 3, 1, 0, 113931.57]])))
print("\nThe probability is: " + str(prediction_04.item()))
print("The answer is: " + str((prediction_04 > 0.5).item()))
print("The true label is: <exit = 1>")

prediction_05 = ffnn(torch.FloatTensor(sc.transform([[0, 1, 0, 653, 1, 58, 1, 132602.90, 1, 1, 0, 5097.67]])))
print("\nThe probability is: " + str(prediction_05.item()))
print("The answer is: " + str((prediction_05 > 0.5).item()))
print("The true label is: <exit = 1>")

prediction_06 = ffnn(torch.FloatTensor(sc.transform([[1, 0, 0, 411, 1, 29, 0, 59697.17, 2, 1, 1, 53483.21]])))
print("\nThe probability is: " + str(prediction_06.item()))
print("The answer is: " + str((prediction_06 > 0.5).item()))
print("The true label is: <exit = 0>")

prediction_07 = ffnn(torch.FloatTensor(sc.transform([[0, 0, 1, 591, 0, 39, 3, 0, 3, 1, 0, 140469.4]])))
print("\nThe probability is: " + str(prediction_07.item()))
print("The answer is: " + str((prediction_07 > 0.5).item()))
print("The true label is: <exit = 1>")

prediction_08 = ffnn(torch.FloatTensor(sc.transform([[0, 1, 0, 585, 1, 36, 5, 146051, 2, 0, 0, 86424.57]])))
print("\nThe probability is: " + str(prediction_08.item()))
print("The answer is: " + str((prediction_08 > 0.5).item()))
print("The true label is: <exit = 0>")

prediction_09 = ffnn(torch.FloatTensor(sc.transform([[0, 1, 0, 655, 1, 41, 8, 125562, 1, 0, 0, 164040.9]])))
print("\nThe probability is: " + str(prediction_09.item()))
print("The answer is: " + str((prediction_09 > 0.5).item()))
print("The true label is: <exit = 1>")

prediction_10 = ffnn(torch.FloatTensor(sc.transform([[1, 0, 0, 646, 0, 46, 4, 0, 3, 1, 0, 93251.42]])))
print("\nThe probability is: " + str(prediction_10.item()))
print("The answer is: " + str((prediction_10 > 0.5).item()))
print("The true label is: <exit = 1>")
