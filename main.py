import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        uniform0 = Uniform( -1 * (1.0 / math.sqrt(d)), (1.0 / math.sqrt(d)) )
        uniform1 = Uniform( -1 * (1.0 / math.sqrt(h)), (1.0 / math.sqrt(h)) )

        self.w0 = Parameter(uniform0.sample(sample_shape=torch.Size([h,d])))
        self.w0.requires_grad = True
        self.w1 = Parameter(uniform1.sample(sample_shape=torch.Size([k,h])))
        self.w1.requires_grad = True

        self.b0 = Parameter(uniform0.sample(sample_shape=torch.Size([1,h])))
        self.b0.requires_grad = True
        self.b1 = Parameter(uniform1.sample(sample_shape=torch.Size([1,k])))
        self.b1.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        inner_net = x @ self.w0.t() + self.b0

        relu_activation = relu(inner_net)

        return relu_activation @ self.w1.t() + self.b1



class F2(Module):
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """
        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        uniform0 = Uniform( -1 * (1.0 / math.sqrt(d)), (1.0 / math.sqrt(d)) )
        uniform1 = Uniform( -1 * (1.0 / math.sqrt(h0)), (1.0 / math.sqrt(h0)) )
        uniform2 = Uniform( -1 * (1.0 / math.sqrt(h1)), (1.0 / math.sqrt(h1)) )

        self.w0 = Parameter(uniform0.sample(sample_shape=torch.Size([h0,d])))
        self.w0.requires_grad = True
        self.w1 = Parameter(uniform1.sample(sample_shape=torch.Size([h1,h0])))
        self.w1.requires_grad = True
        self.w2 = Parameter(uniform2.sample(sample_shape=torch.Size([k,h1])))
        self.w2.requires_grad = True

        self.b0 = Parameter(uniform0.sample(sample_shape=torch.Size([1,h0])))
        self.b0.requires_grad = True
        self.b1 = Parameter(uniform1.sample(sample_shape=torch.Size([1,h1])))
        self.b1.requires_grad = True
        self.b2 = Parameter(uniform2.sample(sample_shape=torch.Size([1,k])))
        self.b2.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        inner_net_0 = x @ self.w0.t() + self.b0
        relu_activation_0 = relu(inner_net_0)

        inner_net_1 = relu_activation_0 @ self.w1.t() + self.b1
        relu_activation_1 = relu(inner_net_1)

        return relu_activation_1 @ self.w2.t() + self.b2


def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    loss_list = []
    epoch = 1
    while True:
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

        loss_list.append(epoch_loss / len(train_loader))

        epoch += 1

        if accuracy > 0.99:
            print("Training stopped. Reached 99% accuracy.")
            break

    return loss_list

def main():
    """
    Main function of this problem.
   
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    model_f1 = F1(h=64, d=784, k=10)
    optimizer_f1 = Adam(model_f1.parameters(), lr=5e-3)
    train_loader_f1 = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    losses_f1 = train(model_f1, optimizer_f1, train_loader_f1)
    evaluate_model(model_f1, x_test, y_test, losses_f1)

    model_f2 = F2(h0=32, h1=32, d=784, k=10)
    optimizer_f2 = Adam(model_f2.parameters(), lr=5e-3)
    train_loader_f2 = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    losses_f2 = train(model_f2, optimizer_f2, train_loader_f2)
    evaluate_model(model_f2, x_test, y_test, losses_f2)


def evaluate_model(model, x_test, y_test, losses):
    y_hat = model(x_test)
    test_preds = torch.argmax(y_hat, 1)
    accuracy_val = torch.sum(test_preds == y_test).item() / len(test_preds)
    test_loss_val = cross_entropy(y_hat, y_test).item()

    print(f"Test Accuracy: {accuracy_val:.4f}")
    print(f"Test Loss: {test_loss_val:.4f}")
    
    plt.plot(range(len(losses)), losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total Number of Parameters: {total_params}")


if __name__ == "__main__":
    main()
