import torch
import random
import sys
import os

# Add plotting tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../plotting_tools'))
try:
    from plotting_tools import plot_loss_curve
except ImportError as e:
    print(f"Warning: plotting_tools not found. Error: {e}")
    plot_loss_curve = None

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """Iterate through dataset in batches."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_scratch(features, labels, true_w, true_b):
    print("\n--- Training from Scratch ---")
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    
    loss_history = []

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            current_loss = train_l.mean().item()
            loss_history.append(current_loss)
            print(f'epoch {epoch + 1}, loss {current_loss:f}')

    print(f'error in w: {true_w - w.reshape(true_w.shape)}')
    print(f'error in b: {true_b - b}')
    
    if plot_loss_curve:
        plot_loss_curve(loss_history, label='Scratch Loss', title='Linear Regression (Scratch) Training Loss')

def train_concise(features, labels, true_w, true_b):
    print("\n--- Training with PyTorch API ---")
    from torch.utils import data
    from torch import nn

    def load_array(data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    loss_history = []
    
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        
        l = loss(net(features), labels)
        loss_history.append(l.item())
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    b = net[0].bias.data
    print(f'error in w: {true_w - w.reshape(true_w.shape)}')
    print(f'error in b: {true_b - b}')

    if plot_loss_curve:
        plot_loss_curve(loss_history, label='API Loss', title='Linear Regression (API) Training Loss')

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    train_scratch(features, labels, true_w, true_b)
    train_concise(features, labels, true_w, true_b)

if __name__ == '__main__':
    main()
