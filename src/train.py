import datetime
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Net
import mlflow
from torchmetrics import Accuracy
from torchinfo import summary

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("tes2")


def train(args, model, device, train_loader, optimizer, loss_fn, metrics_fn, batch_idx):
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    accuracy = metrics_fn(output, target)
    mlflow.log_metric("train-loss", f"{loss:3f}", step=(batch_idx))
    mlflow.log_metric("train-accuracy", f"{accuracy:3f}", step=(batch_idx))
    if batch_idx % args.log_interval == 0:
        print(
            "[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
                accuracy,
            )
        )


def test(args, model, device, test_loader, loss_fn, metrics_fn, batch_idx):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = metrics_fn(output, target)
    if batch_idx % args.log_interval == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} {} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                accuracy,
                100.0 * correct / len(test_loader.dataset),
            )
        )
    mlflow.log_metric("test-loss", f"{test_loss:3f}", step=(batch_idx))
    mlflow.log_metric("test-accuracy", f"{accuracy:3f}", step=(batch_idx))


def get_data_loaders(args, use_cuda):
    # Load the MNIST dataset if it already exists, otherwise download it
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if not os.path.exists("../data"):
        os.makedirs("../data")
        dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    else:
        dataset1 = datasets.MNIST("../data", train=True, transform=transform)
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, test_loader = get_data_loaders(args, use_cuda)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss_fn = F.nll_loss
    # loss_fn = nn.CrossEntropyLoss()
    metrics_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    with mlflow.start_run():
        params = {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "loss_function": loss_fn.__name__,
            "metric_function": metrics_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            for batch_idx in range(len(train_loader)):
                model.train()
                train(args, model, device, train_loader, optimizer, loss_fn, metrics_fn, batch_idx)
                model.eval()
                test(args, model, device, test_loader, loss_fn, metrics_fn, batch_idx)
                if args.dry_run:
                    break
            scheduler.step()

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")

    # if args.save_model:
    #     torch.save(model.state_dict(), f"checkpoints/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}__mnist_cnn.pt")


if __name__ == "__main__":
    main()