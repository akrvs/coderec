from matplotlib import pyplot as plt

def plot_losses(train_losses, val_losses, n_epochs):
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
