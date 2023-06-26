from matplotlib import pyplot as plt

def plot_losses(train_losses, val_losses, n_epochs):
    """
    Plots the train and validation losses over the specified number of epochs.

    Args:
        train_losses: List of train losses.
        val_losses: List of validation losses.
        n_epochs: Total number of epochs.

    Returns:
        None (displays the plot).
    """
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
