from matplotlib import pyplot as plt

def plot_losses(train_losses, validation_metrics, n_epochs, model_type):
    """
    Plots the train losses and validation metrics over the specified number of epochs.

    Args:
        train_losses: List of train losses.
        validation_metrics: List of validation metrics (accuracy for LSTM, for example).
        n_epochs: Total number of epochs.
        model_type: Type of the model ("lstm" or "mlp").

    Returns:
        None (displays the plot).
    """
    epochs = range(1, n_epochs + 1)

    if model_type == "lstm":
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, validation_metrics, label='Validation Accuracy')
        plt.ylabel('Accuracy')
    elif model_type == "mlp":
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, validation_metrics, label='Validation Loss')
        plt.ylabel('Loss')

    plt.xlabel('Epoch')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.show()
