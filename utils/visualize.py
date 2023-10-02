import torch
import matplotlib.pyplot as plt

from typing import Any

from utils.tscp import get_tscp_output_scaled


def visualize_predictions(
    model: Any,
    model_type: str,
    sequences_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    n_pics: int = 10,
    save_path: str = None,
    scale: float = 1.,
    device: str = "cpu",
    batch_num_prefix: int = 0,
) -> None:
    """Visualize model's predictions for a batch of test sequences.

    :param model: trained model (e.g. CPDModel or EnsembleCPDModel)
    :param model_type: type of the model ('seq2seq' or 'tscp')
    :param sequences_batch: batch of test sequences
    :param lavels_batch: batch of corresponding labels
    :param n_pics: number of pictures to plot
    :param save_path: if not None, save pictures
    :param scale - scale parameter to obtain TS-CP predictions
    :param device - 'cpu' or 'cuda'
    :batch_num_prefix - used for picture naming in case of several batches 
    """
    model.to(device)
    sequences_batch = sequences_batch.to(device)
    labels_batch = labels_batch.cpu()

    if len(sequences_batch) < n_pics:
        print("Desired number of pictures is greater than size of the batch provided.")
        n_pics = len(sequences_batch)

    if model_type == "seq2seq":
        preds = model(sequences_batch)

    elif model_type == "tscp":
        preds = get_tscp_output_scaled(
            model,
            sequences_batch,
            window_1=model.window_1,
            window_2=model.window_2,
            scale=scale,
        )
    else:
        raise ValueError(f"Unkown model type: {model_type}")

    preds = preds.detach().cpu().squeeze()

    for idx in range(n_pics):
        plt.figure()
        plt.plot(preds[idx], label="Predictions")
        plt.plot(labels_batch[idx], label="Labels")
        plt.title("Model predictions", fontsize=14)
        plt.legend(fontsize=12)
        if save_path is not None:
            plt.savefig(f"{save_path}/batch_{batch_num_prefix}_seq_{idx}.png")
        plt.show()
