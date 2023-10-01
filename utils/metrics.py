"""Module with functions for metrics calculation."""
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import gc

from tqdm import tqdm

import pytorch_lightning as pl

from utils import cpd_models, tscp

# ------------------------------------------------------------------------------------------------------------#
#                         Evaluate seq2seq, KL-CPD and TS-CP2 baseline models                                #
# ------------------------------------------------------------------------------------------------------------#


def find_first_change(mask: np.array) -> np.array:
    """Find first change in batch of predictions.

    :param mask:
    :return: mask with -1 on first change
    """
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind


def calculate_errors(
    real: torch.Tensor, pred: torch.Tensor, seq_len: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real true change points idxs for a batch
    :param pred: predicted change point idxs for a batch
    :param seq_len: length of sequence
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
    """
    FP_delay = torch.zeros_like(real, requires_grad=False)
    delay = torch.zeros_like(real, requires_grad=False)

    tn_mask = torch.logical_and(real == pred, real == -1)
    fn_mask = torch.logical_and(real != pred, pred == -1)
    tp_mask = torch.logical_and(real <= pred, real != -1)
    fp_mask = torch.logical_or(
        torch.logical_and(torch.logical_and(real > pred, real != -1), pred != -1),
        torch.logical_and(pred != -1, real == -1),
    )

    TN = tn_mask.sum().item()
    FN = fn_mask.sum().item()
    TP = tp_mask.sum().item()
    FP = fp_mask.sum().item()

    FP_delay[tn_mask] = seq_len
    FP_delay[fn_mask] = seq_len
    FP_delay[tp_mask] = real[tp_mask]
    FP_delay[fp_mask] = pred[fp_mask]

    delay[tn_mask] = 0
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    delay[fp_mask] = 0

    assert (TN + TP + FN + FP) == len(real)

    return TN, FP, FN, TP, FP_delay, delay


def calculate_conf_matrix_margin(
    real: torch.Tensor, pred: torch.Tensor, margin: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real labels of change points
    :param pred: predicted labels (0 or 1) of change points
    :param margin: if |true_cp_idx - pred_cp_idx| <= margin, report TP
    :return: tuple of (TN, FP, FN, TP)
    """
    tn_mask_margin = torch.logical_and(real == pred, real == -1)
    fn_mask_margin = torch.logical_and(real != pred, pred == -1)

    tp_mask_margin = torch.logical_and(
        torch.logical_and(torch.abs(real - pred) <= margin, real != -1), pred != -1
    )

    fp_mask_margin = torch.logical_or(
        torch.logical_and(
            torch.logical_and(torch.abs(real - pred) > margin, real != -1), pred != -1
        ),
        torch.logical_and(pred != -1, real == -1),
    )

    TN_margin = tn_mask_margin.sum().item()
    FN_margin = fn_mask_margin.sum().item()
    TP_margin = tp_mask_margin.sum().item()
    FP_margin = fp_mask_margin.sum().item()

    assert (TN_margin + TP_margin + FN_margin + FP_margin) == len(
        real
    ), "Check TP, TN, FP, FN cases."

    return TN_margin, FP_margin, FN_margin, TP_margin


def calculate_metrics(
    true_labels: torch.Tensor, predictions: torch.Tensor, margin_list: List[int] = None
) -> Tuple[int, int, int, int, np.array, np.array, int]:
    """Calculate confusion matrix, detection delay, time to false alarms, covering.

    :param true_labels: true labels (0 or 1) of change points
    :param predictions: predicted labels (0 or 1) of change points
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
        - covering
    """
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)

    TN, FP, FN, TP, FP_delay, delay = calculate_errors(
        real_change_ind, predicted_change_ind, seq_len
    )
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    # add margin metrics
    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = {}, {}, {}, {}
        for margin in margin_list:
            TN_margin, FP_margin, FN_margin, TP_margin = calculate_conf_matrix_margin(
                real_change_ind, predicted_change_ind, margin
            )
            TN_margin_dict[margin] = TN_margin
            FP_margin_dict[margin] = FP_margin
            FN_margin_dict[margin] = FN_margin
            TP_margin_dict[margin] = TP_margin

    return (TN, FP, FN, TP, FP_delay, delay, cover), (
        TN_margin_dict,
        FP_margin_dict,
        FN_margin_dict,
        TP_margin_dict,
    )


def get_models_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    model_type: str = "seq2seq",
    device: str = "cuda",
    scale: int = None,
) -> List[torch.Tensor]:
    """Get model's prediction.

    :param inputs: input data
    :param labels: true labels
    :param model: CPD model
    :param model_type: default "seq2seq" for BCE model, "klcpd" for KLCPD model
    :param device: device name
    :param scales: scale parameter for KL-CPD predictions
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: model's predictions
    """
    try:
        inputs = inputs.to(device)
    except:
        inputs = [t.to(device) for t in inputs]

    true_labels = labels.to(device)

    if model_type == "tscp":
        outs = tscp.get_tscp_output_scaled(
            model, inputs, model.window_1, model.window_2, scale=scale
        )
        uncertainties = None

    elif model_type == "seq2seq":
        outs = model(inputs)
        uncertainties = None

    else:
        raise ValueError(f"Wrong model type {model_type}.")

    return outs, uncertainties, true_labels


def collect_model_predictions_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    verbose: bool = True,
    model_type: str = "seq2seq",
    device: str = "cuda",
    scale: int = None,
):
    if model is not None:
        model.eval()
        model.to(device)

    test_out_bank, test_uncertainties_bank, test_labels_bank = [], [], []

    with torch.no_grad():
        if verbose:
            print("Collectting model's outputs")

        # collect model's predictions once and reuse them
        for test_inputs, test_labels in tqdm(test_loader):
            test_out, test_uncertainties, test_labels = get_models_predictions(
                test_inputs,
                test_labels,
                model,
                model_type=model_type,
                device=device,
                scale=scale,
            )

            try:
                test_out = test_out.squeeze(2)
                test_uncertainties = test_uncertainties.squeeze(2)
            except:
                try:
                    test_out = test_out.squeeze(1)
                    test_uncertainties = test_uncertainties.squeeze(1)
                except:
                    test_out = test_out
                    test_uncertainties = test_uncertainties

            # in case of different sizes, crop start of labels sequence (for TS-CP)
            crop_size = test_labels.shape[1] - test_out.shape[1]
            test_labels = test_labels[:, crop_size:]

            test_out_bank.append(test_out.cpu())
            test_uncertainties_bank.append(
                test_uncertainties.cpu()
                if test_uncertainties is not None
                else test_uncertainties
            )
            test_labels_bank.append(test_labels.cpu())

    del test_labels, test_out, test_uncertainties, test_inputs
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return test_out_bank, test_uncertainties_bank, test_labels_bank


def evaluate_metrics_on_set(
    test_out_bank: List[torch.Tensor],
    test_uncertainties_bank: List[torch.Tensor],
    test_labels_bank: List[torch.Tensor],
    threshold: float = 0.5,
    verbose: bool = True,
    device: str = "cuda",
    uncert_th: float = None,
    margin_list: List[int] = None,
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.

    :param model: trained CPD model for evaluation
    :param test_loader: dataloader with test data
    :param threshold: alarm threshold (if change prob > threshold, report about a CP)
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param device: 'cuda' or 'cpu'
    :param scale: scale factor (for KL-CPD and TSCP models)
    :param uncert_th: std threshold for CPD-with-rejection, set to 'None' if not rejection is needed
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: tuple of
        - TN, FP, FN, TP
        - mean time to a false alarm
        - mean detection delay
        - mean covering
    """

    FP_delays = []
    delays = []
    covers = []
    TN, FP, FN, TP = (0, 0, 0, 0)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
            {},
            {},
            {},
            {},
        )
        for margin in margin_list:
            TN_margin_dict[margin] = 0
            FP_margin_dict[margin] = 0
            FN_margin_dict[margin] = 0
            TP_margin_dict[margin] = 0

    with torch.no_grad():
        for test_out, test_uncertainties, test_labels in zip(
            test_out_bank, test_uncertainties_bank, test_labels_bank
        ):
            if test_uncertainties is not None and uncert_th is not None:
                cropped_outs = (test_out > threshold) & (test_uncertainties < uncert_th)

            else:
                cropped_outs = test_out > threshold

            (
                (tn, fp, fn, tp, FP_delay, delay, cover),
                (tn_margin_dict, fp_margin_dict, fn_margin_dict, tp_margin_dict),
            ) = calculate_metrics(test_labels, cropped_outs, margin_list)

            TN += tn
            FP += fp
            FN += fn
            TP += tp

            if margin_list is not None:
                for margin in margin_list:
                    TN_margin_dict[margin] += tn_margin_dict[margin]
                    FP_margin_dict[margin] += fp_margin_dict[margin]
                    FN_margin_dict[margin] += fn_margin_dict[margin]
                    TP_margin_dict[margin] += tp_margin_dict[margin]

            FP_delays.append(FP_delay.detach().cpu())
            delays.append(delay.detach().cpu())
            covers.extend(cover)

    mean_FP_delay = torch.cat(FP_delays).float().mean().item()
    mean_delay = torch.cat(delays).float().mean().item()
    mean_cover = np.mean(covers)

    if verbose:
        print(
            "TN: {}, FP: {}, FN: {}, TP: {}, DELAY:{}, FP_DELAY:{}, COVER: {}".format(
                TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover
            )
        )

    del FP_delays, delays, covers
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return (
        (TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover),
        (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
    )


def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


def overlap(A: set, B: set):
    """Return the overlap (i.e. Jaccard index) of two sets.

    :param A: set #1
    :param B: set #2
    return Jaccard index of the 2 sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations: List[int], n_obs: int) -> List[set]:
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    :param locations: idxs of the change points
    :param n_obs: length of the sequence
    :return partition of the sequence (list of sets with idxs)
    """
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(true_partitions: List[set], pred_partitions: List[set]) -> float:
    """Compute the covering of a true segmentation by a predicted segmentation.

    :param true_partitions: partition made by true CPs
    :param true_partitions: partition made by predicted CPs
    """
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))

    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(
            overlap(t_part, p_part) for p_part in pred_partitions
        )
    cover /= seq_len
    return cover


def calculate_cover(
    real_change_ind: List[int], predicted_change_ind: List[int], seq_len: int
) -> List[float]:
    """Calculate covering for a given sequence.

    :param real_change_ind: indexes of true CPs
    :param predicted_change_ind: indexes of predicted CPs
    :param seq_len: length of the sequence
    :return cover
    """
    covers = []

    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))
    return covers


def F1_score(confusion_matrix: Tuple[int, int, int, int]) -> float:
    """Calculate F1-score.

    :param confusion_matrix: tuple with elements of the confusion matrix
    :return: f1_score
    """
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


def evaluation_pipeline(
    model: pl.LightningModule,
    test_dataloader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
    verbose: bool = False,
    model_type: str = "seq2seq",
    scale: int = None,
    uncert_th: float = None,
    margin_list: List[int] = None,
) -> Tuple[Tuple[float], dict, dict]:
    """Evaluate trained CPD model.

    :param model: trained CPD model to be evaluated
    :param test_dataloader: test data for evaluation
    :param threshold_list: listh of alarm thresholds
    :param device: 'cuda' or 'cpu'
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param scale: scale factor (for KL-CPD and TSCP models)
    :param uncert_th: std threshold for CPD-with-rejection, set to 'None' if not rejection is needed
    :return: tuple of
        - threshold th_1 corresponding to the maximum F1-score
        - mean time to a False Alarm corresponding to th_1
        - mean Detection Delay corresponding to th_1
        - Area under the Detection Curve
        - number of TN, FP, FN, TP corresponding to th_1
        - value of Covering corresponding to th_1
        - threshold th_2 corresponding to the maximum Covering metric
        - maximum value of Covering
    """
    try:
        model.to(device)
        model.eval()
    except:
        print("Cannot move model to device")

    (
        test_out_bank,
        test_uncertainties_bank,
        test_labels_bank,
    ) = collect_model_predictions_on_set(
        model=model,
        test_loader=test_dataloader,
        verbose=verbose,
        model_type=model_type,
        device=device,
        scale=scale,
    )

    cover_dict = {}
    f1_dict = {}

    if margin_list is not None:
        final_f1_margin_dict = {}

    delay_dict = {}
    fp_delay_dict = {}
    confusion_matrix_dict = {}

    if model_type == "cusum_aggr":
        if verbose and len(threshold_list) > 1:
            print("No need in threshold list for CUSUM. Take threshold = 0.5.")
        threshold_list = [0.5]

    for threshold in threshold_list:
        final_f1_margin_dict[threshold] = {}

        (
            (TN, FP, FN, TP, mean_delay, mean_fp_delay, cover),
            (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
        ) = evaluate_metrics_on_set(
            test_out_bank=test_out_bank,
            test_uncertainties_bank=test_uncertainties_bank,
            test_labels_bank=test_labels_bank,
            threshold=threshold,
            verbose=verbose,
            device=device,
            uncert_th=uncert_th,
            margin_list=margin_list,
        )

        confusion_matrix_dict[threshold] = (TN, FP, FN, TP)
        delay_dict[threshold] = mean_delay
        fp_delay_dict[threshold] = mean_fp_delay

        cover_dict[threshold] = cover
        f1_dict[threshold] = F1_score((TN, FP, FN, TP))

        if margin_list is not None:
            f1_margin_dict = {}
            for margin in margin_list:
                (TN_margin, FP_margin, FN_margin, TP_margin) = (
                    TN_margin_dict[margin],
                    FP_margin_dict[margin],
                    FN_margin_dict[margin],
                    TP_margin_dict[margin],
                )
                f1_margin_dict[margin] = F1_score(
                    (TN_margin, FP_margin, FN_margin, TP_margin)
                )
            final_f1_margin_dict[threshold] = f1_margin_dict

    # fix dict structure
    if margin_list is not None:
        final_f1_margin_dict_fixed = {}
        for margin in margin_list:
            f1_scores_for_margin_dict = {}
            for threshold in threshold_list:
                f1_scores_for_margin_dict[threshold] = final_f1_margin_dict[threshold][
                    margin
                ]
            final_f1_margin_dict_fixed[margin] = f1_scores_for_margin_dict

    if model_type == "cusum_aggr":
        auc = None
    else:
        auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))

    # Conf matrix and F1
    best_th_f1 = max(f1_dict, key=f1_dict.get)

    best_conf_matrix = (
        confusion_matrix_dict[best_th_f1][0],
        confusion_matrix_dict[best_th_f1][1],
        confusion_matrix_dict[best_th_f1][2],
        confusion_matrix_dict[best_th_f1][3],
    )
    best_f1 = f1_dict[best_th_f1]

    # Cover
    best_cover = cover_dict[best_th_f1]

    best_th_cover = max(cover_dict, key=cover_dict.get)
    max_cover = cover_dict[best_th_cover]

    if margin_list is not None:
        max_f1_margins_dict = {}
        max_th_f1_margins_dict = {}
        for margin in margin_list:
            curr_max_th_f1_margin = max(
                final_f1_margin_dict_fixed[margin],
                key=final_f1_margin_dict_fixed[margin].get,
            )
            max_th_f1_margins_dict[margin] = curr_max_th_f1_margin
            max_f1_margins_dict[margin] = final_f1_margin_dict_fixed[margin][
                curr_max_th_f1_margin
            ]
    else:
        max_f1_margins_dict, max_th_f1_margins_dict = None, None

    # Time to FA, detection delay
    best_time_to_FA = fp_delay_dict[best_th_f1]
    best_delay = delay_dict[best_th_f1]

    print("AUC:", round(auc, 4) if auc is not None else auc)
    print(
        "Time to FA {}, delay detection {} for best-F1 threshold: {}".format(
            round(best_time_to_FA, 4), round(best_delay, 4), round(best_th_f1, 4)
        )
    )
    print(
        "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}".format(
            best_conf_matrix[0],
            best_conf_matrix[1],
            best_conf_matrix[2],
            best_conf_matrix[3],
            round(best_th_f1, 4),
        )
    )
    print(
        "Max F1 {}: for best-F1 threshold {}".format(
            round(best_f1, 4), round(best_th_f1, 4)
        )
    )
    print(
        "COVER {}: for best-F1 threshold {}".format(
            round(best_cover, 4), round(best_th_f1, 4)
        )
    )

    print(
        "Max COVER {}: for threshold {}".format(
            round(cover_dict[max(cover_dict, key=cover_dict.get)], 4),
            round(max(cover_dict, key=cover_dict.get), 4),
        )
    )
    if margin_list is not None:
        for margin in margin_list:
            print(
                "Max F1 with margin {}: {} for best threshold {}".format(
                    margin,
                    round(max_f1_margins_dict[margin], 4),
                    round(max_th_f1_margins_dict[margin], 4),
                )
            )

    return (
        (
            best_th_f1,
            best_time_to_FA,
            best_delay,
            auc,
            best_conf_matrix,
            best_f1,
            best_cover,
            best_th_cover,
            max_cover,
        ),
        (max_th_f1_margins_dict, max_f1_margins_dict),
        delay_dict,
        fp_delay_dict,
    )


# ------------------------------------------------------------------------------------------------------------#
#                                      Evaluate classic baselines                                            #
# ------------------------------------------------------------------------------------------------------------#


def get_classic_baseline_predictions(
    dataloader: DataLoader,
    baseline_model: cpd_models.ClassicBaseline,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get predictions of a classic baseline model.

    :param dataloader: validation dataloader
    :param baseline_model: core model of a classic baseline (from ruptures package)
    :return: tuple of
        - predicted labels
        - true pabels
    """
    all_predictions = []
    all_labels = []
    for inputs, labels in dataloader:
        all_labels.append(labels)
        baseline_pred = baseline_model(inputs)
        all_predictions.append(baseline_pred)

    all_labels = torch.from_numpy(np.vstack(all_labels))
    all_predictions = torch.from_numpy(np.vstack(all_predictions))
    return all_predictions, all_labels


def classic_baseline_metrics(
    all_labels: torch.Tensor,
    all_preds: torch.Tensor,
    threshold: float = 0.5,
    margin_list: List[int] = None,
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """Calculate metrics for a classic baseline model.

    :param all_labels: tensor of true labels
    :param all_preds: tensor of predictions
    :param threshold: alarm threshold (=0.5 for classic models)
    :return: turple of metrics
        - best threshold for F1-score (always 0.5)
        - mean Time to a False Alarm
        - mean Detection Delay
        - None (no AUC metric for classic baselines)
        - best confusion matrix (number of TN, FP, FN and TP predictions)
        - F1-score
        - covering metric
        - best thresold for covering metric (always 0.5)
        - covering metric
    Note that we return some unnecessary values for consistency with our general evaluation pipeline.
    """
    (
        (TN, FP, FN, TP, FP_delay, delay, cover),
        (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
    ) = calculate_metrics(all_labels, all_preds > threshold, margin_list)

    f1 = F1_score((TN, FP, FN, TP))
    FP_delay = torch.mean(FP_delay.float()).item()
    delay = torch.mean(delay.float()).item()
    cover = np.mean(cover)

    if margin_list is not None:
        f1_margin_dict = dict()
        for m in margin_list:
            f1_margin_dict[m] = F1_score(
                (
                    TN_margin_dict[m],
                    FP_margin_dict[m],
                    FN_margin_dict[m],
                    TP_margin_dict[m],
                )
            )
    else:
        f1_margin_dict = None

    return (
        0.5,
        FP_delay,
        delay,
        None,
        (TN, FP, FN, TP),
        f1,
        f1_margin_dict,
        cover,
        0.5,
        cover,
    )


def calculate_baseline_metrics(
    model: cpd_models.ClassicBaseline,
    val_dataloader: DataLoader,
    verbose: bool = False,
    margin_list: List[int] = None,
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """Calculate metrics for a classic baseline model.

    :param model: core model of a classic baseline (from ruptures package)
    :param val_dataloader: validation dataloader
    :param verbose: if true, print the metrics to the console
    :return: tuple of metrics (see 'classic_baseline_metrics' function)
    """
    pred, labels = get_classic_baseline_predictions(val_dataloader, model)
    metrics = classic_baseline_metrics(labels, pred, margin_list=margin_list)
    (
        _,
        mean_FP_delay,
        mean_delay,
        _,
        (TN, FP, FN, TP),
        f1,
        f1_margin_dict,
        cover,
        _,
        _,
    ) = metrics

    if verbose:
        print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
        print(
            f"DELAY: {np.round(mean_delay, 2)}, FP_DELAY:{np.round(mean_FP_delay, 2)}"
        )
        print(f"F1:{np.round(f1, 4)}")
        print(f"COVER: {np.round(cover, 4)}")
        if margin_list is not None:
            for margin in margin_list:
                print(
                    "Max F1 with margin {}: {}".format(
                        margin, np.round(f1_margin_dict[margin], 4)
                    )
                )
    return metrics
