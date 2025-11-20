import torch
import torch.nn.functional as F
import numpy as np


def compute_threshold_from_calibration(
    model,
    calibration_loader,
    device: str = "cuda",
    alpha: float = 0.1,
) -> float:
    """
    Compute the (1 - alpha) conformal prediction threshold based on a calibration set,
    using -log(predicted probability of the true class) as the nonconformity score
    (i.e., a negative-log-likelihood style score).

    Args:
        model:
            PyTorch model that outputs logits or probabilities with shape
            [B, Classes, Freq, Time] (Freq can be 1).
        calibration_loader:
            DataLoader yielding dicts with keys:
                'input':  tensor [B, C, F, T]
                'target': tensor [B, Classes, F, T] (one-hot over Classes).
        device:
            Computation device, e.g. "cuda" or "cpu".
        alpha:
            Desired miscoverage level (e.g. 0.1 for 90% confidence).

            Typical thresholds (for this specific setup):
                alpha = 0.10 -> thr ≈ 2.45
                alpha = 0.05 -> thr ≈ 3.19
                alpha = 0.02 -> thr ≈ 4.19

    Returns:
        q_alpha:
            Quantile threshold for (1 - alpha) confidence conformal prediction.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in calibration_loader:
            x_calib = batch["input"].to(device)          # [B, C, F, T]
            target = batch["target"].to(device)          # [B, Classes, F, T]

            outputs = model(x_calib)                     # [B, Classes, F, T] (Freq may be 1)
            probs = F.softmax(outputs, dim=1).squeeze(2) # [B, Classes, T] if F=1

            probs = probs.cpu()                          # work on CPU
            target = target[:, :, 0, :].cpu()            # collapse frequency dimension -> [B, Classes, T]

            # Loop over batch and time frames
            batch_size, num_classes, T = probs.shape
            for b in range(batch_size):
                for frame in range(T):
                    # True class index at this frame
                    curr_target = torch.argmax(target[b, :, frame]).item()
                    # Predicted probability of the true class
                    prob = probs[b, curr_target, frame].item()
                    # NLL-style nonconformity score (higher = worse prediction)
                    scores.append(-np.log(prob + 1e-12))

    # Compute the (1 - alpha) quantile over all calibration scores
    all_scores = np.asarray(scores, dtype=float)
    q_alpha = np.quantile(all_scores, 1.0 - alpha)

    # Basic statistics
    print("Calibration statistics:")
    print(f"  Number of scores: {len(all_scores)}")
    print(f"  Mean score: {np.mean(all_scores):.3f}")
    print(f"  Std score:  {np.std(all_scores):.3f}")
    print(f"  {(1 - alpha):.0%} quantile threshold: {q_alpha:.3f}")

    return q_alpha


def do_conformal_prediction(
    result: dict,
    angle_start: int = 10,
    angle_step: int = 5,
    fallback_error: float = 10.0,
    threshold: float = 2.16,
) -> dict:
    """
    Apply conformal prediction at test time using a precomputed threshold.

    This function expects `result` to contain:
        - 'probabilities': array-like of shape [Classes, T] or equivalent,
                           which will be stacked to [1, Classes, T].
        - 'target':        torch.Tensor of shape [1, Classes, T] (one-hot labels).

    It computes frame-wise conformal prediction sets based on the nonconformity
    score S(c) = -log p(c), using `threshold` obtained from calibration.

    For each time frame:
        1. Build prediction set:
               CP_t = { c : S_t(c) < threshold }.
        2. If CP_t is empty, use the argmin of S_t (most probable class) and
           assign `fallback_error` as the predicted error.
        3. Otherwise, define the set of angles corresponding to CP_t and set
           the error to the width (max - min) of that set, clipped by fallback_error.

    The function adds the following fields to `result`:
        - 'CP_sets':      list of length T, each entry an array of class indices.
        - 'coverage_hits': list of length T, 1 if true class ∈ CP_t, else 0.
        - 'CP_coverage':  scalar = mean of coverage_hits.

    Args:
        result:
            Dictionary holding probabilities and targets.
        angle_start:
            Starting angle of the DOA classes (in degrees).
        angle_step:
            Angular resolution between classes (in degrees).
        fallback_error:
            Angular error used when the prediction set is empty.
        threshold:
            Threshold computed from `compute_threshold_from_calibration`.
            Default 2.16 corresponds to one particular calibration run
            (e.g., ~0.1 miscoverage level).

    Returns:
        result:
            The same dict, updated with CP sets and coverage statistics.
    """
    # Probabilities: shape [1, Classes, T]
    probs = np.stack(result["probabilities"], axis=0)
    # Target: one-hot labels [1, Classes, T]
    target_tensor = result["target"]
    gt_class_indices = (
        target_tensor.argmax(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )  # Shape [T]

    CP_error_esti = []
    class_indices_list = []
    coverage_hits = []

    num_classes = probs.shape[1]
    T = probs.shape[2]

    for frame in range(T):
        # Probability distribution over classes at frame t
        pred_dist = probs[0, :, frame]  # [Classes]
        # Nonconformity scores for all classes at frame t
        test_score = -np.log(pred_dist + 1e-12)  # [Classes]

        # Identify indices inside prediction set
        thr_test_bool = test_score < threshold
        class_indices = np.where(thr_test_bool)[0]

        if class_indices.size == 0:
            # Empty prediction set: fall back to most probable class
            class_indices = np.atleast_1d(np.argmin(test_score))
            predicted_angle_error = fallback_error
        else:
            # Map class indices to angles
            prediction_set_angles = class_indices * angle_step + angle_start
            # Width of the prediction interval (in degrees), clipped by fallback_error
            predicted_angle_error = min(
                np.max(prediction_set_angles) - np.min(prediction_set_angles),
                fallback_error,
            )

        class_indices_list.append(class_indices)
        CP_error_esti.append(predicted_angle_error)

        # Coverage: check if the true class is in the prediction set
        is_covered = int(gt_class_indices[frame] in class_indices)
        coverage_hits.append(is_covered)

    result["CP_sets"] = class_indices_list
    result["coverage_hits"] = coverage_hits
    result["CP_coverage"] = float(np.mean(coverage_hits))

    return result
