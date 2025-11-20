import numpy as np
import torch


def one_hot_to_angle(
    one_hot_tensor,
    angle_step: int = 5,
    angle_start: int = 10,
) -> np.ndarray:
    """
    Convert one-hot (or probability) vectors to DOA angles.

    Args:
        one_hot_tensor:
            Torch tensor or NumPy array of shape [batch, num_classes, width],
            where the class axis corresponds to discrete DOA angles.
            The angle is taken as argmax along the class axis.
        angle_step:
            Step size in degrees between each class.
        angle_start:
            Angle (in degrees) corresponding to class index 0.

    Returns:
        estimated_angles:
            NumPy array of shape [batch, width] with estimated angles in degrees.
    """
    if isinstance(one_hot_tensor, torch.Tensor):
        one_hot_mat = one_hot_tensor.detach().cpu().numpy()
    else:
        one_hot_mat = np.asarray(one_hot_tensor)

    # Get the class index with the maximum value along the 'class' dimension
    # Shape: [batch, width]
    max_indices = np.argmax(one_hot_mat, axis=1)

    # Convert class indices to angles
    estimated_angles = max_indices * angle_step + angle_start  # [batch, width]

    # Compute valid angle range from number of classes
    num_classes = one_hot_mat.shape[1]
    angle_end = angle_start + (num_classes - 1) * angle_step

    # Validate angle range
    if np.any((estimated_angles < angle_start) | (estimated_angles > angle_end)):
        raise ValueError(
            f"Estimated angles must be between {angle_start} and {angle_end} degrees. "
            f"Found min={estimated_angles.min()}, max={estimated_angles.max()}."
        )

    return estimated_angles


def accuracy_at_k(
    probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int = 1,
) -> float:
    """
    Calculate Accuracy@k for predictions with spatial/time dimensions.

    Args:
        probabilities:
            Network outputs (usually softmax probabilities) of shape
            (batch_size, num_classes, width).
        ground_truth:
            Ground truth one-hot vectors of shape (batch_size, num_classes, width).
        k:
            Top-k predictions to consider.

    Returns:
        Accuracy@k over all frames and batch items (scalar).
    """
    # Move to CPU and convert to NumPy
    probs_np = probabilities.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()

    batch_size, num_classes, width = probs_np.shape

    # Get the top-k indices along the class axis
    # Shape: (batch_size, num_classes, width) -> (batch_size, k, width)
    top_k_indices = np.argsort(-probs_np, axis=1)[:, :k, :]

    # Ground truth indices for each time/width position: (batch_size, width)
    ground_truth_indices = np.argmax(gt_np, axis=1)

    # Hit if ground truth class is among top-k predicted classes
    # top_k_indices:    (B, k, W)
    # ground_truth_idx: (B, W) -> (B, 1, W) for broadcasting
    hits = (top_k_indices == ground_truth_indices[:, None, :])  # (B, k, W)
    hits_any = hits.any(axis=1)  # (B, W)

    return float(hits_any.mean())


def mean_absolute_error(
    probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
    angle_step: int = 5,
    angle_start: int = 10,
) -> float:
    """
    Calculate Mean Absolute Error (MAE) between predicted and ground truth DOA.

    Both predictions and ground truth are assumed to be one-hot or probability
    distributions over classes – the DOA is taken as argmax over classes.

    Args:
        probabilities:
            Network outputs, shape (batch_size, num_classes, width).
        ground_truth:
            Ground truth one-hot vectors, same shape as probabilities.
        angle_step:
            Step size in degrees between each class.
        angle_start:
            Angle (in degrees) corresponding to class index 0.

    Returns:
        MAE score (scalar) in degrees.
    """
    predicted_angles = one_hot_to_angle(
        probabilities, angle_step=angle_step, angle_start=angle_start
    )
    ground_truth_angles = one_hot_to_angle(
        ground_truth, angle_step=angle_step, angle_start=angle_start
    )

    # Average over batch and frames
    return float(np.mean(np.abs(predicted_angles - ground_truth_angles)))
