import numpy as np


# def build_transition_matrix(
#     num_classes: int,
#     sigma_deg: float = 5.0,
#     doa_step: float = 2.0,
# ) -> np.ndarray:
#     """
#     Build a Gaussian transition matrix for DOA classes.

#     The entry (i, j) is proportional to the probability of transitioning
#     from class j at time t-1 to class i at time t, assuming a Gaussian
#     prior over class index distance.

#     Args:
#         num_classes: Number of DOA classes (e.g., 32 or 90).
#         sigma_deg: Standard deviation (in degrees) for transitions.
#         doa_step: Step size between DOA classes (in degrees).

#     Returns:
#         transition_matrix: Array of shape (num_classes, num_classes),
#             where each column sums to 1.
#     """
#     sigma_classes = sigma_deg / doa_step  # convert sigma to "class index" units
#     transition_matrix = np.zeros((num_classes, num_classes), dtype=float)

#     for s_prev in range(num_classes):
#         for s in range(num_classes):
#             distance = abs(s - s_prev)
#             value = np.exp(-0.5 * (distance / sigma_classes) ** 2)

#             # Cap maximum value to avoid degenerate sharp spikes
#             transition_matrix[s, s_prev] = min(value, 1.0)

#     # Normalize each column so it forms a proper probability distribution
#     transition_matrix /= transition_matrix.sum(axis=0, keepdims=True)

#     return transition_matrix

def build_transition_matrix(
    num_classes: int,
    sigma_deg: float = 5.0,
    doa_step: float = 2.0,
) -> np.ndarray:
    """
    Build a Gaussian transition matrix for DOA classes.
    
    The entry (i, j) is proportional to the probability of transitioning
    from class i at time t-1 to class j at time t, assuming a Gaussian
    prior over class index distance.
    
    Args:
        num_classes: Number of DOA classes (e.g., 32 or 90).
        sigma_deg: Standard deviation (in degrees) for transitions.
        doa_step: Step size between DOA classes (in degrees).

    Returns:
        transition_matrix: Array of shape (num_classes, num_classes),
            where each row sums to 1.
    """
    sigma_classes = sigma_deg / doa_step  # convert sigma to "class index" units

    transition_matrix = np.zeros((num_classes, num_classes), dtype=float)
    
    for i in range(num_classes):
        for j in range(num_classes):
            distance = abs(j - i)
            value = np.exp(-0.5 * (distance / sigma_classes) ** 2)
            # Cap maximum value if needed (optional)
            transition_matrix[i, j] = min(value, 1.0)

    # Normalize each row so it forms a proper probability distribution
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    return transition_matrix

def compute_smoothness(
    CP_set,
    lambda_spacing: float = 3.0,
    lambda_size: float = 0.1,
    lambda_span: float = 0.2,
) -> float:
    """
    Compute a smoothness/confidence score for a conformal prediction (CP) set.

    The score penalizes:
        1. Irregular spacing between elements in the CP set (variance of gaps).
        2. The size of the CP set (number of elements - 1).
        3. The span of the set (max index - min index).

    A higher score (close to 1) corresponds to:
        - small CP set
        - tightly packed indices
        - regular spacing

    The score is:

        smoothness = exp(-[ λ_spacing * var(gaps)
                             + λ_size * (|CP_set| - 1)
                             + λ_span * span ])

    Typical ranges:
        - lambda_spacing: 0.1 (weak) to ~3 (strong), up to 10.
        - lambda_size: 0.1 (weak) to ~3 (strong), up to 10.
        - lambda_span: similar scale to lambda_size.

    Args:
        CP_set: Iterable of int. Indices of DOA classes in the CP set.
        lambda_spacing: Weight for penalizing spacing irregularity.
        lambda_size: Weight for penalizing larger CP sets.
        lambda_span: Weight for penalizing the total span of the set.

    Returns:
        smoothness: Scalar in (0, 1], higher means smoother/more confident.
    """
    CP_set = list(CP_set)

    if len(CP_set) < 2:
        # Singletons (or empty) are treated as maximally smooth
        return 1.0

    CP_sorted = sorted(CP_set)
    gaps = np.diff(CP_sorted)
    var_gaps = np.var(gaps)
    span = CP_sorted[-1] - CP_sorted[0]

    penalty = (
        lambda_spacing * var_gaps
        + lambda_size * (len(CP_set) - 1)
        + lambda_span * span
    )
    smoothness = np.exp(-penalty)

    return float(smoothness)


def initialize_belief(
    softmax_probs: np.ndarray,
    CP_set,
    num_classes: int,
) -> np.ndarray:
    """
    Initialize the belief at t=0 with CP masking.

    Args:
        softmax_probs: Array of shape (num_classes,) – model output at frame 0.
        CP_set: Iterable of int – indices in the conformal prediction set.
        num_classes: Number of classes (e.g., 32).

    Returns:
        belief: Array of shape (num_classes,) – initial normalized belief.
    """
    cp_mask = np.zeros(num_classes, dtype=float)
    cp_mask[list(CP_set)] = 1.0

    belief = softmax_probs * cp_mask

    if belief.sum() > 0:
        belief /= belief.sum()

    return belief


def update_belief(
    prior_belief: np.ndarray,
    softmax_probs: np.ndarray,
    CP_set,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """
    Update the belief at one frame t.

    Args:
        prior_belief: Array of shape (num_classes,) – belief at previous frame.
        softmax_probs: Array of shape (num_classes,) – softmax output at t.
        CP_set: Iterable of int – indices in CP set at frame t.
        transition_matrix: Array of shape (num_classes, num_classes) – transition matrix.

    Returns:
        updated_belief: Array of shape (num_classes,) – updated normalized belief.
    """
    # Predict prior using the transition model
    p_prior = transition_matrix @ prior_belief

    # Compute CP set smoothness to control blending
    w_smooth = compute_smoothness(CP_set)

    # Blend prior and model output
    p_blend = (1.0 - w_smooth) * p_prior + w_smooth * softmax_probs

    # Apply CP mask
    cp_mask = np.zeros_like(p_blend)
    cp_mask[list(CP_set)] = 1.0
    p_blend *= cp_mask

    # Normalize
    if p_blend.sum() > 0:
        p_blend /= p_blend.sum()

    return p_blend


def track_doa(
    result: dict,
    transition_matrix: np.ndarray,
    lambda_spacing: float,
    lambda_size: float,
    lambda_span: float,
) -> dict:
    """
    Track DOA belief over time using:
        - a Markov transition model,
        - conformal prediction masking,
        - and smoothness-based blending with model outputs.

    This function operates in-place on the `result` dict, adding:
        - 'belief_over_time': np.ndarray of shape (num_classes, T)
        - 'filter_doa':       np.ndarray of shape (T,)

    Expected `result` fields:
        - 'probabilities': torch.Tensor or np.ndarray with shape (1, num_classes, T)
        - 'CP_sets': list of length T, each entry is an iterable of class indices.

    Args:
        result: Dictionary holding model outputs and metadata.
        transition_matrix: Array of shape (num_classes, num_classes).
        lambda_spacing: Parameter passed to `compute_smoothness`.
        lambda_size: Parameter passed to `compute_smoothness`.
        lambda_span: Parameter passed to `compute_smoothness`.

    Returns:
        result: The same dict, with 'belief_over_time' and 'filter_doa' added.
    """
    # These depend on your DOA discretization (e.g., 10°, 15°, ..., 165°).
    starting_angle = 10
    angle_step = 5

    softmax_all = result["probabilities"].numpy()  # [1, num_classes, T]
    CP_sets = result["CP_sets"]

    _, num_classes, T = softmax_all.shape
    belief_over_time = np.zeros((num_classes, T), dtype=float)

    # --- Initialize at t = 0 ---
    softmax_probs_0 = softmax_all[0, :, 0]
    initial_belief = softmax_probs_0.copy()
    if initial_belief.sum() > 0:
        initial_belief /= initial_belief.sum()

    belief_over_time[:, 0] = initial_belief

    filter_doa = np.empty(T, dtype=float)
    filter_doa[0] = starting_angle + np.argmax(initial_belief) * angle_step

    # --- Track over time (t >= 1) ---
    for t in range(1, T):
        prior_belief = belief_over_time[:, t - 1]
        softmax_probs_t = softmax_all[0, :, t]
        CP_set_t = CP_sets[t]

        # Predict prior
        # p_prior = transition_matrix @ prior_belief
        p_prior = transition_matrix.T @ prior_belief

        # Compute CP set smoothness and blend
        w_smooth = compute_smoothness(
            CP_set_t, lambda_spacing=lambda_spacing,
            lambda_size=lambda_size, lambda_span=lambda_span
        )
        p_blend = (1.0 - w_smooth) * p_prior + w_smooth * softmax_probs_t

        # Normalize
        if p_blend.sum() > 0:
            p_blend /= p_blend.sum()

        belief_over_time[:, t] = p_blend
        filter_doa[t] = starting_angle + np.argmax(p_blend) * angle_step

    result["belief_over_time"] = belief_over_time
    result["filter_doa"] = filter_doa

    return result
