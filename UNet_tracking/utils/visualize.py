import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

eps = sys.float_info.epsilon


def plot_img_and_mask(img, mask):
    """
    Quick helper to visualize an image with its segmentation masks.
    """
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f"Mask (class {i + 1})")
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_and_save(signals_stft, vad_binary, output_prefix="STFT_and_VAD"):
    """
    Plot log-magnitude STFT and time-frequency VAD mask for each microphone.

    Args:
        signals_stft: array-like of shape (F, T, M)
        vad_binary:   array-like of shape (F, T, M) or broadcastable
        output_prefix: prefix for saved JPEG files.
    """
    # Log-magnitude for visualization
    signals_stft = np.log(np.abs(signals_stft) + 1e-10)
    vad_binary = np.array(vad_binary)

    num_freq_bins, num_frames, num_mics = signals_stft.shape

    for mic in range(num_mics):
        plt.figure(figsize=(12, 8))

        # STFT magnitude
        plt.subplot(2, 1, 1)
        plt.imshow(
            signals_stft[:, :, mic],
            aspect="auto",
            origin="lower",
            extent=[0, num_frames, 0, num_freq_bins],
            cmap="jet",
        )
        plt.title(f"Log-Magnitude of STFT (Mic {mic + 1}) – note: not the raw input!")
        plt.xlabel("Frame")
        plt.ylabel("Frequency Bin")
        plt.colorbar(label="Magnitude")
        plt.grid(False)

        # VAD mask
        plt.subplot(2, 1, 2)
        plt.imshow(
            vad_binary[:, :, mic],
            aspect="auto",
            origin="lower",
            extent=[0, num_frames, 0, 1],
            cmap="gray",
            alpha=0.5,
        )
        plt.title(f"Voice Activity Detection (VAD) – Mic {mic + 1}")
        plt.xlabel("Frame")
        plt.ylabel("VAD Decision")
        plt.ylim(-0.1, 1.1)
        plt.colorbar(label="VAD Decision", ticks=[0, 1], format="%.0f")
        plt.grid(False)

        plt.tight_layout()
        plt.savefig(
            f"dataset_verification/{output_prefix}_mic_{mic + 1}.jpeg",
            format="jpeg",
            dpi=300,
        )
        plt.close()


def plot_and_save_irf(irtf, vad_binary, output_prefix="IRT_and_VAD"):
    """
    Plot imaginary part of IRTF and VAD mask for each microphone.

    Args:
        irtf:        array-like of shape (F, T, M)
        vad_binary:  array-like of shape (F, T, M)
        output_prefix: prefix for saved JPEG files.
    """
    irtf = np.imag(irtf)
    vad_binary = np.array(vad_binary)

    num_freq_bins, num_frames, num_mics = irtf.shape

    for mic in range(num_mics):
        plt.figure(figsize=(12, 8))

        # IRTF
        plt.subplot(2, 1, 1)
        plt.imshow(
            irtf[:, :, mic],
            aspect="auto",
            origin="lower",
            extent=[0, num_frames, 0, num_freq_bins],
            cmap="jet",
        )
        plt.title(f"Imaginary part of IRTF (Mic {mic + 1})")
        plt.xlabel("Frame")
        plt.ylabel("Frequency Bin")
        plt.colorbar(label="Magnitude")
        plt.grid(False)

        # VAD mask
        plt.subplot(2, 1, 2)
        plt.imshow(
            vad_binary[:, :, mic],
            aspect="auto",
            origin="lower",
            extent=[0, num_frames, 0, 1],
            cmap="gray",
            alpha=0.5,
        )
        plt.title(f"Voice Activity Detection (VAD) – Mic {mic + 1}")
        plt.xlabel("Frame")
        plt.ylabel("VAD Decision")
        plt.ylim(-0.1, 1.1)
        plt.colorbar(label="VAD Decision", ticks=[0, 1], format="%.0f")
        plt.grid(False)

        plt.tight_layout()
        plt.savefig(
            f"dataset_verification/{output_prefix}_mic_{mic + 1}.jpeg",
            format="jpeg",
            dpi=300,
        )
        plt.close()


def plot_error_vs_true_DOA(true_doa_mapping, doa_error, name="DOA_error_vs_true_DOA.jpeg"):
    """
    Plot DOA error as a function of true DOA (sorted by true DOA).
    """
    true_doa_mapping = np.array(true_doa_mapping)
    doa_error = np.array(doa_error)

    sorted_indices = np.argsort(true_doa_mapping)
    true_sorted = true_doa_mapping[sorted_indices]
    err_sorted = doa_error[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(true_sorted, err_sorted, marker="o", linestyle="-", color="b")
    plt.xlabel("True DOA (degrees)")
    plt.ylabel("DOA Error (degrees)")
    plt.title("DOA Error vs True DOA")
    plt.savefig(name, format="jpeg", dpi=300)
    plt.close()


def plot_histograms_for_all_DOA(GT_angles_list, model_angles_list, name):
    """
    For each unique true DOA, plot a histogram of predicted DOAs.
    """
    GT_angles_list = np.array(GT_angles_list)
    model_angles_list = np.array(model_angles_list)

    unique_true_doas = np.unique(GT_angles_list)
    fig, axes = plt.subplots(8, 4, figsize=(15, 20))
    axes = axes.flatten()

    for i, true_doa in enumerate(unique_true_doas):
        model_predictions = model_angles_list[GT_angles_list == true_doa]
        # model_predictions is a list/array of per-frame lists
        flattened_predictions = np.concatenate(
            [np.array(pred).flatten() for pred in model_predictions]
        )

        axes[i].hist(flattened_predictions, bins=30, color="blue", alpha=0.7)
        axes[i].set_title(f"True DOA: {true_doa}°")
        axes[i].set_xlabel("Predicted DOA (degrees)")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(name, format="jpeg", dpi=300)
    plt.close()


def plot_doa_heatmap(
    predictions,
    true_doa,
    angle_step: int = 5,
    debug_level: int = 0,
    name: str = "test_plots/Heat_map_plot.jpeg",
    cmap: str = "viridis",
):
    """
    OLD VERSION (without CP):

    Plot a heatmap of angle probabilities over time with the true DOA (class index)
    as a dashed line.

    Args:
        predictions: array-like [B, Classes, T]
        true_doa:    torch.Tensor [B, Classes, T] (one-hot)
    """
    start_angle = 10
    end_angle = 170

    predictions = predictions[0, :, :]  # [Classes, T]
    true_doa_target = true_doa[0, :, :].cpu()  # [Classes, T]

    angles = np.arange(start_angle, end_angle, angle_step)
    true_doa_deg = angles[torch.argmax(true_doa_target, dim=0).numpy()]  # [T]
    true_doa_index = np.round((true_doa_deg - start_angle) / angle_step).astype(int)

    time_frames = np.arange(predictions.shape[1])

    plt.figure(figsize=(15, 11))
    plt.imshow(
        predictions,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="none",
    )
    plt.plot(time_frames, true_doa_index, color="red", linestyle="--", linewidth=2, label="True DOA")
    plt.xlabel("Time (frames)")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Probabilities Over Time")
    plt.colorbar(label="Probability")
    plt.yticks(
        np.linspace(0, predictions.shape[0] - 1, len(angles)),
        angles,
    )

    plt.legend()
    if debug_level == 2:
        plt.savefig(name, format="jpeg", dpi=300)
    plt.close()


def plot_CP_DOA_results(
    predictions,
    true_doa,
    CP_sets,
    belief_over_time,
    name: str,
    angle_step: int = 5,
    debug_level: int = 0,
    cmap: str = "viridis",
):
    """
    CP-aware visualization:

    - Heatmap of angle probabilities with:
        - true DOA (red dashed)
        - CP interval (orange band)
        - argmax prediction (black dots)
    - Heatmap of belief_over_time with true DOA + argmax belief.

    Args:
        predictions:      [B, Classes, T] (torch or numpy)
        true_doa:         [B, Classes, T] (torch one-hot)
        CP_sets:          list of length T, each entry iterable of class indices
        belief_over_time: [Classes, T] numpy or torch
    """
    start_angle = 10
    end_angle = 170

    # First batch only
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions[0].detach().cpu().numpy()
    else:
        predictions_np = np.asarray(predictions)[0]

    true_doa_target = true_doa[0, :, :].cpu()  # [Classes, T]
    angles = np.arange(start_angle, end_angle, angle_step)

    # True DOA curve
    true_doa_deg = angles[torch.argmax(true_doa_target, dim=0).numpy()]  # [T]
    true_doa_index = np.round((true_doa_deg - start_angle) / angle_step).astype(int)

    time_frames = np.arange(predictions_np.shape[1])

    # CP lower/upper bounds in class index
    cp_lower, cp_upper = [], []
    for cp_set in CP_sets:
        if len(cp_set) > 0:
            cp_lower.append(min(cp_set))
            cp_upper.append(max(cp_set))
        else:
            cp_lower.append(0)
            cp_upper.append(0)

    # ---- Plot probability heatmap with CP band ----
    plt.figure(figsize=(10, 6))
    plt.imshow(
        predictions_np,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="none",
    )
    plt.plot(time_frames, true_doa_index, color="red", linestyle="--", linewidth=2, label="True DOA")
    plt.fill_between(time_frames, cp_lower, cp_upper, color="orange", alpha=0.3, label="CP Interval")

    pred_argmax = np.argmax(predictions_np, axis=0)
    plt.scatter(time_frames, pred_argmax, color="black", marker="o", s=10, label="Predicted Max")

    plt.xlabel("Time (frames)")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Probabilities Over Time")
    plt.colorbar(label="Probability")
    plt.yticks(np.linspace(0, predictions_np.shape[0] - 1, len(angles)), angles)
    plt.legend(loc="upper left")

    if debug_level == 2:
        plt.savefig(name, format="jpeg", dpi=200)
    plt.close()

    # ---- Plot belief_over_time heatmap ----
    if isinstance(belief_over_time, torch.Tensor):
        belief_np = belief_over_time.detach().cpu().numpy()
    else:
        belief_np = np.asarray(belief_over_time)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        belief_np,
        aspect="auto",
        cmap="plasma",
        origin="lower",
        interpolation="none",
    )
    plt.plot(time_frames, true_doa_index, color="red", linestyle="--", linewidth=2, label="True DOA")

    belief_argmax = np.argmax(belief_np, axis=0)
    plt.scatter(time_frames, belief_argmax, color="black", marker="o", s=10, label="Predicted Max")

    plt.xlabel("Time (frames)")
    plt.ylabel("Angle (degrees)")
    plt.title("Belief Over Time")
    plt.colorbar(label="Probability")
    plt.yticks(np.linspace(0, belief_np.shape[0] - 1, len(angles)), angles)
    plt.legend(loc="upper left")

    if debug_level == 2:
        belief_name = name.replace(".jpeg", "_belief.jpeg")
        plt.savefig(belief_name, format="jpeg", dpi=200)
    plt.close()
