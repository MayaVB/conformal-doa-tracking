import os
import sys
import logging

from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io.wavfile import write, read
from silero_vad import load_silero_vad, get_speech_timestamps

from utils.preprocess import filter_VAD_in_stft_domain, EVG_RTF_estimation
from utils.visualize import plot_and_save_irf, plot_and_save
from utils.vad import compute_time_freq_spp

eps = sys.float_info.epsilon


def frame_raw_DOA(y_train: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Segment a time-level DOA trajectory into frame-level DOAs by averaging.

    Args:
        y_train:
            DOA values per time tap, shape [T].
        num_frames:
            Number of target frames to compute.

    Returns:
        DOA values per frame, shape [num_frames].
    """
    if y_train.size == 1:
        return y_train

    taps_per_frame = len(y_train) // num_frames
    y_train_frames = []

    for f in range(num_frames):
        start = f * taps_per_frame
        end = (f + 1) * taps_per_frame if f < num_frames - 1 else len(y_train)
        frame_doa = y_train[start:end]
        mean_doa = np.mean(frame_doa)
        y_train_frames.append(mean_doa)

    return np.array(y_train_frames)


def min_max_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalize along the last axis.

    Args:
        x:
            Input array, e.g. (batch_size, channels, height, width).
        eps:
            Small constant to avoid division by zero.

    Returns:
        Normalized array of the same shape as x.
    """
    x_min = np.min(x, axis=-1, keepdims=True)
    x_max = np.max(x, axis=-1, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


class CostumDataset(Dataset):
    """
    Dataset for DOA estimation with RTF/iRTF features.

    Supports:
      - Direct HDF5-based loading (`data_path` is .h5).
      - (Partially implemented) directory-based loading for raw .wav scenes.
      - Optional precomputed feature loading from `target_file` when `load_rtf=True`.
    """

    def __init__(
        self,
        data_path: str,
        spec_size: int,
        spec_fixed_var: float,
        frame_size: int = 512,
        overlap: float = 0.75,
        nfft_size: int = 1024,
        ref_channel: int = 2,
        window: str = "hann",
        rtf_esti_op: str = "iRTF",
        use_context_frame: bool = False,
        num_context_frame: int = 10,
        feature_op: str = "ReImWithoutSpec",
        num_classes: int = int(180 / 5),
        transform=None,
        load_rtf: bool = False,
        target_file: str = "data/rtf_data.h5",
        debug_level: int = 1,
        use_center_crop: bool = True,
        group_name: Optional[str] = None,
    ):
        # General configuration
        self.data_path = data_path
        self.spec_size = spec_size
        self.spec_fixed_var = spec_fixed_var

        # Spectrogram / STFT configuration
        self.frame_size = frame_size
        self.overlap = overlap
        self.nfft_size = nfft_size
        self.ref_channel = ref_channel
        self.window = window
        self.rtf_esti_op = rtf_esti_op
        self.use_context_frame = use_context_frame
        self.num_context_frame = num_context_frame

        # RTF / feature options
        self.feature_op = feature_op
        self.num_classes = num_classes
        self.transform = transform

        # Feature saving / loading
        self.load_rtf = load_rtf
        self.target_file = target_file

        # Debug and cropping
        self.debug_level = debug_level
        self.use_center_crop = use_center_crop

        # For HDF5 files with multiple groups, allow focusing on a single group
        self.group_name = group_name

        # Determine dataset length
        if not self.data_path.endswith(".h5"):
            # Directory-based loading (scene_*/src_pos*/_chX.wav)
            self.data_len = 0
            for scene_folder in os.listdir(self.data_path):
                scene_path = os.path.join(self.data_path, scene_folder)
                if not os.path.isdir(scene_path):
                    continue

                # Assume 1 speaker per scene
                speaker_dirs = [
                    d for d in os.listdir(scene_path)
                    if os.path.isdir(os.path.join(scene_path, d))
                ]
                if not speaker_dirs:
                    continue

                speaker_path = os.path.join(scene_path, speaker_dirs[0])
                for src_pos_folder in os.listdir(speaker_path):
                    src_pos_path = os.path.join(speaker_path, src_pos_folder)
                    if os.path.isdir(src_pos_path):
                        wav_files = [
                            f for f in os.listdir(src_pos_path)
                            if f.endswith(".wav")
                        ]
                        if wav_files:
                            # For now we count per source position
                            self.data_len += 1
        else:
            # HDF5-based loading
            with h5py.File(data_path, "r", swmr=True) as h5f:
                if self.group_name:
                    # Single scenario if group exists
                    self.data_len = 1 if self.group_name in h5f else 0
                else:
                    self.data_len = h5f.attrs["data_len"]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one sample (features + labels + metadata).
        """
        if self.load_rtf:
            # Load precomputed features from target_file
            with h5py.File(self.target_file, "r", swmr=True) as h5f:
                group_name = list(h5f.keys())[idx]
                grp = h5f[group_name]

                x_train_processed = grp["x_train_processed"][:]       # [C, F, T]
                y_train_processed = grp["y_train_processed"][:]       # [Classes, F, T]
                mic_pos = grp["mic_positions"][:]                     # [M, 3] or similar
                x_train_wav = grp.get("data_raw", np.array([]))       # Optional raw data
                vad_masks = grp.get("vad_masks", np.ones_like(y_train_processed[0], dtype=bool))

                # If raw DOA trajectory exists, use it; otherwise just store class targets
                if "speaker_DOA_trj" in grp:
                    y_train = grp["speaker_DOA_trj"][:]
                else:
                    y_train = grp["speaker_DOA_"][()]
        elif not self.data_path.endswith(".h5"):
            # Directory-based loading case
            # NOTE: This branch is only partially implemented since `y_train`
            # is not stored with the wavs in this setup.
            # Keep behavior but make the limitation explicit.
            scene_idx = idx // (32 * 5)
            sub_idx = idx % (32 * 5)
            source_idx = sub_idx // 5
            _wav_idx = sub_idx % 5

            scene_folder = os.path.join(self.data_path, f"scene_{scene_idx + 1}")
            speaker_folder = os.path.join(
                scene_folder, os.listdir(scene_folder)[0]
            )  # Only one speaker per scene
            wav_dir = os.path.join(speaker_folder, f"src_pos{source_idx + 1}")

            # Collect all channel wavs
            num_channels = len(
                [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
            )
            wav_files = [
                os.path.join(wav_dir, f"_ch{ch_idx}.wav")
                for ch_idx in range(1, num_channels + 1)
            ]

            wav_list = []
            for wav_file in wav_files:
                sr, x_train_wav_ch = read(wav_file)
                x_train_wav_ch = x_train_wav_ch.astype(float) / 32768.0
                wav_list.append(x_train_wav_ch)

            wav_list = np.stack(wav_list, axis=0)  # [M, T]

            # TODO: y_train should be provided from dataset creation.
            y_train = np.array([0.0])  # placeholder
            x_train_processed, vad_masks, x_train_wav = self.process_data(
                wav_list, y_train
            )
            y_train_processed = self.process_target(x_train_processed, y_train)
            mic_pos = np.zeros((wav_list.shape[0], 3), dtype=float)  # placeholder
        else:
            # Default HDF5-based processing
            with h5py.File(self.data_path, "r", swmr=True) as h5f:
                if self.group_name:
                    grp = h5f[self.group_name]
                else:
                    group_name = list(h5f.keys())[idx]
                    grp = h5f[group_name]

                x_train = grp["signals_"][:]             # [M, T]
                y_train = grp["speaker_DOA_"][()]        # scalar or trajectory
                mic_pos = grp["mic_positions"][:]

                if "speaker_DOA_trj" in grp:
                    y_train = grp["speaker_DOA_trj"][:]

                if self.debug_level == 3:
                    print(grp)

                x_train_processed, vad_masks, x_train_wav = self.process_data(
                    x_train, y_train
                )
                y_train_processed = self.process_target(x_train_processed, y_train)
                y_train = frame_raw_DOA(y_train, x_train_processed.shape[2])

        # Convert to torch tensors
        x_train_wav = torch.tensor(x_train_wav, dtype=torch.float32)
        x_train_processed = torch.tensor(x_train_processed, dtype=torch.float32)
        vad_masks = torch.tensor(vad_masks, dtype=torch.bool)
        y_train_processed = torch.tensor(y_train_processed, dtype=torch.float32)

        sample = {
            "input": x_train_processed,
            "spp_masks": vad_masks,
            "target": y_train_processed,
            "raw_input": x_train_wav,
            "mic_pos": mic_pos,
            "raw_doa": y_train,
        }

        if self.transform:
            sample["input"] = self.transform(sample["input"])
            sample["target"] = self.transform(sample["target"])

        return sample

    # ------------------------------------------------------------------ #
    #                    FEATURE / TARGET PROCESSING                     #
    # ------------------------------------------------------------------ #

    def process_data(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process raw multichannel waveforms into network features.

        Steps:
          - STFT per microphone (Torch STFT).
          - Silero VAD per microphone; combine to a global VAD.
          - Filter frames by VAD in STFT domain.
          - Center crop in time if requested.
          - Estimate iRTF / context-RTF / EVD-based RTF depending on `rtf_esti_op`.
          - Build final feature tensor according to `feature_op`.

        Args:
            x_train:
                Raw waveforms, shape [n_mics, n_samples].
            y_train:
                DOA (scalar) or trajectory; not used here directly, but kept for API
                symmetry with `process_target`.

        Returns:
            x_train_final:
                Feature tensor, shape [C, F, T].
            vad_masks:
                Time-frequency speech masks, shape [F, T, M] (for iRTF/EVD variants).
            x_train_wav:
                VAD-filtered raw waveforms, shape [n_mics, T_speech].
        """
        x_train = x_train + eps
        ref_channel = self.ref_channel
        fs = 16000  # WSJ0 param

        signals_stft = []
        vad_binary = []
        model = load_silero_vad()

        # Select onesided spectra option
        if self.rtf_esti_op == "iRTF":
            use_onesided_spectra = False
        elif self.rtf_esti_op == "EVD":
            use_onesided_spectra = True
        else:
            use_onesided_spectra = False

        # Per-mic STFT + Silero VAD
        for mic in range(x_train.shape[0]):
            x_train_tensor = torch.tensor(x_train[mic, :], dtype=torch.float32)

            Zxx = torch.stft(
                x_train_tensor,
                n_fft=self.nfft_size,
                hop_length=int(self.frame_size * (1 - self.overlap)),
                win_length=self.frame_size,
                window=torch.hann_window(self.frame_size),
                return_complex=True,
                onesided=use_onesided_spectra,
            )

            signals_stft.append(Zxx)

            speech_timestamps = get_speech_timestamps(x_train[mic, :], model)
            vad_binary_mic = np.zeros(len(x_train[mic, :]), dtype=bool)

            for timestamp in speech_timestamps:
                start = int(timestamp["start"])
                end = int(timestamp["end"])
                vad_binary_mic[start:end] = True

            vad_binary.append(vad_binary_mic)

            if self.debug_level == 2:
                os.makedirs("dataset_verification", exist_ok=True)
                write(
                    f"dataset_verification/wav_mic_{mic}.wav",
                    fs,
                    x_train[mic, :],
                )

        # Stack STFT results: [freq, frames, mics]
        signals_stft = torch.stack(signals_stft, dim=-1).cpu().numpy()
        vad_binary = np.stack(vad_binary, axis=-1)

        # Global VAD: True only if all mics detect speech
        vad_binary_all = np.all(vad_binary, axis=1).astype(bool)

        # Time-domain VAD filter + truncation for raw waveforms
        x_train_wav = x_train[:, vad_binary_all]
        desired_time_size4raw = 60000
        x_train_wav = x_train_wav[:, 0:desired_time_size4raw]

        # Frame-level VAD in STFT domain
        signals_stft_filtered, vad_binary_framed = filter_VAD_in_stft_domain(
            signals_stft, vad_binary_all, self.frame_size, self.overlap, fs
        )

        if self.debug_level == 2:
            plot_and_save(
                signals_stft,
                (signals_stft * vad_binary_framed[np.newaxis, :, np.newaxis]).astype(bool),
                output_prefix="stft_vad_plot",
            )
            plot_and_save(
                signals_stft_filtered,
                np.ones(signals_stft_filtered.shape, dtype=bool),
                output_prefix="stft_vad_filtered_plot",
            )

        signals_stft = signals_stft_filtered

        # Center crop in time
        desired_time_size = 400
        k, n, p = signals_stft.shape

        if self.use_center_crop:
            center_index = n // 2
            half_k = desired_time_size // 2
            start_index = max(0, center_index - half_k)
            end_index = min(n, center_index + half_k)

            current_size = end_index - start_index
            if current_size < desired_time_size:
                print(
                    f"current time size < desired time: "
                    f"{current_size} < {desired_time_size}"
                )
                if start_index == 0:
                    end_index = start_index + desired_time_size
                elif end_index == n:
                    start_index = end_index - desired_time_size
                else:
                    start_index = max(0, center_index - half_k)
                    end_index = start_index + desired_time_size

                if end_index - start_index < desired_time_size:
                    logging.error(
                        "Concatenated signals do not match in the time domain - "
                        "check sizes!"
                    )
        else:
            start_index = 0
            end_index = n - 1

        signals_stft = signals_stft[:, start_index:end_index, :]

        # ------------------------------------------------------------------ #
        #                  RTF / iRTF ESTIMATION + SPP MASK                  #
        # ------------------------------------------------------------------ #
        if self.rtf_esti_op == "iRTF":
            # Remove high freqs
            signals_stft = signals_stft[: self.spec_size, :, :]

            vad_masks = compute_time_freq_spp(
                signals_stft, threshold=0.5, low_freq_emphasis=False
            )

            if self.debug_level == 2:
                mic_num = 0

                os.makedirs("dataset_verification", exist_ok=True)

                # STFT plot
                plt.figure(figsize=(10, 6))
                plt.imshow(
                    np.log1p(np.abs(signals_stft[:, :, mic_num])),
                    aspect="auto",
                    origin="lower",
                    cmap="inferno",
                    interpolation="none",
                )
                plt.title("STFT")
                plt.xlabel("Time (frames)")
                plt.ylabel("Frequency (bins)")
                plt.colorbar(label="Magnitude")
                plt.savefig(
                    f"dataset_verification/stft_before_irtf_mic{mic_num}.jpeg",
                    format="jpeg",
                    dpi=300,
                )
                plt.close()

                # SPP mask
                plt.figure(figsize=(10, 6))
                plt.imshow(
                    vad_masks[:, :, mic_num],
                    aspect="auto",
                    origin="lower",
                    cmap="gray",
                    interpolation="none",
                )
                plt.title("SPP Mask")
                plt.xlabel("Time (frames)")
                plt.ylabel("Frequency (bins)")
                plt.colorbar(label="Speech Presence")
                plt.savefig(
                    f"dataset_verification/vad_mask_mic{mic_num}.jpeg",
                    format="jpeg",
                    dpi=300,
                )
                plt.close()

                # STFT + mask overlay
                plt.imshow(
                    np.log1p(np.abs(signals_stft[:, :, mic_num])),
                    aspect="auto",
                    origin="lower",
                    cmap="inferno",
                    interpolation="none",
                )
                plt.imshow(
                    vad_masks[:, :, mic_num],
                    aspect="auto",
                    origin="lower",
                    cmap="gray",
                    alpha=0.5,
                    interpolation="none",
                )
                plt.title("STFT Magnitude with VAD Mask Overlay")
                plt.xlabel("Time (frames)")
                plt.ylabel("Frequency (bins)")
                plt.colorbar(label="Magnitude")
                plt.tight_layout()
                plt.savefig(
                    f"dataset_verification/stft_and_vad_mask_mic{mic_num}.jpeg",
                    format="jpeg",
                    dpi=300,
                )
                plt.close()

            irtf = signals_stft / (
                signals_stft[:, :, self.ref_channel][:, :, np.newaxis] + eps
            )
            irtf = np.delete(irtf, self.ref_channel, axis=2)

        elif self.use_context_frame:
            signals_stft = signals_stft[: self.spec_size, :, :]

            vad_masks = compute_time_freq_spp(
                signals_stft, threshold=0.5, low_freq_emphasis=False
            )

            nc = self.num_context_frame
            irtf = np.zeros(
                (signals_stft.shape[0], signals_stft.shape[1], signals_stft.shape[2] - 1),
                dtype=complex,
            )

            for mic in range(signals_stft.shape[2]):
                if mic == ref_channel:
                    continue

                curr_irtf = np.zeros_like(
                    signals_stft[:, :, ref_channel], dtype=complex
                )

                for l in range(nc, signals_stft.shape[0] - nc):
                    for kk in range(signals_stft.shape[1]):
                        numerator = np.sum(
                            signals_stft[l - nc : l + nc + 1, kk, mic]
                            * np.conj(
                                signals_stft[l - nc : l + nc + 1, kk, ref_channel]
                            )
                        )
                        denominator = (
                            np.sum(
                                signals_stft[l - nc : l + nc + 1, kk, ref_channel]
                                * np.conj(
                                    signals_stft[l - nc : l + nc + 1, kk, ref_channel]
                                )
                            )
                            + eps
                        )
                        curr_irtf[l, kk] = numerator / denominator

                irtf[:, :, mic if mic < ref_channel else mic - 1] = curr_irtf

        elif self.rtf_esti_op == "EVD":
            signals_stft = signals_stft[: self.spec_size, :, :]

            vad_masks = compute_time_freq_spp(
                signals_stft, threshold=0.5, low_freq_emphasis=False
            )

            irtf = EVG_RTF_estimation(
                signals_stft, fs, self.ref_channel, self.overlap, self.frame_size
            )

            if self.debug_level == 2:
                plot_and_save_irf(
                    irtf, np.ones(irtf.shape, dtype=bool), output_prefix="EVD_rtf_vad_plot"
                )

            irtf = irtf[: self.spec_size, :, :]

        else:
            raise ValueError(f"Unknown rtf_esti_op: {self.rtf_esti_op}")

        # ------------------------------------------------------------------ #
        #                        FEATURE CONSTRUCTION                        #
        # ------------------------------------------------------------------ #

        # Normalize per-microphone
        irtf_std = irtf.std(axis=(0, 1))
        irtf_std = irtf_std[np.newaxis, np.newaxis, :]
        irtf = irtf / (irtf_std + 1e-10)

        if self.debug_level == 2 and signals_stft.shape[0] > self.spec_size:
            os.makedirs("dataset_verification", exist_ok=True)
            frame_selected = 200
            mic1 = 2
            plt.figure()
            plt.plot(np.fft.ifftshift(np.fft.ifft(irtf[:, frame_selected, mic1])))
            plt.savefig(
                f"dataset_verification/instantaneous_rtf_estimation_frame_selected_"
                f"{frame_selected}_mic_selected{mic1}.jpeg",
                format="jpeg",
                dpi=300,
            )
            plt.close()

            frame_selected = 100
            mic1 = 1
            plt.figure()
            plt.plot(np.fft.ifftshift(np.fft.ifft(irtf[:, frame_selected, mic1])))
            plt.savefig(
                f"dataset_verification/instantaneous_rtf_estimation_frame_selected_"
                f"{frame_selected}_mic_selected{mic1}.jpeg",
                format="jpeg",
                dpi=300,
            )
            plt.close()

        # Build feature vector
        if self.feature_op == "SinCosWithSpec":
            x_angle = np.angle(irtf)
            x_cos = np.cos(x_angle)
            x_sin = np.sin(x_angle)

            x_spec = np.log(eps + np.abs(signals_stft[: self.spec_size, :, ref_channel]))
            x_spec = ((x_spec - x_spec.min()) / (x_spec.max() - x_spec.min() + eps)) * 2 - 1
            x_spec = (
                x_spec
                * (np.sqrt(self.spec_fixed_var) / (eps + x_spec.std()))
            )
            x_spec = np.expand_dims(x_spec, 2)

            x_train_final = np.concatenate((x_cos, x_sin, x_spec), axis=-1)

        elif self.feature_op == "SinCosWithoutSpec":
            x_angle = np.angle(irtf[: self.spec_size, :, :])
            x_cos = np.cos(x_angle)
            x_sin = np.sin(x_angle)
            x_train_final = np.concatenate((x_cos, x_sin), axis=-1)

        elif self.feature_op == "ReImWithSpec":
            x_re = np.real(irtf)
            x_im = np.imag(irtf)

            x_spec = np.log(eps + np.abs(signals_stft[: self.spec_size, :, ref_channel]))
            x_spec = ((x_spec - x_spec.min()) / (x_spec.max() - x_spec.min() + eps)) * 2 - 1
            x_spec = (
                x_spec
                * (np.sqrt(self.spec_fixed_var) / (eps + x_spec.std()))
            )
            x_spec = np.expand_dims(x_spec, 2)

            x_train_final = np.concatenate((x_re, x_im, x_spec), axis=-1)

        elif self.feature_op == "ReImWithoutSpec":
            x_re = np.real(irtf)
            x_im = np.imag(irtf)
            x_train_final = np.concatenate((x_re, x_im), axis=-1)

        else:
            raise ValueError(f"Unknown feature_op: {self.feature_op}")

        # Final layout: [C, F, T]
        x_train_final = x_train_final.transpose(2, 0, 1)

        return x_train_final, vad_masks, x_train_wav

    def process_target(
        self, x_train_final: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """
        Convert continuous DOA(s) into per-frame, per-frequency one-hot class targets.

        For a DOA trajectory:
          - Segment into frame-level DOAs.
          - Quantize each frame DOA to a class.
          - One-hot encode and repeat over frequency bins.

        For a single DOA:
          - Quantize to a class.
          - One-hot encode and tile over [freq, frames].

        Args:
            x_train_final:
                Feature tensor of shape [C, F, T].
            y_train:
                DOA scalar or trajectory.

        Returns:
            Target tensor of shape [num_classes, F, T].
        """
        num_frames = x_train_final.shape[2]
        spec_size = x_train_final.shape[1]
        num_classes = self.num_classes

        class_resolution = int(180 / num_classes)

        if y_train.size > 1:
            taps_per_frame = len(y_train) // num_frames
            y_class_frames = []

            for f in range(num_frames):
                start = f * taps_per_frame
                end = (f + 1) * taps_per_frame if f < num_frames - 1 else len(y_train)

                frame_doa = y_train[start:end]
                mean_doa = np.mean(frame_doa)

                class_idx = int(np.round((mean_doa - 10) / class_resolution))
                class_idx = np.clip(class_idx, 0, num_classes - 1)
                y_class_frames.append(class_idx)

            y_class_frames = np.array(y_class_frames)  # [frames]

            one_hot_targets = np.eye(num_classes, dtype="uint8")[y_class_frames]  # [frames, classes]
            one_hot_targets = one_hot_targets.T  # [classes, frames]

            # [classes, freqs, frames]
            target = np.repeat(
                one_hot_targets[:, np.newaxis, :],
                spec_size,
                axis=1,
            )

        else:
            y_class = int((np.round(y_train) - 10) // class_resolution)
            y_class = int(np.clip(y_class, 0, num_classes - 1))

            one_hot_target = np.eye(num_classes, dtype="uint8")[y_class]
            target = np.tile(
                one_hot_target, (spec_size, num_frames, 1)
            )  # [spec_size, frames, num_classes]
            target = np.transpose(target, (2, 0, 1))  # [classes, freqs, frames]

        return target

    def process_and_save_features(self) -> None:
        """
        Offline preprocessing: read raw HDF5 data, compute features and targets,
        and save them into a new HDF5 file (`target_file`).
        """
        with h5py.File(self.target_file, "a") as h5f_write:
            with h5py.File(self.data_path, "r", swmr=True) as h5f:
                group_names = list(h5f.keys())
                for key_ind, group_name in tqdm(
                    enumerate(group_names),
                    total=len(group_names),
                    desc="Processing Groups",
                ):
                    grp_read = h5f[group_name]

                    x_train = grp_read["signals_"][:]
                    y_train = grp_read["speaker_DOA_"][()]
                    mic_pos = grp_read["mic_positions"][:]

                    if self.debug_level == 3:
                        print(f"running file: {key_ind} from total of {len(group_names)}")

                    x_train_processed, vad_masks, data_raw = self.process_data(
                        x_train, y_train
                    )
                    y_train_processed = self.process_target(x_train_processed, y_train)

                    grp_write = h5f_write.create_group(group_name)
                    grp_write.create_dataset("x_train_processed", data=x_train_processed)
                    grp_write.create_dataset("y_train_processed", data=y_train_processed)
                    grp_write.create_dataset("speaker_DOA_", data=y_train)
                    grp_write.create_dataset("mic_positions", data=mic_pos)
                    grp_write.create_dataset("vad_masks", data=vad_masks)
                    grp_write.create_dataset("data_raw", data=data_raw)

        print(f"Finished calculating features. Saved to {self.target_file}")
