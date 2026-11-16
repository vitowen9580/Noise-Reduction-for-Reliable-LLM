
import os
import random
import warnings
import argparse
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(999)
random.seed(999)

noise_class_dictionary = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

############################################

# High
# SNR_MIN = -8.0
# SNR_MAX = -4.0

## Mid for Syntec PDF
# SNR_MIN = -4.0
# SNR_MAX = 0.0
##Mid for Robust PDF
# SNR_MIN = -5.0
# SNR_MAX = -1.0

# #LOW
# SNR_MIN = 4.0
# SNR_MAX = 8.0

############################################

SNR_MIN = None
SNR_MAX = None
SELECTED_NOISE_CLASSES = []

target_folder = "datasets/testB"

############################################

NOISE_VOLUME_BOOST_DB = 0.0

Urban8Kdir = "datasets/UrbanSound8K/audio/"
dest_folder = "datasets/testA"

fold_names = [f"fold{i}" for i in range(1, 11)]


def load_audio_numpy(file_path):
    data, sr = sf.read(file_path, dtype="float32")
    data = np.reshape(data, -1)
    return data, sr


def build_noise_cache(base_dir, folds, classes):
    cache = {c: [] for c in classes}

    for fold in folds:
        fold_dir = os.path.join(base_dir, fold)

        if not os.path.isdir(fold_dir):
            continue

        for fname in os.listdir(fold_dir):
            if fname.endswith(".wav"):
                parts = fname.split("-")

                if len(parts) > 1:
                    try:
                        cls_id = int(parts[1])

                        if cls_id in cache:
                            cache[cls_id].append(
                                os.path.join(fold_dir, fname)
                            )

                    except ValueError:
                        continue

    return cache


def sample_noise_file(noise_type):
    files = NOISE_CACHE.get(noise_type, [])
    return random.choice(files) if files else None


def makeCorruptedFile_mixed_noises(
    filename, dest, noise_classes, snr, boost_db=0.0
):
    true_path = os.path.join(target_folder, filename)

    if not os.path.exists(true_path):
        print(f"\n[ERROR] Source audio file not found: {true_path}")
        return

    try:
        audio_clean = AudioSegment.from_file(true_path)
        clean_np, sr = load_audio_numpy(true_path)

    except Exception as e:
        print(f"\n[ERROR] Failed to read source file ({filename}): {e}")
        return

    mixed_noise_seg = None
    mixed_noise_np_list = []

    for c in noise_classes:
        noise_path = sample_noise_file(c)

        if noise_path is None:
            continue

        try:
            seg = AudioSegment.from_file(noise_path)
            arr, _ = load_audio_numpy(noise_path)

            if mixed_noise_seg is None:
                mixed_noise_seg = seg
            else:
                mixed_noise_seg = mixed_noise_seg.overlay(seg, times=3)

            mixed_noise_np_list.append(arr)

        except Exception:
            continue

    if mixed_noise_seg is None or len(mixed_noise_np_list) == 0:
        print(
            f"\n[WARNING] Could not load selected noises for "
            f"{filename}, skipping."
        )
        return

    clean_watts = np.mean(clean_np**2)
    clean_db = 10 * np.log10(clean_watts + 1e-10)
    target_noise_db = clean_db - snr

    max_len = max(len(a) for a in mixed_noise_np_list)
    combined_np = np.zeros(max_len, dtype=np.float32)

    for a in mixed_noise_np_list:
        combined_np[:len(a)] += a

    combined_watts = np.mean(combined_np**2)
    combined_db = 10 * np.log10(combined_watts + 1e-10)

    db_change = (target_noise_db - combined_db) + boost_db
    scaled_noise_seg = mixed_noise_seg + db_change

    final_audio = audio_clean.overlay(scaled_noise_seg, times=5)

    target_dest = os.path.join(dest, filename)
    final_audio.export(target_dest, format="wav")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate noisy audio with selected SNR and noise types."
    )

    parser.add_argument(
        "--SNR",
        type=float,
        required=True,
        help="Fixed Signal-to-Noise Ratio (e.g., -8, -4, 4)."
    )

    parser.add_argument(
        "-type",
        type=str,
        required=True,
        help="Noise classes separated by commas (e.g., 2,4,7)."
    )

    args = parser.parse_args()

    # Fixed SNR: MIN and MAX use the same value
    SNR_MIN = args.SNR
    SNR_MAX = SNR_MIN+4.0

    # Parse selected noise classes
    try:
        SELECTED_NOISE_CLASSES = [
            int(x.strip()) for x in args.type.split(",")
        ]
    except ValueError:
        parser.error("Invalid noise type. Use comma-separated integers.")

    if not SELECTED_NOISE_CLASSES:
        parser.error("At least one noise type must be selected.")

    for c in SELECTED_NOISE_CLASSES:
        if c not in noise_class_dictionary:
            parser.error(f"Invalid noise type: {c}. Available types: 0-9.")

    # Build noise cache
    NOISE_CACHE = build_noise_cache(
        Urban8Kdir,
        fold_names,
        SELECTED_NOISE_CLASSES
    )

    for cls_id in SELECTED_NOISE_CLASSES:
        count = len(NOISE_CACHE.get(cls_id, []))

        if count == 0:
            raise FileNotFoundError(
                f"[ERROR] No audio files found for class {cls_id} "
                f"({noise_class_dictionary[cls_id]}), "
                "please check the UrbanSound8K path!"
            )

        print(
            f"[CACHE] Class {cls_id} "
            f"({noise_class_dictionary[cls_id]}): "
            f"found {count} audio files"
        )

        os.makedirs(dest_folder, exist_ok=True)

        print("=" * 60)
        print("Noise Generation Configuration")
        print("-" * 60)
        print(f"SNR Range: {SNR_MIN} ~ {SNR_MAX} dB")
        print(f"Noise Type IDs: {SELECTED_NOISE_CLASSES}")
        print(
            "Noise Types: "
            f"{[noise_class_dictionary[c] for c in SELECTED_NOISE_CLASSES]}"
        )
        print(f"Number of Noise Types: {len(SELECTED_NOISE_CLASSES)}")
        print(f"Extra Gain: {NOISE_VOLUME_BOOST_DB} dB")
        print(f"Input Directory: {target_folder}")
        print(f"Output Directory: {dest_folder}")
        print("=" * 60)

    wav_files = sorted([
        f for f in os.listdir(target_folder)
        if f.endswith(".wav")
    ])

    for file in tqdm(
        wav_files,
        desc="Mixed Noise Generation Progress"
    ):

        # Fixed SNR: SNR_MIN = SNR_MAX
        snr_val = random.uniform(SNR_MIN, SNR_MAX)

        makeCorruptedFile_mixed_noises(
            file,
            dest_folder,
            noise_classes=SELECTED_NOISE_CLASSES,
            snr=snr_val,
            boost_db=NOISE_VOLUME_BOOST_DB,
        )

    print(
        f"\n[DONE] All audio files mixed with background noise "
        f"classes {SELECTED_NOISE_CLASSES}!"
    )
