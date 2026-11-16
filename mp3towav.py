import argparse
from pathlib import Path
from pydub import AudioSegment


def batch_convert_mp3_to_wav(
    folder_path: str = "testB", delete_source: bool = False
):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return

    mp3_files = sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".mp3"
        ]
    )

    if not mp3_files:
        print(f"No MP3 files found in {folder_path}.")
        return

    print(f"Found {len(mp3_files)} MP3 files, starting conversion...\n")

    success_count = 0
    for mp3_path in mp3_files:
        wav_path = mp3_path.with_suffix(".wav")
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            print(f"✓ Converted: {mp3_path.name} -> {wav_path.name}")
            success_count += 1

            if delete_source:
                mp3_path.unlink()

        except Exception as e:
            print(f"✗ Conversion failed {mp3_path.name}: {e}")

    print(f"\nProcessing complete! Succeeded: {success_count}/{len(mp3_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert MP3 files to WAV.")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="datasets/testB_syntec",
        help="Target directory containing MP3 files (default: datasets/testB_syntec)",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source MP3 files after conversion",
    )
    args = parser.parse_args()

    batch_convert_mp3_to_wav(
        folder_path=args.folder_path,
        delete_source=args.delete_source,
    )

# python mp3towav.py --folder_path mp3 --delete-source