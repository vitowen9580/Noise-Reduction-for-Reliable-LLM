
import argparse
from pathlib import Path
from pydub import AudioSegment


def batch_convert_mp3_to_wav(
    input_dir: str,
    output_dir: str,
    delete_source: bool = False,
):
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)

    if not input_folder.exists():
        print(f"Input folder does not exist: {input_dir}")
        return

    # Create output directory if it does not exist
    output_folder.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp3"
    )

    if not mp3_files:
        print(f"No MP3 files found in {input_dir}.")
        return

    print(f"Input directory : {input_folder}")
    print(f"Output directory: {output_folder}")
    print(f"Found {len(mp3_files)} MP3 files.\n")

    success_count = 0

    for mp3_path in mp3_files:
        wav_path = output_folder / f"{mp3_path.stem}.wav"

        try:
            audio = AudioSegment.from_mp3(str(mp3_path))
            audio.export(str(wav_path), format="wav")

            print(f"✓ Converted: {mp3_path.name} -> {wav_path.name}")
            success_count += 1

            # Delete source MP3 only after successful conversion
            if delete_source:
                mp3_path.unlink()
                print(f"  Deleted source: {mp3_path.name}")

        except Exception as e:
            print(f"✗ Conversion failed {mp3_path.name}: {e}")

    print(
        f"\nProcessing complete! "
        f"Succeeded: {success_count}/{len(mp3_files)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch convert MP3 files to WAV."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="datasets/testB_syntec",
        help="Input directory containing MP3 files."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="datasets/testB",
        help="Output directory for converted WAV files."
    )

    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source MP3 files after successful conversion."
    )

    args = parser.parse_args()

    batch_convert_mp3_to_wav(
        input_dir=args.input,
        output_dir=args.output,
        delete_source=args.delete_source,
    )
