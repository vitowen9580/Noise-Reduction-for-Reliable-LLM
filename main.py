import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from utils.util import *


ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "syntec_pdf": {
        "pdf_path": ROOT / "datasets" / "syntec220plus30qa.pdf",
        "qa_path": ROOT / "datasets" / "syntec_220plus_30qa.json",
        "asr_model": ROOT / "models" / "faster-whisper-large-v3-turbo",
        "is_chinese": True,
    },
    "Robust_pdf": {
        "pdf_path": ROOT / "datasets" / "Robust_tool_wear_Audio_RAG_questions.pdf",
        "qa_path": ROOT / "datasets" / "audio_rag_question_set_recreated.json",
        "asr_model": ROOT / "models" / "faster-whisper-small.en",
        "is_chinese": False,
    },
}

DEFAULT_NOISY_AUDIO_DIR = ROOT / "datasets" / "testA"
DEFAULT_CLEAN_AUDIO_DIR = ROOT / "datasets" / "testB"
DEFAULT_OUT_DIR = ROOT / "outputs" / "four_conditions_comparison"

EMBEDDING_MODEL = ROOT / "models" / "bge-small-zh-v1.5"
RERANKER_MODEL = ROOT / "models" / "bge-reranker-v2-m3"
GEMMA_MODEL = ROOT / "models" / "gemma-4-31B"
JUDGE_MODEL = ROOT / "models" / "Qwen2.5-32B-Instruct"
SEPARATOR_CKPT = ROOT / "models" / "best_separator_on_valid.bin"

SAMPLE_RATE = 16000
TOP_DENSE = 24
TOP_K = 4
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
MAX_NEW_TOKENS = 180
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

CONDITION_NAMES = [
    "A_noisy_audio",
    "B_separated_audio",
    "C_enhanced_audio",
    "D_clean_audio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Audio-RAG Evaluation Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Robust",
        choices=list(DATASET_CONFIGS.keys()),
        help="Select dataset configuration: 'syntec' or 'Robust' (default: Robust)",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all questions")
    parser.add_argument("--clean-dir", type=str, default=str(DEFAULT_CLEAN_AUDIO_DIR))
    parser.add_argument("--noisy-dir", type=str, default=str(DEFAULT_NOISY_AUDIO_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--judge-threshold", type=int, choices=[1, 2], default=2)
    return parser.parse_args()


def validate_paths(paths: List[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file or model not found: {path}")


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    summary: Dict[str, Any] = {"total_questions": total}
    for cond in CONDITION_NAMES:
        correct_cnt = sum(r.get(f"{cond}_is_correct") is True for r in rows)
        scores = [r.get(f"{cond}_judge_score", 0) for r in rows]
        snr_vals = [r.get(f"{cond}_SNR_dB", 0.0) for r in rows]
        sdr_vals = [r.get(f"{cond}_SDR_dB", 0.0) for r in rows]
        summary[cond] = {
            "correct_count": correct_cnt,
            "accuracy": round(correct_cnt / total, 4) if total else 0.0,
            "average_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "average_SNR_dB": round(float(np.mean(snr_vals)), 2) if snr_vals else 0.0,
            "average_SDR_dB": round(float(np.mean(sdr_vals)), 2) if sdr_vals else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    is_chinese = cfg["is_chinese"]
    pdf_path: Path = cfg["pdf_path"]
    qa_path: Path = cfg["qa_path"]
    asr_model: Path = cfg["asr_model"]

    clean_dir = Path(args.clean_dir).expanduser().resolve()
    noisy_dir = Path(args.noisy_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    separated_audio_dir = out_dir / "separated_audio"
    enhanced_audio_dir = out_dir / "enhanced_audio"
    non_speech_audio_dir = out_dir / "non_speech_audio"

    validate_paths([
        pdf_path,
        qa_path,
        asr_model,
        EMBEDDING_MODEL,
        RERANKER_MODEL,
        GEMMA_MODEL,
        JUDGE_MODEL,
        SEPARATOR_CKPT,
    ])

    if is_chinese:
        ASR = ASR_CH(
            asr_model=asr_model,
            DEVICE=DEVICE,
            ASR_COMPUTE_TYPE=ASR_COMPUTE_TYPE,
        )
        Critic = Critic_CH(
            judge_model_path=JUDGE_MODEL,
            device=DEVICE,
            load_in_4bit=True,
        )
        LLM = Gemma_CH(MAX_NEW_TOKENS)
    else:
        ASR = ASR_EN(
            asr_model=asr_model,
            DEVICE=DEVICE,
            ASR_COMPUTE_TYPE=ASR_COMPUTE_TYPE,
        )
        Critic = Critic_EN(
            judge_model_path=JUDGE_MODEL,
            device=DEVICE,
            load_in_4bit=True,
        )
        LLM = Gemma_EN(MAX_NEW_TOKENS)

    rag = RAG(
        EMBEDDING_MODEL=EMBEDDING_MODEL,
        RERANKER_MODEL=RERANKER_MODEL,
        MAX_CHUNK_CHARS=MAX_CHUNK_CHARS,
        CHUNK_OVERLAP=CHUNK_OVERLAP,
        TOP_DENSE=TOP_DENSE,
        TOP_K=TOP_K,
        PDF_PATH=pdf_path,
        DEVICE=DEVICE,
    )

    ###############################
    # Noise Reduction Model
    ################################
    Separator = audio_seperator(
        SEPARATED_AUDIO_DIR=separated_audio_dir,
        NON_SPEECH_AUDIO_DIR=non_speech_audio_dir,
        ENHANCED_AUDIO_DIR=enhanced_audio_dir,
        SEPARATOR_CKPT=SEPARATOR_CKPT,
        DEVICE=DEVICE,
        SAMPLE_RATE=SAMPLE_RATE,
    )

    data_helper = Datasets(SAMPLE_RATE)
    metrics_helper = util()
    tokenizer, gemma = LLM.load_gemma(GEMMA_MODEL)

    questions = data_helper.load_questions(qa_path)
    if args.limit > 0:
        questions = questions[: args.limit]

    for q in questions:
        q["clean_audio"] = data_helper.resolve_audio_file(q, clean_dir)
        q["noisy_audio"] = data_helper.resolve_audio_file(q, noisy_dir)

    missing_clean = [q["qid"] for q in questions if not q.get("clean_audio")]
    missing_noisy = [q["qid"] for q in questions if not q.get("noisy_audio")]
    if missing_clean:
        raise RuntimeError(f"Cannot resolve CLEAN audio for QIDs: {missing_clean}")
    if missing_noisy:
        raise RuntimeError(f"Cannot resolve NOISY audio for QIDs: {missing_noisy}")

    print(
        f"[INIT] dataset={args.dataset} | questions={len(questions)} "
        f"| device={DEVICE} | out_dir={out_dir}"
    )

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['qid']} | {q['question']}")
        print(f"[REFERENCE]: {q['answer']}\n")

        row: Dict[str, Any] = {
            "qid": q["qid"],
            "benchmark_question": q["question"],
            "reference_answer": q["answer"],
            "clean_audio_path": str(q["clean_audio"]),
            "noisy_audio_path": str(q["noisy_audio"]),
        }

        raw_noisy_waveform = data_helper.load_audio_tensor(q["noisy_audio"])
        clean_waveform = data_helper.load_audio_tensor(q["clean_audio"])


        ###############################
        # Noise Reduction Model
        ################################
        sep_speech_waveform, non_speech_waveform = Separator.separate_speech(raw_noisy_waveform)
        sep_path = data_helper.save_audio_file(
            sep_speech_waveform, separated_audio_dir / f"{q['qid']}_separated.wav"
        )
        non_speech_path = data_helper.save_audio_file(
            non_speech_waveform, non_speech_audio_dir / f"{q['qid']}_nonspeech.wav"
        )
        enhanced_waveform = Separator.apply_mafb_enhancement(
            raw_noisy_waveform, sep_speech_waveform
        )
        enh_path = data_helper.save_audio_file(
            enhanced_waveform, enhanced_audio_dir / f"{q['qid']}_enhanced.wav"
        )

        row["separated_audio_path"] = str(sep_path)
        row["non_speech_audio_path"] = str(non_speech_path)
        row["enhanced_audio_path"] = str(enh_path)





        eval_cases = [
            ("A_noisy_audio", raw_noisy_waveform),
            ("B_separated_audio", sep_speech_waveform),
            ("C_enhanced_audio", enhanced_waveform),
            ("D_clean_audio", clean_waveform),
        ]

        for cond_name, waveform in eval_cases:
            prefix = f"{cond_name}_"
            try:
                audio_metrics = metrics_helper.compute_audio_metrics(waveform, clean_waveform)
                row[f"{prefix}SNR_dB"] = audio_metrics["SNR_dB"]
                row[f"{prefix}SDR_dB"] = audio_metrics["SDR_dB"]

                transcript, asr_sec = ASR.transcribe_audio(waveform)
                _, context = rag.retrieve(transcript)

                if is_chinese:
                    prediction, gen_sec = LLM.generate_answer(
                        tokenizer, gemma, transcript, context, is_chinese=True
                    )
                else:
                    prediction, gen_sec = LLM.generate_answer(
                        tokenizer, gemma, transcript, context
                    )

                score, reason = Critic.judge_answer(
                    gold_question=q["question"],
                    reference=q["answer"],
                    prediction=prediction,
                )
                is_correct = bool(score >= args.judge_threshold)

                row[f"{prefix}asr_transcript"] = transcript
                row[f"{prefix}asr_seconds"] = round(asr_sec, 3)
                row[f"{prefix}prediction"] = prediction
                row[f"{prefix}generation_seconds"] = round(gen_sec, 3)
                row[f"{prefix}judge_score"] = score
                row[f"{prefix}judge_reason"] = reason
                row[f"{prefix}is_correct"] = is_correct
                row.update(
                    {
                        f"{prefix}{k}": v
                        for k, v in Critic.lexical_metrics(q["answer"], prediction).items()
                    }
                )

                print(
                    f"[{cond_name}] (Audio Quality: SNR={audio_metrics['SNR_dB']} dB, "
                    f"SDR={audio_metrics['SDR_dB']} dB)"
                )
                print(f"asr={transcript!r}")
                print(f"score={score}; is_correct={is_correct};")
                print(f"prediction={prediction!r}\n")

            except Exception as exc:
                row[f"{prefix}error"] = repr(exc)
                row[f"{prefix}judge_score"] = 0
                row[f"{prefix}is_correct"] = False
                print(f"[{cond_name}]")
                print("score=0; is_correct=False;")
                print(f"prediction='ERROR: {repr(exc)}'\n")

        rows.append(row)
        data_helper.save_incremental_json_and_csv(rows, CONDITION_NAMES, out_dir)

    summary = build_summary(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[DONE] Evaluation completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()