from collections import Counter
import csv
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import librosa
from faster_whisper import WhisperModel
import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.resunet_stft import SealGPT_Sep


def norm(value):
    value = str(value or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", value).strip()



class Critic_CH:
    SPECIAL_STOP_TOKENS = ("<end_of_turn>", "<eos>", "<|im_end|>", "<|eot_id|>")
    INVALID_PREDICTIONS = ("answer", "output", "empty", "none", "回答", "無")
    NEG_MARKERS = (
        "incorrect",
        "contradict",
        "fails to",
        "does not capture",
        "off-topic",
        "hallucinat",
        "錯誤",
        "矛盾",
        "不相符",
        "答非所問",
        "未能正確",
        "幻覺",
        "回答不正確",
        "遺漏核心",
        "完全不同",
    )
    POS_MARKERS = (
        "accurately captures",
        "fully correct",
        "matches the reference",
        "correctly identifies",
        "完全正確",
        "正確回答",
        "符合標準答案",
        "完全符合",
        "核心要素均正確",
        "答對",
        "完整涵蓋",
        "精確給出",
    )
    PART_MARKERS = (
        "partially correct",
        "captures some factual details",
        "misses key",
        "部分正確",
        "遺漏部分",
        "僅提到",
        "缺少條件",
        "不夠完整",
        "部分相符",
    )

    def __init__(
        self,
        judge_model_path: Path,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ) -> None:
        judge_path = Path(judge_model_path)
        if not judge_path.exists():
            raise FileNotFoundError(f"找不到指定的評審模型路徑: {judge_path}")

        print(f"[載入] 評審模型: {judge_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(judge_path), trust_remote_code=True
        )

        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(judge_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(judge_path),
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            ).eval()

        self.device = self.model.get_input_embeddings().weight.device

        self.stop_ids: List[int] = []
        if isinstance(self.tokenizer.eos_token_id, list):
            self.stop_ids.extend(self.tokenizer.eos_token_id)
        elif self.tokenizer.eos_token_id is not None:
            self.stop_ids.append(self.tokenizer.eos_token_id)

        for token in self.SPECIAL_STOP_TOKENS:
            try:
                val = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(val, int) and val not in self.stop_ids and val >= 0:
                    self.stop_ids.append(val)
            except Exception:
                pass

        self.pad_token_id = self.tokenizer.eos_token_id or self.stop_ids[0]

    def judge_answer(
        self, gold_question: str, reference: str, prediction: str
    ) -> Tuple[int, str]:
        clean_pred = str(prediction or "").strip().lower()
        if not clean_pred or clean_pred in self.INVALID_PREDICTIONS:
            return 0, "模型輸出為空或無效回答"

        prompt = f"""你是一名著重於「實際語意理解」的客觀評審專家。請比對【模型回答】與【標準答案】的事實核心是否一致。

        【基準題目``】: {gold_question}
        【標準答案】: {reference}
        【模型回答】: {prediction}

        【評判核心原則】:
        1. 意思相同即算正確（寬鬆語意對齊）：
        - 只要【模型回答】所傳達的實質意思、核心數值或操作邏輯與【標準答案】基本一致，且無嚴重事實矛盾，即應判定為「得分 2」。
        - 允許同義詞置換、不同表述方式（如具體稱呼 vs 通用稱呼）、簡寫、或者直接引述技術手冊原文。
        2. 不苛求非提問重點的附帶細節：
        - 只要模型已正面答出題目所問之核心重點，未提及標準答案中額外補充的次要說明（如：警告彈窗、次要操作前提、歷史舊版本差異等），「不可」因此扣分。
        3. 忽略無關贅字：
        - 自動忽略回答開頭的「答案是：」或格式性標籤，僅審視回答之實質語意。

        【評分標準】:
        - 得分 2 (完全正確/意思相同): 核心意思、主要數值、公式或指令正確，實質語意與標準答案相符或等價，未造成實際操作矛盾。
        - 得分 1 (部分正確): 題目有兩個以上核心提問，模型僅答出其中一個；或數值與核心邏輯有明顯錯誤但提及部分概念。
        - 得分 0 (完全錯誤): 核心意思完全不符、互相矛盾、答非所問或胡言亂語。

        請嚴格輸出合法 JSON 格式，不要包含 Markdown 程式碼區塊標記：
        {{"score": 2, "reason": "簡短評分理由"}}
        """
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {
            k: v.to(self.device)
            for k, v in dict(encoded).items()
            if torch.is_tensor(v)
        }
        input_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=self.stop_ids,
                pad_token_id=self.pad_token_id,
                use_cache=True,
            )
        raw_response = self.tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True
        ).strip()

        clean_response = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.I)
        clean_response = re.sub(r"\s*```$", "", clean_response).strip()

        json_blocks = re.findall(r"\{[\s\S]*?\}", clean_response)
        for block in reversed(json_blocks):
            try:
                parsed = json.loads(block)
                if "score" in parsed and str(parsed["score"]).strip() in ("0", "1", "2"):
                    return int(parsed["score"]), str(parsed.get("reason", "")).strip()
            except Exception:
                continue

        match = re.search(
            r'["\']?score["\']?\s*[:=]\s*["\']?([012])["\']?', clean_response, re.I
        )
        if match:
            score = int(match.group(1))
            reason_match = re.search(
                r'["\']?reason["\']?\s*[:=]\s*["\']?(.*?)(?:["\'\}\n]|$)',
                clean_response[match.end() :],
                re.I,
            )
            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "從評分標籤中解析得出"
            )
            return score, reason

        resp_lower = clean_response.lower()
        has_negative = any(k in resp_lower for k in self.NEG_MARKERS)
        has_positive = any(k in resp_lower for k in self.POS_MARKERS)
        has_partial = any(k in resp_lower for k in self.PART_MARKERS)

        if has_positive and not has_negative:
            return 2, f"評審確認完全正確: {clean_response[:100]}"
        if has_partial or (has_positive and has_negative):
            return 1, f"評審確認部分正確: {clean_response[:100]}"
        if has_negative and not has_positive:
            return 0, f"評審判定錯誤: {clean_response[:100]}"

        return 0, f"評審輸出格式解析失敗: {clean_response[:100]}"

    def lexical_metrics(
        self, reference: str, prediction: str, is_chinese: bool = True
    ) -> Dict[str, float]:
        ref_text = (reference or "").lower().strip()
        pred_text = (prediction or "").lower().strip()

        if is_chinese:
            token_pattern = re.compile(
                r"[\u4e00-\u9fa5]|[a-zA-Z0-9_]+", flags=re.UNICODE
            )
            ref = token_pattern.findall(ref_text)
            pred = token_pattern.findall(pred_text)
        else:
            ref = re.findall(r"\w+", ref_text, flags=re.UNICODE)
            pred = re.findall(r"\w+", pred_text, flags=re.UNICODE)

        if not ref or not pred:
            return {"Token_F1": 0.0, "Precision": 0.0, "Recall": 0.0, "EM": 0.0}

        common = sum((Counter(ref) & Counter(pred)).values())
        precision = common / len(pred)
        recall = common / len(ref)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        return {
            "Token_F1": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "EM": float(ref == pred),
        }

class ASR_CH:
    INITIAL_PROMPT: str = (
        "繁體中文，數控銑床、G碼、MDI、新代控制器、加工參數、切削力。"
    )

    def __init__(
        self,
        asr_model: str,
        DEVICE: str,
        ASR_COMPUTE_TYPE: str,
        language: Optional[str] = "zh",
    ) -> None:
        self.DEVICE = DEVICE
        self.language = language
        self.initial_prompt = self.INITIAL_PROMPT
        self.asr = self.load_asr(asr_model, ASR_COMPUTE_TYPE)

    def load_asr(self, model_path: str, ASR_COMPUTE_TYPE: str) -> WhisperModel:
        print(f"[LOAD] Whisper: {model_path}")
        return WhisperModel(
            str(model_path), device=self.DEVICE, compute_type=ASR_COMPUTE_TYPE
        )

    def transcribe_audio(
        self, audio_input: Any, language: Optional[str] = None
    ) -> Tuple[str, float]:
        start = time.time()
        lang = language if language is not None else self.language

        if isinstance(audio_input, torch.Tensor):
            audio = (
                audio_input.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
        else:
            audio = str(audio_input)

        segments, _ = self.asr.transcribe(
            audio,
            beam_size=3,
            language=lang,
            initial_prompt=self.initial_prompt,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        seg_texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        transcript = "".join(seg_texts) if lang == "zh" else " ".join(seg_texts)
        transcript = norm(transcript)
        return transcript, time.time() - start

class Gemma_CH:
    def __init__(self, MAX_NEW_TOKENS):
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        pass

    def load_gemma(self, model_path):
        print(f"[LOAD] Gemma: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
                "<start_of_turn>model\n"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}<end_of_turn>\n"
                "{% endif %}{% endfor %}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        return tokenizer, model

    def clean_model_output(self, raw_answer: str) -> str:
        if not raw_answer:
            return ""

        if "<start_of_turn>model" in raw_answer:
            raw_answer = raw_answer.split("<start_of_turn>model")[-1]
        if "<start_of_turn>user" in raw_answer:
            raw_answer = raw_answer.split("<start_of_turn>user")[0]

        stop_patterns = [
            "</think>", "</thought>", "</answer>", "</model>",
            "</user>", "</s>", "<eos>", "<end_of_turn>", "</code>"
        ]
        for pattern in stop_patterns:
            if pattern in raw_answer:
                raw_answer = raw_answer.split(pattern)[0]

        leak_markers = [
            "<unused", "geqquser", "user\nYou are", "Question:", "PDF evidence:",
            "\n\nHuman:", "\n\nAssistant:", "]]></thought>", "<output>", "</output>",
            "]]>", "papers/", "# [A Survey", "## Write a very high-quality",
            "Let me think step-by-step:", "Let's analyze the relevant",
            "From the second piece of evidence:", "From the evidence:",
            "Based on the provided PDF evidence:", "根據提供的 PDF 證據：", "根據提供的PDF證據，",
            "根據手冊內容：", "分析如下：", "答案如下："
        ]
        for marker in leak_markers:
            if marker in raw_answer:
                raw_answer = raw_answer.split(marker)[-1]

        code_match = re.search(r"```(?:json|markdown)?\s*(.*?)\s*```", raw_answer, re.S)
        if code_match:
            answer = code_match.group(1).strip()
        else:
            answer = re.sub(r"```[a-zA-Z]*", "", raw_answer).strip()

        if answer.startswith("{") and "answer" in answer:
            try:
                parsed = json.loads(answer)
                if "answer" in parsed:
                    answer = str(parsed["answer"]).strip()
            except Exception:
                pass

        answer = re.sub(
            r"^(?:final answer|answer|output|the answer is|response|答案是|回答|回答如下)[\s\n:：]*",
            "", answer, flags=re.I
        ).strip()
        answer = re.sub(
            r"^(?:scratchpad|thought|reasoning|思考過程)[\s\n:：]*",
            "", answer, flags=re.I
        ).strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        return answer

    def generate_answer(self, tokenizer, model, transcript: str, context: str, is_chinese: bool = True) -> Tuple[str, float]:
        if is_chinese:
            user_content = (
                f"【手冊資料】\n{context}\n\n"
                f"【問題】{transcript}\n\n"
                "請根據手冊資料，用繁體中文直接回答問題（1~2句話，不要有前言與分析）："
            )
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "答案是："}
            ]
        else:
            user_content = (
                f"PDF Evidence:\n{context}\n\n"
                f"Question: {transcript}\n\n"
                "Provide a direct factual answer in 1-2 sentences:"
            )
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "The answer is:"}
            ]

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        device = model.get_input_embeddings().weight.device
        encoded = {k: v.to(device) for k, v in dict(encoded).items() if torch.is_tensor(v)}
        input_len = encoded["input_ids"].shape[1]

        stop_ids = [tokenizer.eos_token_id]
        for token in ("<end_of_turn>", "<eos>", "\n\n", "```"):
            t_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(t_id, int) and t_id > 0 and t_id not in stop_ids:
                stop_ids.append(t_id)

        kwargs = {
            "max_new_tokens": 120,
            "do_sample": False,
            "repetition_penalty": 1.02,
            "use_cache": True,
            "eos_token_id": stop_ids,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or stop_ids[0],
        }

        start = time.time()
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        elapsed = time.time() - start

        raw_answer = tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
        answer = self.clean_model_output(raw_answer)

        if answer and not answer.startswith("答案是"):
            answer = "答案是：" + answer

        return answer, elapsed


class Critic_EN:
    SPECIAL_STOP_TOKENS = ("<end_of_turn>", "<eos>", "<|im_end|>", "<|eot_id|>")
    INVALID_PREDICTIONS = ("answer", "output", "empty", "none")
    NEG_MARKERS = (
        "incorrect",
        "contradict",
        "fails to",
        "does not capture",
        "off-topic",
        "unrelated",
        "hallucinat",
    )
    POS_MARKERS = (
        "accurately captures",
        "fully correct",
        "matches the reference",
        "correctly identifies",
        "fully matches",
    )
    PART_MARKERS = (
        "partially correct",
        "captures some factual details",
        "partially captures",
        "misses key",
    )

    def __init__(
        self,
        judge_model_path: Path,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ) -> None:
        judge_path = Path(judge_model_path)
        if not judge_path.exists():
            raise FileNotFoundError(
                f"Dedicated Judge Model path not found: {judge_path}"
            )

        print(f"[LOAD] Dedicated Judge Model: {judge_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(judge_path), trust_remote_code=True
        )

        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(judge_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(judge_path),
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            ).eval()

        self.device = self.model.get_input_embeddings().weight.device

        self.stop_ids: List[int] = []
        if isinstance(self.tokenizer.eos_token_id, list):
            self.stop_ids.extend(self.tokenizer.eos_token_id)
        elif self.tokenizer.eos_token_id is not None:
            self.stop_ids.append(self.tokenizer.eos_token_id)

        for token in self.SPECIAL_STOP_TOKENS:
            try:
                val = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(val, int) and val not in self.stop_ids and val >= 0:
                    self.stop_ids.append(val)
            except Exception:
                pass

        self.pad_token_id = self.tokenizer.eos_token_id or self.stop_ids[0]

    def judge_answer(
        self, gold_question: str, reference: str, prediction: str
    ) -> Tuple[int, str]:
        clean_pred = str(prediction or "").strip().lower()
        if not clean_pred or clean_pred in self.INVALID_PREDICTIONS:
            return 0, "Empty or truncated prediction"

        prompt = (
            "You are a strict academic evaluator. Your task is to evaluate the factual "
            "correctness of a Model Prediction compared to the Reference Ground Truth.\n\n"
            f"Benchmark Question: {gold_question}\n"
            f"Reference Ground Truth: {reference}\n"
            f"Model Prediction: {prediction}\n\n"
            "Evaluation Criteria:\n"
            "- Score 2: Fully correct. The core factual points/names/values are correctly "
            "captured. Minor phrasing differences, synonyms, or trailing sentence cutoffs "
            "do not reduce the score.\n"
            "- Score 1: Partially correct. Captures some factual details but misses key "
            "components or conditions.\n"
            "- Score 0: Incorrect, contradictory, off-topic, or completely hallucinated.\n\n"
            "Output strictly in valid JSON format without markdown code fences:\n"
            '{"score": 2, "reason": "brief explanation"}\n'
        )
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {
            k: v.to(self.device)
            for k, v in dict(encoded).items()
            if torch.is_tensor(v)
        }
        input_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.2,
                eos_token_id=self.stop_ids,
                pad_token_id=self.pad_token_id,
                use_cache=True,
            )
        raw_response = self.tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True
        ).strip()

        clean_response = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.I)
        clean_response = re.sub(r"\s*```$", "", clean_response).strip()

        json_blocks = re.findall(r"\{[\s\S]*?\}", clean_response)
        for block in reversed(json_blocks):
            try:
                parsed = json.loads(block)
                if "score" in parsed and str(parsed["score"]).strip() in ("0", "1", "2"):
                    return int(parsed["score"]), str(parsed.get("reason", "")).strip()
            except Exception:
                continue

        match = re.search(
            r'["\']?score["\']?\s*[:=]\s*["\']?([012])["\']?', clean_response, re.I
        )
        if match:
            score = int(match.group(1))
            reason_match = re.search(
                r'["\']?reason["\']?\s*[:=]\s*["\']?(.*?)(?:["\'\}\n]|$)',
                clean_response[match.end() :],
                re.I,
            )
            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "Parsed from score tag"
            )
            return score, reason

        resp_lower = clean_response.lower()
        has_negative = any(k in resp_lower for k in self.NEG_MARKERS)
        has_positive = any(k in resp_lower for k in self.POS_MARKERS)
        has_partial = any(k in resp_lower for k in self.PART_MARKERS)

        if has_positive and not has_negative:
            return 2, f"Judge affirmed correctness: {clean_response[:120]}"
        if has_partial or (has_positive and has_negative):
            return 1, f"Judge affirmed partial correctness: {clean_response[:120]}"
        if has_negative and not has_positive:
            return 0, f"Judge indicated failure: {clean_response[:120]}"

        return 0, f"Judge parse failed: {clean_response[:120]}"

    def lexical_metrics(self, reference: str, prediction: str) -> Dict[str, float]:
        ref = re.findall(r"\w+", (reference or "").lower(), flags=re.UNICODE)
        pred = re.findall(r"\w+", (prediction or "").lower(), flags=re.UNICODE)
        if not ref or not pred:
            return {"Token_F1": 0.0, "Precision": 0.0, "Recall": 0.0, "EM": 0.0}

        common = sum((Counter(ref) & Counter(pred)).values())
        precision = common / len(pred)
        recall = common / len(ref)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "Token_F1": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "EM": float(ref == pred),
        }

class ASR_EN:
    def __init__(
        self,
        asr_model: str,
        DEVICE: str,
        ASR_COMPUTE_TYPE: str,
    ) -> None:
        self.DEVICE = DEVICE
        self.asr = self.load_asr(asr_model, ASR_COMPUTE_TYPE)

    def load_asr(self, model_path: str, ASR_COMPUTE_TYPE: str) -> WhisperModel:
        print(f"[LOAD] Whisper: {model_path}")
        return WhisperModel(
            str(model_path), device=self.DEVICE, compute_type=ASR_COMPUTE_TYPE
        )

    def transcribe_audio(self, audio_input: Any) -> Tuple[str, float]:
        start = time.time()
        if isinstance(audio_input, torch.Tensor):
            audio = (
                audio_input.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
        else:
            audio = str(audio_input)

        segments, _ = self.asr.transcribe(
            audio,
            beam_size=3,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = norm(" ".join(seg.text.strip() for seg in segments if seg.text.strip()))
        return transcript, time.time() - start

class Gemma_EN:
    DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
        "<start_of_turn>model\n"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}<end_of_turn>\n"
        "{% endif %}{% endfor %}"
    )
    LEAK_MARKERS = (
        "<unused",
        "geqquser",
        "user\nYou are",
        "Question:",
        "PDF evidence:",
        "\n\nHuman:",
        "\n\nAssistant:",
        "]]></thought>",
        "<output>",
        "</output>",
        "]]>",
        "papers/",
        "# [A Survey",
        "## Write a very high-quality",
        "Let me think step-by-step:",
        "Let's analyze the relevant",
        "From the second piece of evidence:",
        "From the evidence:",
        "Based on the provided PDF evidence:",
    )
    SPECIAL_STOP_TOKENS = ("<end_of_turn>", "<eos>", "<unused53>")

    def __init__(self, MAX_NEW_TOKENS: int) -> None:
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS

    def load_gemma(self, model_path: Path) -> Tuple[Any, Any]:
        print(f"[LOAD] Gemma: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        return tokenizer, model

    def clean_model_output(self, raw_answer: str) -> str:
        if not raw_answer:
            return ""

        if "<start_of_turn>model" in raw_answer:
            raw_answer = raw_answer.split("<start_of_turn>model")[-1]
        if "<start_of_turn>user" in raw_answer:
            raw_answer = raw_answer.split("<start_of_turn>user")[0]

        for marker in self.LEAK_MARKERS:
            if marker in raw_answer:
                raw_answer = raw_answer.split(marker)[0]

        code_match = re.search(
            r"```(?:json|markdown)?\s*(.*?)\s*```", raw_answer, re.S
        )
        if code_match:
            answer = code_match.group(1).strip()
        else:
            answer = re.sub(r"```[a-zA-Z]*", "", raw_answer).strip()

        if answer.startswith("{") and "answer" in answer:
            try:
                parsed = json.loads(answer)
                if "answer" in parsed:
                    answer = str(parsed["answer"]).strip()
            except Exception:
                pass

        answer = re.sub(
            r"^(?:final answer|answer|output|the answer is|response)[\s\n:]*",
            "",
            answer,
            flags=re.I,
        ).strip()
        answer = re.sub(
            r"^(?:scratchpad|thought|reasoning)[\s\n:]*", "", answer, flags=re.I
        ).strip()
        return re.sub(r"\n{3,}", "\n\n", answer).strip()

    def _build_stop_ids(self, tokenizer: Any) -> List[int]:
        stop_ids = [tokenizer.eos_token_id]
        for token in self.SPECIAL_STOP_TOKENS:
            try:
                val = tokenizer.convert_tokens_to_ids(token)
                if isinstance(val, int) and val not in stop_ids and val >= 0:
                    stop_ids.append(val)
            except Exception:
                pass
        return stop_ids

    def _run_generation(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        device: torch.device,
        stop_ids: List[int],
        repetition_penalty: float,
    ) -> str:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {
            k: v.to(device)
            for k, v in dict(encoded).items()
            if torch.is_tensor(v)
        }
        input_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                eos_token_id=stop_ids,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or stop_ids[0],
            )
        raw_answer = tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True
        ).strip()
        return self.clean_model_output(raw_answer)

    def generate_answer(
        self, tokenizer: Any, model: Any, transcript: str, context: str
    ) -> Tuple[str, float]:
        user_content = (
            "You are a helpful assistant. Answer the question using ONLY the provided PDF evidence.\n"
            "Requirements:\n"
            "1. Directly provide the concise, factual answer in 1 to 2 complete sentences.\n"
            "2. Do NOT write any thoughts, intro, preamble, reasoning steps, or bullet points.\n"
            "3. Do NOT copy long paragraphs verbatim from the context.\n\n"
            f"PDF Evidence:\n{context}\n\n"
            f"Question: {transcript}"
        )
        device = model.get_input_embeddings().weight.device
        stop_ids = self._build_stop_ids(tokenizer)

        start = time.time()
        answer = self._run_generation(
            tokenizer=tokenizer,
            model=model,
            prompt=user_content,
            device=device,
            stop_ids=stop_ids,
            repetition_penalty=1.1,
        )
        elapsed = time.time() - start

        if len(answer) <= 6 or answer.lower() in ("answer", "output", ""):
            retry_cleaned = self._run_generation(
                tokenizer=tokenizer,
                model=model,
                prompt=user_content,
                device=device,
                stop_ids=stop_ids,
                repetition_penalty=1.0,
            )
            if len(retry_cleaned) > len(answer):
                answer = retry_cleaned

        if (
            "let me think" in answer.lower()
            or "the user wants me to" in answer.lower()
        ):
            direct_prompt = (
                f"PDF Evidence:\n{context}\n\n"
                f"Question: {transcript}\n\n"
                "State the final answer directly in 1-2 sentences. Do not write any thoughts or analysis."
            )
            cot_cleaned = self._run_generation(
                tokenizer=tokenizer,
                model=model,
                prompt=direct_prompt,
                device=device,
                stop_ids=stop_ids,
                repetition_penalty=1.0,
            )
            if len(cot_cleaned) > 8 and "let me think" not in cot_cleaned.lower():
                answer = cot_cleaned

        return answer, elapsed


class RAG:
    def __init__(self,EMBEDDING_MODEL,RERANKER_MODEL,MAX_CHUNK_CHARS,CHUNK_OVERLAP,TOP_DENSE,TOP_K,PDF_PATH,DEVICE):
        self.MAX_CHUNK_CHARS=MAX_CHUNK_CHARS
        self.CHUNK_OVERLAP=CHUNK_OVERLAP
        self.TOP_DENSE=TOP_DENSE
        self.TOP_K=TOP_K
        self.chunks = self.extract_pdf_chunks(PDF_PATH)
        print(f"[INIT] PDF chunks={len(self.chunks)}")

        self.embedder = SentenceTransformer(str(EMBEDDING_MODEL), device=DEVICE)
        self.reranker = CrossEncoder(str(RERANKER_MODEL), max_length=512, device=DEVICE)
        texts = [f"Page {c['page']}\n{c['text']}" for c in self.chunks]
        self.chunk_embeddings = self.embedder.encode(
            texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)



    def extract_pdf_chunks(self,pdf_path):
        chunks = []
        reader = PdfReader(str(pdf_path))
        for page_no, page in enumerate(reader.pages, 1):
            text = norm(page.extract_text() or "")
            if not text:
                continue
            start = 0
            part_no = 1
            while start < len(text):
                end = min(start + self.MAX_CHUNK_CHARS, len(text))
                if end < len(text):
                    candidates = [
                        text.rfind(". ", start, end),
                        text.rfind("; ", start, end),
                        text.rfind("\n", start, end),
                        text.rfind(" ", start, end),
                    ]
                    boundary = max(candidates)
                    if boundary > start + int(self.MAX_CHUNK_CHARS * 0.55):
                        end = boundary + 1
                piece = text[start:end].strip()
                if piece:
                    chunks.append({
                        "chunk_id": f"p{page_no:04d}_c{part_no:03d}",
                        "page": page_no,
                        "text": piece,
                    })
                    part_no += 1
                if end >= len(text):
                    break
                start = max(end - self.CHUNK_OVERLAP, start + 1)
        return chunks



    def retrieve(self,query):

        query = norm(query) or "Answer the question using the PDF document."

        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)

        dense_scores = self.chunk_embeddings @ query_embedding
        top_indices = np.argsort(dense_scores)[::-1][:min(self.TOP_DENSE, len(self.chunks))]
        candidates = [self.chunks[int(idx)] for idx in top_indices]

        pairs = [[query, f"Page {c['page']}\n{c['text']}"] for c in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, top_indices, rerank_scores),
            key=lambda item: float(item[2]),
            reverse=True,
        )[:self.TOP_K]

        retrieved = [chunk for chunk, idx, score in ranked]
        context = "\n\n---\n\n".join(
            f"[PDF page {c['page']} | {c['chunk_id']}]\n{c['text']}" for c in retrieved
        )
        return retrieved, context


class Datasets:
    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

    def __init__(self, SAMPLE_RATE: int) -> None:
        self.SAMPLE_RATE = SAMPLE_RATE

    @staticmethod
    def first_value(
        d: Dict[str, Any], keys: List[str], default: Any = ""
    ) -> Any:
        for key in keys:
            if key in d and d[key] not in (None, ""):
                return d[key]
        return default

    def load_questions(self, path: Path) -> List[Dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("questions", "data", "items", "qa", "records"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]

        result: List[Dict[str, Any]] = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            nested = (
                item.get("qa", {}) if isinstance(item.get("qa"), dict) else {}
            )
            qid = str(
                self.first_value(
                    item, ["qid", "QID", "id", "question_id"], f"Q{i:03d}"
                )
            ).strip()
            if qid.isdigit():
                qid = f"Q{int(qid):03d}"

            question = norm(
                self.first_value(
                    item,
                    ["question", "Question", "query", "text", "question_text"],
                )
            )
            if not question:
                question = norm(self.first_value(nested, ["question", "text"]))

            answer = norm(
                self.first_value(
                    item,
                    [
                        "answer",
                        "reference_answer",
                        "Reference_Answer",
                        "gold_answer",
                        "gold",
                        "reference",
                    ],
                )
            )
            if not answer:
                answer = norm(
                    self.first_value(
                        nested, ["answer", "reference_answer", "gold_answer"]
                    )
                )

            audio_field = str(
                self.first_value(
                    item,
                    [
                        "audio_path",
                        "Audio_Path",
                        "audio",
                        "wav_path",
                        "file_path",
                    ],
                )
            )

            result.append(
                {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "audio_field": audio_field,
                }
            )
        return result

    def resolve_audio_file(
        self, q: Dict[str, Any], folder: Path
    ) -> Optional[Path]:
        qid = q["qid"]
        audio_field = q.get("audio_field", "")
        stem = Path(audio_field).stem if audio_field else qid

        for ext in self.AUDIO_EXTENSIONS:
            for candidate_name in (stem, qid, qid.lower()):
                candidate_path = folder / f"{candidate_name}{ext}"
                if candidate_path.is_file():
                    return candidate_path.resolve()
        return None

    def load_audio_tensor(self, path: Path) -> torch.Tensor:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
        if sr != self.SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.SAMPLE_RATE
            )
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return torch.from_numpy(audio).unsqueeze(0).float()

    def save_audio_file(
        self, waveform: torch.Tensor, out_path: Path
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = (
            waveform.detach()
            .cpu()
            .float()
            .reshape(-1)
            .numpy()
            .astype(np.float32)
        )
        sf.write(str(out_path), arr, self.SAMPLE_RATE, subtype="PCM_16")
        return out_path

    def save_incremental_json_and_csv(
        self,
        rows: List[Dict[str, Any]],
        condition_names: List[str],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "four_conditions_comparison_results.json"
        csv_path = out_dir / "four_conditions_comparison_results.csv"
        summary_path = out_dir / "summary.json"

        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            k: (
                                json.dumps(v, ensure_ascii=False)
                                if isinstance(v, (list, dict))
                                else v
                            )
                            for k, v in row.items()
                        }
                    )

        total = len(rows)
        summary: Dict[str, Any] = {
            "total_processed": total,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": {},
        }
        for cname in condition_names:
            pfx = f"{cname}_"
            scores = [
                r.get(f"{pfx}judge_score", 0)
                for r in rows
                if isinstance(r.get(f"{pfx}judge_score"), (int, float))
            ]
            correct_cnt = sum(r.get(f"{pfx}is_correct") is True for r in rows)
            snr_list = [
                r.get(f"{pfx}SNR_dB", 0.0) for r in rows if f"{pfx}SNR_dB" in r
            ]
            sdr_list = [
                r.get(f"{pfx}SDR_dB", 0.0) for r in rows if f"{pfx}SDR_dB" in r
            ]

            summary["conditions"][cname] = {
                "correct_count": correct_cnt,
                "accuracy": round(correct_cnt / total, 4) if total else 0.0,
                "average_score": (
                    round(float(np.mean(scores)), 4) if scores else 0.0
                ),
                "average_SNR_dB": (
                    round(float(np.mean(snr_list)), 2) if snr_list else 0.0
                ),
                "average_SDR_dB": (
                    round(float(np.mean(sdr_list)), 2) if sdr_list else 0.0
                ),
            }

        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

class util:
    EPS: float = 1e-10

    def __init__(self) -> None:
        pass

    def save_results(self, rows: List[Dict[str, Any]], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "clean_vs_noisy_comparison_results.json"
        csv_path = out_dir / "clean_vs_noisy_comparison_results.csv"

        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not rows:
            return

        fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, dict))
                            else v
                        )
                        for k, v in row.items()
                    }
                )

    def compute_audio_metrics(
        self, estimated_audio: torch.Tensor, clean_reference: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        if clean_reference is None:
            return {"SNR_dB": 0.0, "SDR_dB": 0.0}

        ref = (
            clean_reference.detach()
            .cpu()
            .reshape(-1)
            .numpy()
            .astype(np.float64)
        )
        est = (
            estimated_audio.detach()
            .cpu()
            .reshape(-1)
            .numpy()
            .astype(np.float64)
        )

        min_len = min(len(ref), len(est))
        if min_len == 0:
            return {"SNR_dB": 0.0, "SDR_dB": 0.0}

        ref = ref[:min_len]
        est = est[:min_len]

        ref_power = float(np.sum(ref ** 2)) + self.EPS
        noise = est - ref
        noise_power = float(np.sum(noise ** 2)) + self.EPS
        snr_val = 10.0 * np.log10(ref_power / noise_power)

        scale = float(np.dot(est, ref)) / ref_power
        s_target = scale * ref
        e_noise = est - s_target

        target_energy = float(np.sum(s_target ** 2)) + self.EPS
        noise_energy = float(np.sum(e_noise ** 2)) + self.EPS
        sdr_val = 10.0 * np.log10(target_energy / noise_energy)

        return {
            "SNR_dB": round(float(snr_val), 2),
            "SDR_dB": round(float(sdr_val), 2),
        }

class audio_seperator:
    ALPHA_SP: float = 0.5

    def __init__(
        self,
        SEPARATED_AUDIO_DIR: Path,
        NON_SPEECH_AUDIO_DIR: Path,
        ENHANCED_AUDIO_DIR: Path,
        SEPARATOR_CKPT: Path,
        DEVICE: str,
        SAMPLE_RATE: int,
    ) -> None:
        SEPARATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        NON_SPEECH_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        ENHANCED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.SEPARATOR_CKPT = SEPARATOR_CKPT
        self.DEVICE = DEVICE
        self.SAMPLE_RATE = SAMPLE_RATE
        self.separator = self.load_separator()

    def load_separator(self) -> SealGPT_Sep:
        print(f"[LOAD] SealGPT_Sep checkpoint: {self.SEPARATOR_CKPT}")
        if not self.SEPARATOR_CKPT.is_file():
            raise FileNotFoundError(
                f"Separator checkpoint not found: {self.SEPARATOR_CKPT}"
            )
        separator = SealGPT_Sep(output_channels=1, dprnn=False)
        ckpt = torch.load(
            str(self.SEPARATOR_CKPT), map_location="cpu", weights_only=False
        )
        separator.load_state_dict(ckpt["generator"])
        return separator.eval().to(self.DEVICE)

    @torch.inference_mode()
    def separate_speech(
        self, audio_cpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = audio_cpu.to(self.DEVICE)
        if audio.shape[-1] > int(self.SAMPLE_RATE * 10):
            outputs = self.separator.chunk_inference(
                {"mixture": audio[:, None, :].float()},
                NL=1.0,
                NC=8.0,
                NR=1.0,
                sr=self.SAMPLE_RATE,
            )
            speech = torch.from_numpy(outputs[0, :]).float()
        else:
            outputs = self.separator({"mixture": audio[:, None, :].float()})
            speech = (
                outputs["pred_speech"][:, 0, :]
                .detach()
                .float()
                .cpu()
                .squeeze(0)
            )

        speech = speech.cpu().reshape(1, -1)
        raw_cpu = audio_cpu.cpu().reshape(1, -1)

        min_len = min(speech.shape[-1], raw_cpu.shape[-1])
        speech_aligned = speech[:, :min_len]
        raw_aligned = raw_cpu[:, :min_len]

        non_speech = raw_aligned - speech_aligned
        return speech_aligned, non_speech

    def apply_mafb_enhancement(
        self, raw_audio: torch.Tensor, sep_audio: torch.Tensor
    ) -> torch.Tensor:
        raw_flat = raw_audio.detach().cpu().reshape(-1)
        sep_flat = sep_audio.detach().cpu().reshape(-1)
        min_len = min(raw_flat.shape[0], sep_flat.shape[0])
        raw_aligned = raw_flat[:min_len]
        sep_aligned = sep_flat[:min_len]

        enhanced = (
            self.ALPHA_SP * sep_aligned + (1.0 - self.ALPHA_SP) * raw_aligned
        )
        peak = float(torch.max(torch.abs(enhanced)))
        if peak > 1.0:
            enhanced = enhanced / peak
        return enhanced.unsqueeze(0)