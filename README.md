# Noise Reduction for Reliable LLM

This project investigates the impact of background noise and speech enhancement on Large Language Model (LLM)-based question answering in industrial environments.

## System Architecture

<p align="center">
  <img src="Figure/architecture.png" alt="System Architecture" width="100%">
</p>

## Experimental Workflow

### Step 1. Generate Question Dataset and Speech Audio

1. Provide the original PDF to an LLM (e.g., ChatGPT) to generate question-answer pairs in JSON format.
2. Save the generated JSON file as `dataset/syntec_220plus_30qa.json`.
3. Use [ElevenLabs Text to Speech](https://elevenlabs.io/app/speech-synthesis/text-to-speech) to convert the questions into MP3 audio.
4. Convert MP3 files to WAV:

```bash
python mp3towav.py
