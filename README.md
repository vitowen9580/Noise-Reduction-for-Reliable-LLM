# Noise Reduction for Reliable LLM

This project investigates the impact of background noise and speech enhancement on Large Language Model (LLM)-based question answering in industrial environments.

## System Architecture

<p align="center">
  <img src="Figure/Figure1.png" alt="System Architecture" width="100%">
</p>

## Experimental Workflow

### Step 1. Generate Question Dataset and Speech Audio

1. Provide the original PDF to an LLM (e.g., ChatGPT) to generate question-answer pairs in JSON format.
2. Place both the original PDF and the generated JSON file in the `dataset/` directory.
3. Use [ElevenLabs Text to Speech](https://elevenlabs.io/app/speech-synthesis/text-to-speech) to convert the questions into MP3 audio.
4. Convert MP3 files to WAV:

```bash
python mp3towav.py --input $folder$ --output datasets/testB
```

**Parameters:**
- `--input`: Input folder containing MP3 files.
- `--output`: Output folder for converted WAV files.

**Final directory structure:**

```text
dataset/
├── syntec220plus30qa.pdf
├── syntec_220plus_30qa.json
└── testB/
    ├── Q001.wav
    ├── Q002.wav
    └── ...
```

### Step 2. Generate Noisy Audio

1. Download the **UrbanSound8K** dataset:

[UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

After downloading, extract the dataset into the `datasets/` directory.

For example:

```text
datasets/
├── UrbanSound8K/
├── testA/
└── testB/
```

2. Run the following command:

   ```bash
   python white_noise_dataset_generator.py --SNR -8 -type 2,4,7
   ```

   **Parameters:**

   - **SNR:** Signal-to-Noise Ratio. Available values: `[-8, -4, 4]`.
     - Lower SNR indicates stronger background noise.
   - **type:** Background noise category. Available values: `0–9`.
     - Multiple noise types can be selected using commas (`,`).
     - Example: `-type 2,4,7` selects `children_playing`, `drilling`, and `jackhammer`.

     | Type | Noise |
     |---|---|
     | 0 | air_conditioner |
     | 1 | car_horn |
     | 2 | children_playing |
     | 3 | dog_bark |
     | 4 | drilling |
     | 5 | engine_idling |
     | 6 | gun_shot |
     | 7 | jackhammer |
     | 8 | siren |
     | 9 | street_music |

The generated noisy audio will be automatically saved in:

```text
datasets/testA/
```

### Step 3. Run Noise Reduction and LLM Evaluation

1. Make sure the following dataset files and directories are available:

   ```text
   datasets/
   ├── syntec220plus30qa.pdf
   ├── syntec_220plus_30qa.json
   ├── UrbanSound8K/
   ├── testA/
   └── testB/
   ```

2. Run the evaluation separately for Chinese and English audio:

   **Chinese:**
   ```bash
   python main.py --type CH
   ```

   **English:**
   ```bash
   python main.py --type EN
   ```

### Step 4. Visualize Results

After completing the experiments, generate the accuracy comparison plot:

```bashhttps://github.com/vitowen9580/Noise-Reduction-for-Reliable-LLM/tree/main
python plot_accuracy_comparison.py
```

## Experimental Results

### English Results (EN)

<p align="center">
  <img src="result/EN/Chart1_Grouped_Single_Noises.png" width="32%">
  <img src="result/EN/Chart2_Grouped_Noise_Complexity.png" width="32%">
  <img src="result/EN/Chart3_Grouped_SNR_Levels.png" width="32%">
</p>

### Chinese Results (CH)

<p align="center">
  <img src="result/CH/Chart1_Grouped_Single_Noises.png" width="32%">
  <img src="result/CH/Chart2_Grouped_Noise_Complexity.png" width="32%">
  <img src="result/CH/Chart3_Grouped_SNR_Levels.png" width="32%">
</p>

## Acknowledgements and References

This project is developed with reference to the following works:

**Code Development References**
- [Noise2Noise: Audio Denoising without Clean Training Data](https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data)
- *Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models.*

**Research Reference**
- *Robust Tool Wear Prediction under Novel Operating Conditions via Physics-Guided Unsupervised Domain Adaptation.*

We sincerely thank the authors for their valuable contributions and research, which provided useful references for the development and experimental design of this project.
