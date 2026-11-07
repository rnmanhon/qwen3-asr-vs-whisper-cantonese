# Results — 7 November 2026, NVIDIA GB10

500 clips from WSYue-ASR-eval Short (revision `0f0596e`): all 29 elderly, teen and child clips plus
471 random adult clips, seed 42. 40.2 minutes of audio; clips 2.0–10.9 s, median 4.1 s. Every
clip was re-encoded to webm/opus 48 kHz 32 kbit/s. Both engines used `language=yue`.

Spoken-language check: SenseVoice-Small (`language="auto"`) tagged 492 of the 500 clips as Cantonese
(`yue`), 4 as `zh` (sung lyrics), 2 as `ko` (a sound effect and an exclamation) and 1 as `en` (a 2-second
clip), plus 1 two-second clip of 唔該 tagged `zh`. Rescoring on the 492 Cantonese-tagged clips changes every
CER below by less than 0.1 point (Qwen3-ASR 7.04%, faster-whisper 13.07–14.95%, OpenAI Whisper 16.23–16.24%).

## Accuracy

| Engine | CER | Substituted | Deleted | Inserted | Elderly CER (21 clips) | Cantonese chars kept | Extra standard-Chinese chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | **7.10%** | 5.20% | 0.68% | 1.22% | **4.49%** | **97.2%** | **3** |
| Whisper INT8, beam 5 + VAD | 13.09% | 9.00% | 2.95% | 1.15% | 8.38% | 83.9% | 51 |
| Whisper FP16, beam 5 + VAD | 13.40% | 9.21% | 2.95% | 1.24% | 8.68% | 83.3% | 59 |
| Whisper FP16, greedy | 14.82% | 10.18% | 3.51% | 1.14% | 8.38% | 77.0% | 80 |

Per clip, Qwen3-ASR against Whisper FP16 beam 5 + VAD: better on 291, worse on 63, tied on 146.
The bootstrap 95% interval (2,000 resamples of clips) for Whisper's extra CER is 5.30–7.24
points (FP16 beam 5 + VAD), 5.09–6.91 (INT8) and 6.64–8.89 (FP16 greedy).

## Probes

| Engine | Silence / noise clips returning text (of 12) | Invented characters in 1.5 s fragments (60 clips) |
| --- | ---: | ---: |
| Qwen3-ASR 1.7B | 9, one character each (係。 嗯。 好。) | 4.3% |
| Whisper INT8, beam 5 + VAD | 0 | 25.0% |
| Whisper FP16, beam 5 + VAD | 0 | 24.8% |
| Whisper FP16, greedy | 11 (字幕由 Amara.org 社群提供; 唔好 repeated) | 23.7% |

## Speed (120 clips, 9.6 minutes of audio)

End-to-end seconds per request: median / 95th percentile.

| Engine | 1 at a time | 2 at once | 4 at once | 8 at once | Max clips/s | Peak GPU memory |
| --- | --- | --- | --- | --- | ---: | ---: |
| Qwen3-ASR (vLLM, max-num-seqs 4) | 0.44 / 0.94 | 0.39 / 0.80 | 0.39 / 0.81 | 0.76 / 1.12 | 10.3 | ~15.2 GB reserved |
| Whisper FP16 beam 5 + VAD, 1 worker | 0.56 / 0.99 | 1.07 / 1.61 | 2.19 / 2.90 | 4.48 / 5.43 | 1.8 | 4.2 GB |
| Whisper FP16 beam 5 + VAD, 4 workers | 0.55 / 0.98 | 0.90 / 1.62 | 1.64 / 3.49 | 3.51 / 5.78 | 2.2 | 5.6 GB |
| Whisper FP16 greedy, 1 worker | 0.54 / 0.94 | 1.04 / 1.57 | 2.17 / 2.79 | 4.37 / 5.36 | 1.8 | 4.2 GB |
| Whisper INT8 beam 5 + VAD, 1 worker | 0.51 / 0.90 | 0.96 / 1.44 | 1.96 / 2.56 | 4.03 / 4.96 | 2.0 | 2.4 GB |

Start-up: faster-whisper loaded in 1.8 s with weights cached. vLLM answered its first request
about 3–4 minutes after container start, with its compilation cache warm.

## CPU only: Qwen3-ASR vs faster-whisper vs OpenAI Whisper

Same 500 clips and probes. OpenAI Whisper greedy is matched to faster-whisper greedy (beam 1,
temperature 0, no previous-text conditioning, no VAD); both libraries got the same decoded audio.

| Model | CER | Substituted | Deleted | Inserted | Cantonese chars kept | Silence/noise with text | Median s/clip (CPU) | Time ÷ audio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B, BF16 (Transformers) | **6.56%** | 4.64% | 0.75% | 1.16% | **97.2%** | 12/12 | **3.11** | **0.83** |
| faster-whisper INT8, beam 5 + VAD | 13.09% | 8.97% | 3.01% | 1.11% | 83.9% | 0/12 | 5.25 | 1.19 |
| faster-whisper INT8, greedy | 14.43% | 9.95% | 3.40% | 1.08% | 78.9% | 9/12 | 4.87 | 1.09 |
| faster-whisper FP32, greedy | 14.87% | 10.23% | 3.51% | 1.14% | 77.0% | 11/12 | 8.75 | 2.04 |
| OpenAI Whisper FP32, greedy | 16.24% | 10.95% | 4.02% | 1.28% | 74.0% | 10/12 | 15.54 | 3.92 |

Qwen3-ASR in FP32 on CPU: 5.44 s median per clip (1.48× audio); 6.24% CER on the first 100 clips.
95th-percentile seconds per clip: Qwen3-ASR BF16 8.36, FP32 16.24; faster-whisper INT8 beam 5 + VAD 9.06,
INT8 greedy 7.47, FP32 greedy 14.67; OpenAI Whisper 35.38.

- Qwen3-ASR on CPU against faster-whisper INT8 beam 5 + VAD: better on 296 clips, worse on 55, tied on 149;
  Whisper's extra CER 5.67–7.47 points (bootstrap 95% interval). Extra standard-Chinese characters: 2.
- Qwen3-ASR on CPU (Transformers, BF16) against the GPU vLLM run: 359 of 500 transcripts identical after
  normalisation; CPU CER lower by 0.22–0.85 points. The runs differ in serving stack, architecture variant
  (vLLM used `Qwen3ASRRealtimeGeneration`) and precision, so this is not a device effect.
- Qwen3-ASR on CPU returned text on all 12 silence/noise probes: 係。 on most, 係啊！ on the 1.5 s and 3 s
  hiss, 係啊！係 on 6 s silence and hiss, and 係啊！ four times on 6 s noise. Invented characters in
  1.5 s fragments: 5.2%.

- OpenAI Whisper's extra CER against faster-whisper (bootstrap 95% interval): 0.93–2.79 points
  against INT8 greedy, 0.62–2.24 against FP32 greedy, 2.14–4.18 against INT8 beam 5 + VAD.
  Per clip against INT8 greedy, faster-whisper was better on 90, worse on 56 and tied on 354.
- Ten clips where OpenAI Whisper rewrote the whole sentence in standard written Chinese account for
  84 of its 122 extra character errors against faster-whisper greedy. Without them the CERs are 15.45%
  (OpenAI) and 14.99% (faster-whisper).
- Device does not change Whisper transcripts. OpenAI Whisper on CPU FP32 matched GPU FP32 and GPU FP16 on all
  60 clips checked, and GPU FP16 matched GPU FP32 on 493 of 500. faster-whisper on CPU FP32 matched
  GPU FP16 on 496 of 500. The OpenAI full-set row was run in FP32 on the GPU for that reason. OpenAI
  GPU FP16 scored 16.23%.
- GPU INT8 greedy (`int8_float16`) for faster-whisper scored 14.96%.
- CPU timing: GB10 ARM CPU (BF16-capable), 8 threads, first 30 clips (143 s of audio, 4.8 s average), one
  configuration at a time with other services idle, one warm-up clip, decoding excluded. Whisper encodes a fixed 30-second window, so short clips
  cost relatively more than in our earlier 6–9 second recordings (0.73–0.84× audio length for INT8).
- Elderly-only CER for CPU configurations ranges 8.4–13.8% on 21 clips. That is too few clips to compare.

## Versions

- Qwen/Qwen3-ASR-1.7B `7278e1e`; vLLM 0.29.0, Transformers 5.18.0.dev0, PyTorch 2.13.0+cu130
- Qwen/Qwen3-ASR-1.7B-hf `bcd2b5b` (CPU runs); Transformers 5.18.0.dev0, PyTorch 2.13.0
- Systran/faster-whisper-large-v3 `edaa852`; faster-whisper 1.2.1, CTranslate2 4.8.5 (CUDA 13 source build)
- OpenAI Whisper: openai-whisper 20231117, official `large-v3` checkpoint, PyTorch 2.9.1+cu130
- PyAV 18.1.0, OpenCC 1.4.2, Python 3.11

## Limits

Read and broadcast-style speech, not spontaneous phone recordings; only 21 elderly clips.
WSYue-ASR-eval comes from the same lab as the WenetSpeech-Yue training corpus, and Qwen3-ASR's
training data is not public, so an in-domain advantage cannot be ruled out. Some references use
standard-Chinese wording for Cantonese speech. Timings are single runs on a shared machine, with
other GPU services idle but a large language model loaded (GPU runs).
CPU runs are one request at a time; CPU memory was not measured.
