# Qwen3-ASR vs Whisper large-v3 for Cantonese

Compare [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (served by vLLM on GPU, or through
Transformers on CPU),
[faster-whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) and
[OpenAI Whisper large-v3](https://github.com/openai/whisper), on GPU or CPU, using the
hand-labelled Cantonese benchmark [WSYue-ASR-eval](https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval):

- character error rate (CER) with a substitution / deletion / insertion breakdown
- how much written-Cantonese wording each model keeps
- behaviour on silence, noise and 1.5-second fragments of speech
- latency and throughput with 1, 2, 4 and 8 simultaneous requests

Results from our run on an NVIDIA GB10: [RESULTS.md](RESULTS.md).

## Requirements

- Python 3.10+ with the packages in `requirements.txt`. faster-whisper needs a CUDA-enabled
  CTranslate2 for GPU runs. On ARM64 (GB10, Jetson) the PyPI wheel is CPU-only, so build
  CTranslate2 from source, as described in
  [whisper-cpu-gpu-benchmark](https://github.com/rnmanhon/whisper-cpu-gpu-benchmark/blob/main/standalone/README.md).
- A server with an OpenAI-compatible `/v1/audio/transcriptions` endpoint serving Qwen3-ASR,
  for example vLLM. Our settings:

  ```sh
  vllm serve Qwen/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --gpu-memory-utilization 0.20 --max-model-len 8192 --max-num-seqs 4 \
    --hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}'
  ```

  vLLM needs its audio extras (`pip install "vllm[audio]"`). We used vLLM 0.29.0.

## Data

Download `Short/content.txt` and `Short/wav.tar.gz` (about 930 MB) from
[ASLP-lab/WSYue-ASR-eval](https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval/tree/main/Short)
into one folder. The dataset is licensed CC BY-NC 4.0; no audio is included here.

```sh
mkdir -p data
curl -L -o data/content.txt https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval/resolve/main/Short/content.txt
curl -L -o data/wav.tar.gz  https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval/resolve/main/Short/wav.tar.gz
```

## Run

```sh
# 1. Sample 500 clips (all elderly/teen/child speakers + random adults, seed 42),
#    re-encode to webm/opus, and generate silence/noise and partial-clip probes.
python benchmark.py prepare --data data --n 500

# 2. faster-whisper: greedy and beam 5 + VAD (FP16); repeat with another compute type if wanted.
python benchmark.py whisper --data data
python benchmark.py whisper --data data --compute-type int8_float16 --config beam5_vad

#    faster-whisper on CPU (8 threads).
python benchmark.py whisper --data data --device cpu --compute-type int8 --config greedy beam5_vad
python benchmark.py whisper --data data --device cpu --compute-type float32 --config greedy

#    OpenAI Whisper, greedy (matched to faster-whisper's greedy config). FP32 on GPU gave the same
#    transcripts as FP32 on CPU in our 60-clip check, and is far quicker for the full set.
python benchmark.py openai --data data --device cuda
python benchmark.py openai --data data --device cpu --threads 8

# 3. Qwen3-ASR through the vLLM server.
python benchmark.py qwen --data data --url http://127.0.0.1:8000

#    Qwen3-ASR without a server, through Transformers (Qwen/Qwen3-ASR-1.7B-hf), on CPU in BF16.
#    Needs a CPU with BF16 support to be fast; use --dtype float32 otherwise.
python benchmark.py qwen-local --data data --device cpu --dtype bfloat16 --threads 8

# 4. Score every result file in data/results/ -> data/report.json
python benchmark.py score --data data

# 5. Latency and throughput under 1/2/4/8 simultaneous requests (run one engine at a time,
#    with nothing else using the GPU).
python benchmark.py perf --data data --engine qwen --url http://127.0.0.1:8000 --tag qwen
python benchmark.py perf --data data --engine whisper --tag whisper_fp16
python benchmark.py perf --data data --engine whisper --compute-type int8_float16 --tag whisper_int8
python benchmark.py perf --data data --engine whisper --workers 4 --tag whisper_fp16_w4
```

`--model` accepts a local CTranslate2 folder, and `--openai-model` a local `large-v3.pt`, for
offline use. `python benchmark.py <command> -h` lists every option.

Timing note: the `whisper`, `openai`, `qwen` and `qwen-local` commands time each request including audio decoding.
Our published CPU timings excluded decoding and used 30 clips, each configuration measured alone;
expect small differences.

## How it scores

Both reference and output are converted to Simplified characters (OpenCC `t2s`; the labels are
Simplified), lower-cased, Arabic numerals spelled as Chinese numerals, and punctuation and spaces
removed. CER is pooled over all characters. "Cantonese characters kept" counts how many of the
reference's 嘅 咗 喺 唔 哋 佢 冇 嘢 啲 ... appear in the output. Probe results report any text
produced for silence or noise, and the share of output characters on 1.5-second fragments that
appear nowhere in the full reference sentence.

GPU memory is not measured by the script. We sampled `nvidia-smi --query-compute-apps` once a
second during the `perf` runs.

## Notes

- The published numbers came from an earlier internal version of this script with the same
  sampling, encoding settings, engines and scoring. That version decoded the source WAV files with
  soundfile and torchaudio, where this one uses PyAV, so re-encoded clips may differ very slightly.
  Rescoring the original outputs with this script reproduces the published accuracy figures exactly.
- Output JSON contains transcripts of the dataset's audio. Review its licence before sharing them.

## Upstream projects

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) ·
[vLLM](https://github.com/vllm-project/vllm) ·
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[CTranslate2](https://github.com/OpenNMT/CTranslate2) ·
[WSYue-ASR-eval](https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval) ·
[OpenCC](https://github.com/BYVoid/OpenCC)

Dependencies, models and the dataset keep their own licences.
