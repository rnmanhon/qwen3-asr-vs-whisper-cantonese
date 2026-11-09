#!/usr/bin/env python3
"""Qwen3-ASR vs Whisper large-v3 (faster-whisper and OpenAI Whisper) on Cantonese.

Subcommands, run in order:

  prepare  sample clips from WSYue-ASR-eval Short, re-encode them to webm/opus,
           and generate silence/noise and 1.5-second partial-clip probes
  whisper  transcribe every clip and probe with faster-whisper (GPU or CPU)
  openai   transcribe every clip and probe with OpenAI Whisper (GPU or CPU), greedy
  qwen     transcribe every clip and probe through an OpenAI-compatible
           /v1/audio/transcriptions endpoint (for example vLLM serving Qwen3-ASR)
  qwen-local  transcribe with Qwen3-ASR in-process through Transformers (CPU or GPU)
  score    character error rate (CER), Cantonese-word retention, probe behaviour
  perf     latency and throughput with 1, 2, 4 and 8 simultaneous requests

Every subcommand takes --data DIR. It must contain content.txt and wav.tar.gz from
https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval (folder Short/).
"""
import argparse
import io
import json
import os
import random
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np

SR = 16000

WHISPER_CONFIGS = {
    # Greedy decoding: beam 1, temperature 0, no VAD filter.
    "greedy": dict(beam_size=1, best_of=1, temperature=0, condition_on_previous_text=False, vad_filter=False),
    # faster-whisper's default beam size, default temperature fallback, Silero VAD filter on.
    "beam5_vad": dict(beam_size=5, condition_on_previous_text=False, vad_filter=True),
}


def paths(data):
    return {
        "clips": os.path.join(data, "clips"),
        "probes": os.path.join(data, "probes"),
        "results": os.path.join(data, "results"),
        "meta": os.path.join(data, "meta.json"),
        "probe_meta": os.path.join(data, "probes.json"),
    }


# ---------------------------------------------------------------- prepare
def decode_to_16k(data: bytes) -> np.ndarray:
    """Any audio file -> float32 mono 16 kHz."""
    with av.open(io.BytesIO(data)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def encode_webm(pcm: np.ndarray, path: str):
    """float32 mono 16 kHz -> webm/opus, 48 kHz, 32 kbit/s (similar to a browser MediaRecorder)."""
    out = av.open(path, "w", format="webm")
    st = out.add_stream("libopus", rate=48000)
    st.bit_rate = 32000
    st.layout = "mono"
    res = av.AudioResampler(format="s16", layout="mono", rate=48000)
    frame = av.AudioFrame.from_ndarray((np.clip(pcm, -1, 1) * 32767).astype(np.int16)[None, :],
                                       format="s16", layout="mono")
    frame.sample_rate = SR
    for f in res.resample(frame) + res.resample(None):
        for p in st.encode(f):
            out.mux(p)
    for p in st.encode(None):
        out.mux(p)
    out.close()


def cmd_prepare(a):
    P = paths(a.data)
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(a.data, "content.txt"), encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    # Keep every clip outside the two large adult age groups (elderly, teens, children), then sample adults.
    keep_all = [r for r in rows if not r[1].startswith(("青年", "中年"))]
    adults = [r for r in rows if r[1].startswith(("青年", "中年"))]
    sample = keep_all + rng.sample(adults, a.n - len(keep_all))
    want = {r[0]: r for r in sample}
    os.makedirs(P["clips"], exist_ok=True)
    os.makedirs(P["probes"], exist_ok=True)
    meta, partial_src = [], []
    with tarfile.open(os.path.join(a.data, "wav.tar.gz")) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not m.isfile() or name not in want:
                continue
            pcm = decode_to_16k(tf.extractfile(m).read())
            cid = name[:-4]
            encode_webm(pcm, os.path.join(P["clips"], cid + ".webm"))
            r = want[name]
            meta.append({"id": cid, "age": r[1], "gender": r[3], "ref": r[4], "dur": round(len(pcm) / SR, 2)})
            if len(pcm) > SR * 4 and len(partial_src) < 60:
                partial_src.append((cid, pcm))
    json.dump(meta, open(P["meta"], "w"), ensure_ascii=False, indent=0)

    g = np.random.default_rng(a.seed)
    probes = []
    for secs in (0.5, 1.5, 3.0, 6.0):
        for kind, amp in (("silence", 0.0), ("hiss", 0.003), ("noise", 0.02)):
            pcm = (g.standard_normal(int(SR * secs)) * amp).astype(np.float32)
            pid = f"{kind}_{secs}s"
            encode_webm(pcm, os.path.join(P["probes"], pid + ".webm"))
            probes.append({"id": pid, "kind": kind})
    for cid, pcm in partial_src:  # the first 1.5 s of an utterance, as live captioning sees it
        pid = f"partial_{cid}"
        encode_webm(pcm[: int(SR * 1.5)], os.path.join(P["probes"], pid + ".webm"))
        probes.append({"id": pid, "kind": "partial", "src": cid})
    json.dump(probes, open(P["probe_meta"], "w"), indent=0)
    print(f"prepared {len(meta)} clips ({sum(m['dur'] for m in meta) / 60:.1f} min) and {len(probes)} probes")


def all_items(P, limit=0):
    """All clips plus the probes; with limit, only the first `limit` clips (no probes), as in the
    30-clip timing runs."""
    meta = json.load(open(P["meta"]))
    clips = [(m["id"], os.path.join(P["clips"], m["id"] + ".webm")) for m in meta]
    if limit:
        return clips[:limit]
    probes = json.load(open(P["probe_meta"]))
    return clips + [(p["id"], os.path.join(P["probes"], p["id"] + ".webm")) for p in probes]


# ---------------------------------------------------------------- engines
def load_whisper(a, workers=1):
    from faster_whisper import WhisperModel
    return WhisperModel(a.model, device=a.device, compute_type=a.compute_type, num_workers=workers,
                        cpu_threads=a.threads)


def whisper_fn(model, cfg_name):
    cfg = WHISPER_CONFIGS[cfg_name]

    def fn(path):
        segs, _ = model.transcribe(path, language="yue", task="transcribe", **cfg)
        return "".join(s.text for s in segs).strip()  # segments are lazy: consume them inside the timer
    return fn


def openai_fn(a):
    """OpenAI Whisper, greedy decoding matched to faster-whisper's "greedy" config."""
    import torch
    import whisper
    from faster_whisper.audio import decode_audio  # same 16 kHz decoding as faster-whisper uses
    torch.set_num_threads(a.threads)
    model = whisper.load_model(a.openai_model, device=a.device)

    def fn(path):
        r = model.transcribe(decode_audio(path), language="yue", task="transcribe", temperature=0.0,
                             condition_on_previous_text=False, fp16=a.fp16, verbose=None)
        return r["text"].strip()
    return fn


def qwen_local_fn(a):
    """Qwen3-ASR-1.7B-hf in-process with Transformers (>= 5.13), e.g. on a CPU without a model server."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    torch.set_num_threads(a.threads)
    dtype = getattr(torch, a.dtype)
    processor = AutoProcessor.from_pretrained(a.hf_model)
    model = AutoModelForMultimodalLM.from_pretrained(a.hf_model, dtype=dtype).to(a.device).eval()

    def fn(path):
        with open(path, "rb") as f:
            audio = decode_to_16k(f.read())
        inputs = processor.apply_transcription_request(audio=audio, language="yue").to(a.device, dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.decode(gen, return_format="transcription_only")[0].strip()
    return fn


# Qwen3-ASR's raw output starts with "language <name><asr_text>"; vLLM removes it, llama.cpp's
# llama-server doesn't.
ASR_TAG = re.compile(r"^\s*language\s+[^<]*?<asr_text>", re.IGNORECASE)


def wav_16k(data: bytes) -> bytes:
    """Any audio file -> 16 kHz mono 16-bit WAV (llama-server decodes only WAV/MP3/FLAC itself)."""
    import wave
    pcm = decode_to_16k(data)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def qwen_fn(a):
    """An OpenAI-compatible Qwen3-ASR server: vLLM, or llama.cpp's llama-server (use --wav)."""
    import requests
    sess = requests.Session()
    url = a.url.rstrip("/") + "/v1/audio/transcriptions"

    def fn(path):
        with open(path, "rb") as f:
            audio = f.read()
        name, ctype = os.path.basename(path), "audio/webm"
        if a.wav:  # converted inside the timer, as an application would per request (~10 ms)
            audio, name, ctype = wav_16k(audio), "clip.wav", "audio/wav"
        r = sess.post(url, data={"model": a.served_model, "language": "yue"},
                      files={"file": (name, audio, ctype)}, timeout=300)
        r.raise_for_status()
        return ASR_TAG.sub("", r.json().get("text", ""), count=1)
    return fn


def transcribe_all(P, fn, name, limit=0):
    items = all_items(P, limit)
    fn(items[0][1])  # warm-up
    out = {}
    for cid, path in items:
        t = time.perf_counter()
        try:
            text = fn(path)
        except Exception as e:  # record and continue
            text = f"<<ERROR {e}>>"
        out[cid] = {"text": text, "sec": round(time.perf_counter() - t, 3)}
    os.makedirs(P["results"], exist_ok=True)
    json.dump(out, open(os.path.join(P["results"], name + ".json"), "w"), ensure_ascii=False, indent=0)
    secs = sorted(v["sec"] for v in out.values())
    print(name, "done:", len(out), "items; median", secs[len(secs) // 2], "s per item")


def cmd_whisper(a):
    P = paths(a.data)
    model = load_whisper(a)
    for cfg in a.config:
        transcribe_all(P, whisper_fn(model, cfg), f"whisper_{a.device}_{a.compute_type}_{cfg}", a.limit)


def cmd_openai(a):
    transcribe_all(paths(a.data), openai_fn(a), f"openai_{a.device}_{'fp16' if a.fp16 else 'fp32'}_greedy", a.limit)


def cmd_qwen(a):
    transcribe_all(paths(a.data), qwen_fn(a), a.name, a.limit)


def cmd_qwen_local(a):
    transcribe_all(paths(a.data), qwen_local_fn(a), f"qwen3_asr_local_{a.device}_{a.dtype}", a.limit)


# ---------------------------------------------------------------- scoring
DIG = "零一二三四五六七八九"


def num2zh(m):
    s = m.group(0)
    if len(s) > 4 or s.startswith("0"):
        return "".join(DIG[int(c)] for c in s)  # years and phone numbers: digit by digit
    n, out = int(s), ""
    for v, u in ((1000, "千"), (100, "百"), (10, "十")):
        if n >= v:
            out += DIG[n // v] + u
            n %= v
        elif out and n and not out.endswith("零"):
            out += "零"
    if n:
        out += DIG[n]
    if out.startswith("一十"):
        out = out[1:]
    return out or "零"


PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def norm(text, cc):
    """Simplified script, lower case, numerals as Chinese characters, no punctuation or spaces."""
    text = cc.convert(text).lower()
    text = re.sub(r"\d+", num2zh, text)
    return PUNCT.sub("", text)


def sdi(a, b):
    """Substitutions, deletions, insertions turning reference a into hypothesis b."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            S += a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            D, i = D + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, D, I


CANTONESE = "嘅咗喺唔哋佢冇嘢啲咁嚟睇噉咩乜嗰俾畀啱攞揾搵瞓"  # characters typical of written Cantonese
MANDARIN = "的了在是他们没这那么"  # common standard-Chinese substitutes


def cmd_score(a):
    import opencc
    cc = opencc.OpenCC("t2s")  # the labels are in Simplified characters
    P = paths(a.data)
    meta = json.load(open(P["meta"]))
    probes = json.load(open(P["probe_meta"]))
    refs = {m["id"]: norm(m["ref"], cc) for m in meta}
    report = {}
    for fname in sorted(os.listdir(P["results"])):
        if not fname.endswith(".json"):
            continue
        eng = fname[:-5]
        res = json.load(open(os.path.join(P["results"], fname)))
        N = S = D = I = 0
        by_age, secs = {}, []
        canto_ref = canto_hit = mand_extra = 0
        scored = [m for m in meta if m["id"] in res]  # a --limit run has only the first N clips
        for m in scored:
            r = res[m["id"]]
            ref, hyp = refs[m["id"]], norm(r["text"], cc)
            s, d, i = sdi(ref, hyp)
            N, S, D, I = N + len(ref), S + s, D + d, I + i
            g = by_age.setdefault(m["age"], [0, 0, 0])
            g[0] += s + d + i
            g[1] += len(ref)
            g[2] += 1
            secs.append(r["sec"])
            for ch in set(ref) & set(CANTONESE):
                canto_ref += ref.count(ch)
                canto_hit += min(ref.count(ch), hyp.count(ch))
            for ch in MANDARIN:
                mand_extra += max(0, hyp.count(ch) - ref.count(ch))
        silent = {p["id"]: res[p["id"]]["text"] for p in probes if p["kind"] != "partial" and p["id"] in res}
        invented = total = 0
        for p in probes:
            if p["kind"] == "partial" and p["id"] in res:
                h = norm(res[p["id"]]["text"], cc)
                total += len(h)
                invented += sum(1 for ch in h if ch not in refs[p["src"]])
        secs.sort()
        report[eng] = {
            "clips": len(scored),
            "CER": round(100 * (S + D + I) / N, 2),
            "substitutions": round(100 * S / N, 2), "deletions": round(100 * D / N, 2), "insertions": round(100 * I / N, 2),
            "CER_by_age": {k: {"CER": round(100 * v[0] / v[1], 2), "clips": v[2]} for k, v in sorted(by_age.items())},
            "cantonese_char_retention_pct": round(100 * canto_hit / max(canto_ref, 1), 1),
            "extra_standard_chinese_chars": mand_extra,
            "sequential_sec_median": secs[len(secs) // 2],
            "silence_noise_with_text": {k: v for k, v in silent.items() if norm(v, cc)},
            "partial_clip_invented_char_pct": round(100 * invented / max(total, 1), 1),
        }
        r = report[eng]
        print(f"{eng} ({r['clips']} clips): CER {r['CER']}% (S {r['substitutions']} D {r['deletions']} I {r['insertions']})  "
              f"Cantonese kept {r['cantonese_char_retention_pct']}%  silence/noise with text "
              f"{len(r['silence_noise_with_text'])}/{len(silent)}  partial invented {r['partial_clip_invented_char_pct']}%")
    json.dump(report, open(os.path.join(a.data, "report.json"), "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- performance
def cmd_perf(a):
    P = paths(a.data)
    meta = json.load(open(P["meta"]))[: a.n]
    files = [os.path.join(P["clips"], m["id"] + ".webm") for m in meta]
    audio_s = sum(m["dur"] for m in meta)
    if a.engine == "qwen":
        fn = qwen_fn(a)
        info = {"engine": "qwen", "url": a.url}
    else:
        t = time.perf_counter()
        model = load_whisper(a, a.workers)
        info = {"engine": "whisper", "compute_type": a.compute_type, "config": a.whisper_config,
                "workers": a.workers, "load_s": round(time.perf_counter() - t, 1)}
        fn = whisper_fn(model, a.whisper_config)
    fn(files[0])  # warm-up
    runs = []
    for conc in a.concurrency:
        lat = []

        def one(p):
            t = time.perf_counter()
            fn(p)
            lat.append(time.perf_counter() - t)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            list(ex.map(one, files))
        wall = time.perf_counter() - t0
        lat.sort()
        r = {"concurrency": conc, "clips": len(files), "wall_s": round(wall, 2),
             "clips_per_s": round(len(files) / wall, 2), "p50_s": round(lat[len(lat) // 2], 3),
             "p95_s": round(lat[int(len(lat) * .95)], 3), "rtf": round(wall / audio_s, 3)}
        runs.append(r)
        print(json.dumps(r), flush=True)
    json.dump({**info, "audio_s": round(audio_s, 1), "runs": runs},
              open(os.path.join(a.data, f"perf_{a.tag}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", required=True, help="folder holding content.txt and wav.tar.gz")
        p.add_argument("--limit", type=int, default=0, help="only the first N clips, no probes (0 = all)")

    def whisper_args(p):
        p.add_argument("--model", default="Systran/faster-whisper-large-v3", help="model name or local CTranslate2 folder")
        p.add_argument("--device", default="cuda", help="cuda or cpu")
        p.add_argument("--compute-type", default="float16",
                       help="GPU: float16, int8_float16; CPU: int8, float32")
        p.add_argument("--threads", type=int, default=8, help="CPU threads")

    def qwen_args(p):
        p.add_argument("--url", default="http://127.0.0.1:8000", help="OpenAI-compatible server base URL")
        p.add_argument("--served-model", default="qwen3-asr", help="model name the server expects")
        p.add_argument("--wav", action="store_true", help="send 16 kHz WAV (needed for llama.cpp's llama-server)")

    p = sub.add_parser("prepare"); common(p)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("whisper"); common(p); whisper_args(p)
    p.add_argument("--config", nargs="+", default=list(WHISPER_CONFIGS), choices=list(WHISPER_CONFIGS))
    p.set_defaults(fn=cmd_whisper)

    p = sub.add_parser("openai"); common(p)
    p.add_argument("--openai-model", default="large-v3", help="model name or path to large-v3.pt")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--fp16", action="store_true", help="FP16 decoding (GPU only); default FP32")
    p.add_argument("--threads", type=int, default=8, help="CPU threads")
    p.set_defaults(fn=cmd_openai)

    p = sub.add_parser("qwen-local"); common(p)
    p.add_argument("--hf-model", default="Qwen/Qwen3-ASR-1.7B-hf", help="Transformers-format checkpoint")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--dtype", default="bfloat16", help="bfloat16 or float32")
    p.add_argument("--threads", type=int, default=8, help="CPU threads")
    p.set_defaults(fn=cmd_qwen_local)

    p = sub.add_parser("qwen"); common(p); qwen_args(p)
    p.add_argument("--name", default="qwen3_asr", help="result file name")
    p.set_defaults(fn=cmd_qwen)

    p = sub.add_parser("score"); common(p)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("perf"); common(p); whisper_args(p); qwen_args(p)
    p.add_argument("--engine", choices=["qwen", "whisper"], required=True)
    p.add_argument("--whisper-config", default="beam5_vad", choices=list(WHISPER_CONFIGS))
    p.add_argument("--workers", type=int, default=1, help="faster-whisper num_workers")
    p.add_argument("--n", type=int, default=120, help="clips per concurrency level")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--tag", default="run", help="perf_<tag>.json")
    p.set_defaults(fn=cmd_perf)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
