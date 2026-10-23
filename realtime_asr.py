"""SenseVoiceSmall microphone transcription (bounded pseudo streaming)."""
from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
RATE = 16000
BLOCK = 1600  # 100 ms


class Segmenter:
    """Energy endpoint detector; feed equal-sized float32 mono blocks."""

    def __init__(self, sample_rate=RATE, threshold=.008, silence_seconds=.7,
                 max_seconds=15., min_seconds=.2, preroll_seconds=.2):
        self.threshold = threshold
        self.silence_limit = round(sample_rate * silence_seconds)
        self.max_samples = round(sample_rate * max_seconds)
        self.min_samples = round(sample_rate * min_seconds)
        self.preroll_limit = round(sample_rate * preroll_seconds)
        self.pre = deque()
        self.pre_samples = 0
        self.parts = []
        self.samples = self.speech_samples = self.quiet_samples = 0

    def feed(self, block):
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        voiced = float(np.sqrt(np.mean(block * block))) >= self.threshold
        if not self.parts:
            if not voiced:
                self.pre.append(block.copy())
                self.pre_samples += len(block)
                while self.pre and self.pre_samples > self.preroll_limit:
                    self.pre_samples -= len(self.pre.popleft())
                return None
            self.parts = list(self.pre)
            self.samples = self.pre_samples
            self.pre.clear()
            self.pre_samples = 0
        self.parts.append(block.copy())
        self.samples += len(block)
        self.speech_samples += len(block) if voiced else 0
        self.quiet_samples = 0 if voiced else self.quiet_samples + len(block)
        if self.quiet_samples >= self.silence_limit or self.samples >= self.max_samples:
            return self.flush()
        return None

    def snapshot(self):
        if self.speech_samples < self.min_samples:
            return None
        return np.concatenate(self.parts)

    def flush(self):
        result = self.snapshot()
        self.parts = []
        self.samples = self.speech_samples = self.quiet_samples = 0
        return result


def configure_cache():
    # Keep downloads, compiled caches and temporary files inside this project.
    paths = {
        'MODELSCOPE_CACHE': '.cache/modelscope', 'HF_HOME': '.cache/huggingface',
        'TORCH_HOME': '.cache/torch', 'XDG_CACHE_HOME': '.cache',
        'NUMBA_CACHE_DIR': '.cache/numba', 'MPLCONFIGDIR': '.cache/matplotlib',
        'TEMP': '.cache/codex_temp_runtime', 'TMP': '.cache/codex_temp_runtime',
    }
    for name, relative in paths.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_model(args):
    configure_cache()
    print('正在加载 SenseVoiceSmall，首次使用需要下载模型……', flush=True)
    import torch
    from funasr import AutoModel
    from modelscope import snapshot_download

    model_dir = Path(args.model).expanduser().resolve() if args.model else ROOT / 'models' / 'SenseVoiceSmall'
    if not args.model:
        # ModelScope manages checksums and resuming; a marker is written only on success.
        marker = model_dir / '.asr_download_complete'
        if not marker.exists():
            snapshot_download('iic/SenseVoiceSmall', local_dir=str(model_dir),
                              ignore_patterns=['*.onnx', '*.bin', '*.safetensors'])
            marker.touch()
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.compute == 'auto' else args.compute
    print(f'推理设备：{device}；模型：{model_dir}', flush=True)
    started = time.perf_counter()
    model = AutoModel(model=str(model_dir), device=device, ncpu=args.threads,
                      trust_remote_code=False, disable_update=True, disable_pbar=True,
                      disable_log=True, check_latest=False)
    print(f'模型已就绪，加载耗时 {time.perf_counter() - started:.1f} 秒。', flush=True)
    return model


def recognize(model, audio, language):
    result = model.generate(input=np.ascontiguousarray(audio, dtype=np.float32),
                            cache={}, language=language, use_itn=True, fs=RATE,
                            batch_size=1, disable_pbar=True)
    raw = ''.join(item.get('text', '') for item in result)
    text = re.sub(r'<\|[^|]*\|>', '', raw).strip()
    return text if any(c.isalnum() for c in text) else ''


class Terminal:
    def __init__(self):
        self.width = 0

    def show(self, text, final=False):
        line = ('[确认] ' if final else '[临时] ') + text
        # Chinese characters can occupy two columns; pad the previous line completely.
        width = sum(2 if ord(c) > 127 else 1 for c in line)
        print('\r' + line + ' ' * max(0, self.width - width), end='\n' if final else '', flush=True)
        self.width = 0 if final else width

    def clear(self):
        if self.width:
            print('\r' + ' ' * self.width + '\r', end='', flush=True)
            self.width = 0


def run_file(model, args):
    import soundfile as sf
    from scipy.signal import resample_poly
    data, rate = sf.read(args.file, dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        data = resample_poly(data, RATE // divisor, rate // divisor).astype(np.float32)
    if len(data) == 0:
        raise ValueError('音频文件为空。')
    for start in range(0, len(data), RATE * 20):
        begun = time.perf_counter()
        text = recognize(model, data[start:start + RATE * 20], args.language)
        print(f'[确认] {text}\n[推理耗时] {time.perf_counter() - begun:.2f} 秒', flush=True)


def run_microphone(model, args):
    import sounddevice as sd
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    sd.check_input_settings(device=device, channels=1, dtype='float32', samplerate=RATE)
    frames = queue.Queue(maxsize=200)  # 20 s, fail explicitly rather than lose speech.
    overflow = threading.Event()

    def callback(indata, frame_count, timing, status):
        if status:
            overflow.set()
        try:
            frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            overflow.set()

    segmenter = Segmenter(threshold=args.threshold, silence_seconds=args.silence,
                          max_seconds=args.max_seconds)
    terminal = Terminal()
    last_partial_samples = 0
    partial_step = int(args.partial_interval * RATE)
    stopped = False
    pending_final = None

    def consume(block, partial=True):
        nonlocal last_partial_samples, pending_final
        final = segmenter.feed(block)
        if not segmenter.parts:
            last_partial_samples = 0
        if final is not None:
            pending_final = final
            text = recognize(model, pending_final, args.language)
            pending_final = None
            if text:
                terminal.show(text, final=True)
            else:
                terminal.clear()
        elif partial and partial_step and frames.qsize() <= 2:
            if segmenter.samples - last_partial_samples >= partial_step:
                audio = segmenter.snapshot()
                if audio is not None:
                    text = recognize(model, audio, args.language)
                    last_partial_samples = segmenter.samples
                    if text:
                        terminal.show(text)

    print(f'开始监听：{sd.query_devices(device, "input")["name"]}', flush=True)
    print(f'停顿 {args.silence:g} 秒后确认；阈值 {args.threshold:g}；Ctrl+C 停止。', flush=True)
    print('请保持终端窗口宽一些；识别期间不保存录音。', flush=True)
    start = time.monotonic()
    with sd.InputStream(device=device, channels=1, samplerate=RATE, dtype='float32',
                        blocksize=BLOCK, callback=callback):
        try:
            while not args.duration or time.monotonic() - start < args.duration:
                if overflow.is_set():
                    raise RuntimeError('音频缓冲区溢出或设备丢帧，请加 --partial-interval 0 关闭临时识别后重试。')
                try:
                    consume(frames.get(timeout=.2))
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            stopped = True
    if overflow.is_set():
        raise RuntimeError('音频缓冲区溢出或设备丢帧，请加 --partial-interval 0 后重试。')
    terminal.clear()
    print('\n已停止采集，正在处理剩余语音……', flush=True)
    if pending_final is not None:
        text = recognize(model, pending_final, args.language)
        if text:
            terminal.show(text, final=True)
        pending_final = None
    while not frames.empty():
        consume(frames.get_nowait(), partial=False)
    tail = segmenter.flush()
    if tail is not None:
        text = recognize(model, tail, args.language)
        if text:
            terminal.show(text, final=True)
    terminal.clear()
    print('已退出。' if stopped else '定时采集验证结束。', flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='SenseVoiceSmall 麦克风实时分段转写（伪流式）')
    parser.add_argument('--list-devices', action='store_true', help='列出音频设备后退出')
    parser.add_argument('--device', help='麦克风设备编号或名称，默认使用系统默认输入')
    parser.add_argument('--compute', default='auto', help='auto / cpu / cuda:0')
    parser.add_argument('--model', help='已有的本地 SenseVoiceSmall 模型目录')
    parser.add_argument('--language', default='auto', choices=['auto', 'zh', 'en', 'yue', 'ja', 'ko'])
    parser.add_argument('--threshold', type=float, default=.008, help='语音能量阈值，默认 0.008')
    parser.add_argument('--silence', type=float, default=.7, help='句末静音秒数，默认 0.7')
    parser.add_argument('--max-seconds', type=float, default=15., help='单段最长秒数，默认 15，最大 25')
    parser.add_argument('--partial-interval', type=float, default=1.2, help='临时识别间隔秒数，0 为只输出整句')
    parser.add_argument('--threads', type=int, default=4, help='CPU 推理线程数')
    parser.add_argument('--duration', type=float, default=0, help='监听指定秒数后退出，0 为不限时')
    parser.add_argument('--file', help='识别本地音频文件，不开启麦克风，推荐 WAV')
    parser.add_argument('--download-only', action='store_true', help='下载并加载模型，验证就绪后退出')
    args = parser.parse_args()
    if not (0 < args.threshold < 1 and .1 <= args.silence <= 5 and
            1 <= args.max_seconds <= 25 and args.partial_interval >= 0 and
            args.threads >= 1 and args.duration >= 0):
        parser.error('阈值应在 (0,1)，静音在 [0.1,5]，分段在 [1,25]；其余参数不能为负，线程数至少为 1。')
    try:
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
            return 0
        model = load_model(args)
        if args.download_only:
            return 0
        if args.file:
            run_file(model, args)
        else:
            run_microphone(model, args)
        return 0
    except KeyboardInterrupt:
        print('\n已中止。', flush=True)
        return 130
    except Exception as exc:
        print(f'\n运行失败：{exc}\n请参阅 README_ASR.md；设备问题可运行 --list-devices。', file=sys.stderr)
        if os.environ.get('ASR_DEBUG') == '1':
            raise
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
