
from pathlib import Path
from collections import deque
import time
import argparse
import json

from semantic_client import request_willingness, SemanticServiceError

import numpy as np
import sounddevice as sd



# ========== 1. 配置 ==========

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "SenseVoiceSmall"

SAMPLE_RATE = 16000      # 麦克风采样率
BLOCK_SECONDS = 0.1      # 每次读取 0.1 秒

VOLUME_THRESHOLD = 0.05 # 判断说话的音量阈值，正在测试
SILENCE_SECONDS = 0.8    # 停顿多久认为一句话结束
MIN_SPEECH_SECONDS = 0.3 # 过滤太短的声音
MAX_RECORD_SECONDS = 10  # 每句话最长录制时间
PRE_ROLL_SECONDS = 0.3   # 保留说话前的一点声音

LANGUAGE = "auto"        # 自动识别普通话/粤语等


# ========== 2. 加载 SenseVoiceSmall ==========

def load_model():
    from funasr import AutoModel
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_DIR}")
    print("正在加载 SenseVoiceSmall...")
    loaded = AutoModel(model=str(MODEL_DIR), device="cpu", disable_update=True)
    print("模型加载完成！")
    return loaded


# ========== 3. 从麦克风录制一句话 ==========

def record_sentence():

    block_size = int(SAMPLE_RATE * BLOCK_SECONDS)
    pre_roll_count = max(
        1, int(PRE_ROLL_SECONDS / BLOCK_SECONDS)
    )

    recent_audio = deque(maxlen=pre_roll_count)#这一步是声明一个deque对象，最大长度为prepre_roll_count。
    recorded_audio = []

    speaking = False
    speech_duration = 0.0
    silence_duration = 0.0
    total_duration = 0.0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_size
    ) as stream:

        while True:

            data, overflowed = stream.read(block_size)
            audio = data[:, 0].copy()

            if overflowed:
                print("警告：录音缓冲区发生溢出")

            # 均方根音量
            volume = float(
                np.sqrt(np.mean(audio ** 2))
            )

            is_loud = volume >= VOLUME_THRESHOLD

#             print(
#                     f"音量={volume:.5f}, "
#                     f"阈值={VOLUME_THRESHOLD:.5f}, "
#                     f"is_loud={is_loud}, "
#                     f"静音时长={silence_duration:.1f}s"
# )

            if not speaking:

                recent_audio.append(audio)

                if is_loud:
                    speaking = True
                    print("🎤 检测到说话...")

                    recorded_audio.extend(recent_audio)#这个是extend，recent_audio是一个deque对象，extend方法会将recent_audio中的所有元素添加到recorded_audio中。
                    speech_duration += BLOCK_SECONDS
                    total_duration = len(
                        recorded_audio
                    ) * BLOCK_SECONDS

                continue

            recorded_audio.append(audio)#讲话之后就直接将音频数据添加到recorded_audio中，而不是recent_audio中。
            total_duration += BLOCK_SECONDS

            if is_loud:
                speech_duration += BLOCK_SECONDS
                silence_duration = 0.0
            else:
                silence_duration += BLOCK_SECONDS

            if silence_duration >= SILENCE_SECONDS:
                print("🔇 检测到停顿")
                break

            if total_duration >= MAX_RECORD_SECONDS:
                print("达到最长录音时间,最小检测阈值")
                break

    if speech_duration < MIN_SPEECH_SECONDS:
        print("声音太短，忽略")
        return None

    # 合并成一维 NumPy 音频数组
    return np.concatenate(recorded_audio).astype(
        np.float32
    )


# ========== 4. 识别语音 ==========

def recognize(audio, model):

    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    start = time.perf_counter()

    result = model.generate(
        input=audio,
        fs=SAMPLE_RATE,
        language=LANGUAGE,
        use_itn=True
    )

    elapsed = time.perf_counter() - start

    if not result:
        print("没有识别结果")
        return

    raw_text = result[0].get("text", "")

    clean_text = rich_transcription_postprocess(
        raw_text
    )

    print("原始结果：", raw_text)
    print("识别文字：", clean_text)
    print(f"识别耗时：{elapsed:.2f} 秒")
    return clean_text


# ========== 5. 发送整句文字并显示语义结果 ==========

def send_text(text, args):
    start = time.perf_counter()
    try:
        result = request_willingness(text, server_url=args.server, system_prompt=args.prompt)
    except SemanticServiceError as error:
        print(f"语义请求失败：{error}；本句未获得判断，不计为 no 或 unclear。")
        return False
    print("语义识别：", result['result'])
    print("判断原因：", result['reason'])
    print("完整结果：", json.dumps(result, ensure_ascii=False))
    print(f"语义请求耗时：{time.perf_counter() - start:.2f} 秒")
    return True


def main():
    parser = argparse.ArgumentParser(description="逐句 ASR → 服务器参与意愿判断")
    parser.add_argument('--server', help='语义服务器地址，默认 SEMANTIC_SERVER_URL 或 http://127.0.0.1:8765')
    parser.add_argument('--prompt', help='本轮人工提示，省略时使用服务器的 task3_willingness 提示')
    parser.add_argument('--text', help='直接发送文字进行联调，不加载 ASR、不打开麦克风')
    parser.add_argument('--asr-only', action='store_true', help='仅做原有 ASR，不请求语义服务器')
    args = parser.parse_args()
    if args.text is not None:
        if args.asr_only:
            parser.error('--text 联调不能与 --asr-only 一起使用')
        return 0 if send_text(args.text, args) else 1
    model = load_model()
    print("\n开始麦克风语音识别；按 Ctrl+C 退出程序")
    if not args.asr_only:
        print("本轮任务：是否愿意参加同伴游戏；说完一句后等待服务器返回判断。")
    try:
        while True:
            print("\n正在监听麦克风，请说话...")
            audio = record_sentence()
            if audio is not None:
                text = recognize(audio, model)
                if text and any(c.isalnum() for c in text) and not args.asr_only:
                    send_text(text, args)
    except KeyboardInterrupt:
        print("\n已停止语音识别")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
