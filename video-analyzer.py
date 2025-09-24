import os
import subprocess
import base64
import io
import tempfile
import shutil
from typing import List, Tuple, Optional, Union, BinaryIO
from contextvars import ContextVar

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("未安裝 openai 套件，請先執行: pip install openai")

# 嘗試載入 OpenCV（用於擷取影格）
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# 嘗試載入 PIL（影格縮放或 JPEG 編碼備援）
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# 嘗試載入 MoviePy 的 VideoFileClip（兩種匯入寫法）
MPVideoFileClip = None
HAS_MOVIEPY = False
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip as MPVideoFileClip
    HAS_MOVIEPY = True
except Exception:
    try:
        from moviepy.editor import VideoFileClip as MPVideoFileClip
        HAS_MOVIEPY = True
    except Exception:
        HAS_MOVIEPY = False
        MPVideoFileClip = None


# -------------------------------
# 日誌收集（同時 print 與緩存，供外部取用）
# -------------------------------
_log_list_var: ContextVar[Optional[list]] = ContextVar("_log_list", default=None)

def start_log_capture():
    _log_list_var.set([])

def get_captured_logs() -> List[str]:
    logs = _log_list_var.get()
    return list(logs) if logs is not None else []

def end_log_capture():
    _log_list_var.set(None)

def print_info(msg: str):
    print(f"[INFO] {msg}")
    logs = _log_list_var.get()
    if logs is not None:
        logs.append(msg)


# -------------------------------
# 工具與環境偵測
# -------------------------------
def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)

def has_ffmpeg() -> bool:
    return which("ffmpeg") is not None

def has_ffprobe() -> bool:
    return which("ffprobe") is not None

def ensure_file_exists(path: str) -> bool:
    return os.path.isfile(path)

def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def encode_image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# -------------------------------
# 暫存檔處理（將檔案物件或 bytes 落地一次）
# -------------------------------
def _infer_suffix_from_filename(filename: Optional[str]) -> str:
    if not filename or not isinstance(filename, str):
        return ".mp4"
    base = os.path.basename(filename)
    _, ext = os.path.splitext(base)
    return ext if ext else ".mp4"

def _materialize_to_tempfile(
    src: Union[bytes, bytearray, BinaryIO],
    filename: Optional[str] = None
) -> str:
    """
    將輸入資料寫入暫存檔，回傳暫存檔路徑。呼叫者負責刪除。
    - src 可為 bytes/bytearray 或 file-like（需為二進位模式）
    - filename 僅用於推斷副檔名，內容仍全數從 src 取得
    """
    suffix = _infer_suffix_from_filename(filename)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    try:
        if isinstance(src, (bytes, bytearray)):
            tmp.write(src)
        else:
            # file-like：將指標移到開頭再拷貝
            try:
                src.seek(0)
            except Exception:
                pass
            shutil.copyfileobj(src, tmp)
    finally:
        tmp.flush()
        tmp.close()
    return tmp_path


# -------------------------------
# 音訊偵測與「記憶體」萃取
# -------------------------------
def detect_audio_with_ffprobe(video_path: str) -> bool:
    if not has_ffprobe():
        return False
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        text = out.decode("utf-8").strip()
        return len(text) > 0
    except Exception:
        return False

def detect_audio_with_moviepy(video_path: str) -> bool:
    if not HAS_MOVIEPY:
        return False
    try:
        clip = MPVideoFileClip(video_path)
        has_audio = (clip.audio is not None)
        clip.close()
        return has_audio
    except Exception:
        return False

def extract_audio_bytes_with_ffmpeg(video_path: str) -> Optional[bytes]:
    """
    使用 ffmpeg 直接將音訊輸出為 WAV/PCM 16kHz mono 至 stdout，回傳 bytes（不落地）。
    """
    if not has_ffmpeg():
        return None
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "pipe:1"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out if out and len(out) > 0 else None
    except Exception:
        return None

def detect_and_extract_audio_in_memory(video_path: str) -> Tuple[bool, Optional[bytes]]:
    """
    回傳: (audio_present, wav_bytes or None)
    - 若無音訊或無法確認 → (False, None)
    - 若有音訊但萃取失敗 → (True, None)
    - 若有音訊且成功萃取 → (True, bytes)
    """
    print_info("檢查影片是否包含音訊")
    audio_exists = False

    if detect_audio_with_ffprobe(video_path):
        audio_exists = True
    else:
        if detect_audio_with_moviepy(video_path):
            audio_exists = True

    if not audio_exists:
        print_info("未偵測到音訊或無法確認音訊")
        return False, None

    print_info("偵測到音訊，將以記憶體方式萃取音訊（WAV/PCM 16kHz mono）")
    wav_bytes = extract_audio_bytes_with_ffmpeg(video_path)
    if wav_bytes is not None:
        print_info("音訊萃取成功（記憶體）")
        return True, wav_bytes

    print_info("音訊萃取失敗（記憶體）")
    return True, None


# -------------------------------
# 影片時長與影格切割規則
# -------------------------------
def get_video_duration_seconds(video_path: str) -> Optional[float]:
    # 優先 ffprobe
    if has_ffprobe():
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            dur = float(out.decode("utf-8").strip())
            if dur > 0:
                return dur
        except Exception:
            pass

    # 回退 OpenCV
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            try:
                if fps and frame_count and fps > 0:
                    return float(frame_count / fps)
            except Exception:
                pass

    # 回退 MoviePy
    if HAS_MOVIEPY:
        try:
            clip = MPVideoFileClip(video_path)
            dur = float(clip.duration)
            clip.close()
            return dur
        except Exception:
            pass

    return None

def decide_frame_count_by_duration(duration_sec: float) -> int:
    if duration_sec < 60:
        return 4
    elif 60 <= duration_sec < 180:
        return 7
    elif 180 <= duration_sec < 300:
        return 11
    else:
        return 15

def linspace(start: float, end: float, num: int, include_end: bool = False) -> List[float]:
    if num <= 1:
        return [start]
    span = (end - start)
    if span <= 0:
        return [start for _ in range(num)]
    if include_end:
        step = span / (num - 1)
        return [start + step * i for i in range(num)]
    else:
        step = span / num
        return [start + step * i for i in range(num)]


# -------------------------------
# 影格擷取（純記憶體）
# -------------------------------
def encode_rgb_to_jpeg_bytes(rgb_array, max_side: int = 1280, quality: int = 85) -> Optional[bytes]:
    try:
        import numpy as np  # 保險導入
        if HAS_PIL:
            img = Image.fromarray(rgb_array)  # MoviePy 取得為 RGB
            w, h = img.size
            max_dim = max(w, h)
            if max_dim > max_side:
                scale = max_side / float(max_dim)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
        elif HAS_CV2:
            h, w = rgb_array.shape[:2]
            max_dim = max(w, h)
            bgr = rgb_array[:, :, ::-1]
            if max_dim > max_side:
                scale = max_side / float(max_dim)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                return enc.tobytes()
            return None
        else:
            return None
    except Exception:
        return None

def extract_frames_bytes_opencv(video_path: str, timestamps: List[float], max_side: int = 1280, quality: int = 85) -> List[Optional[bytes]]:
    results: List[Optional[bytes]] = [None] * len(timestamps)
    if not HAS_CV2:
        return results
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return results
    try:
        for idx, t in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t * 1000.0))
            ok, frame = cap.read()
            if (not ok) or (frame is None):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (t - 0.1) * 1000.0))
                ok, frame = cap.read()
            if (not ok) or (frame is None):
                continue
            rgb = frame[:, :, ::-1]
            jpeg_bytes = encode_rgb_to_jpeg_bytes(rgb, max_side=max_side, quality=quality)
            results[idx] = jpeg_bytes
    finally:
        cap.release()
    return results

def extract_frames_bytes_moviepy(video_path: str, timestamps: List[float], max_side: int = 1280, quality: int = 85) -> List[Optional[bytes]]:
    results: List[Optional[bytes]] = [None] * len(timestamps)
    if not HAS_MOVIEPY:
        return results
    try:
        clip = MPVideoFileClip(video_path)
    except Exception:
        return results
    try:
        for idx, t in enumerate(timestamps):
            try:
                ts = max(0.0, min(t, max(0.0, clip.duration - 1e-3)))
                frame_rgb = clip.get_frame(ts)  # numpy array, RGB
                jpeg_bytes = encode_rgb_to_jpeg_bytes(frame_rgb, max_side=max_side, quality=quality)
                results[idx] = jpeg_bytes
            except Exception:
                results[idx] = None
    finally:
        try:
            clip.close()
        except Exception:
            pass
    return results

def extract_frames_bytes_ffmpeg_pipe(video_path: str, timestamps: List[float], max_side: int = 1280, quality_param: int = 2) -> List[Optional[bytes]]:
    results: List[Optional[bytes]] = [None] * len(timestamps)
    if not has_ffmpeg():
        return results

    scale_filter = f"scale=if(gt(iw,ih),min(iw,{max_side}),-2):if(gt(iw,ih),-2,min(ih,{max_side}))"

    for idx, t in enumerate(timestamps):
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-ss", f"{t:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", scale_filter,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", str(quality_param),  # 2~5 高品質
            "pipe:1"
        ]
        try:
            img_bytes = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            if img_bytes and len(img_bytes) > 0:
                results[idx] = img_bytes
        except Exception:
            results[idx] = None
    return results

def gather_frames_in_memory(video_path: str, timestamps: List[float], expected_count: int) -> List[Tuple[int, float, bytes]]:
    print_info("影格將以記憶體方式處理，不會在本地建立暫存影像檔")
    n = len(timestamps)
    frames: List[Optional[bytes]] = [None] * n

    if HAS_CV2:
        ocv = extract_frames_bytes_opencv(video_path, timestamps, max_side=1280, quality=85)
        for i in range(n):
            if ocv[i] is not None:
                frames[i] = ocv[i]

    if any(f is None for f in frames):
        ff = extract_frames_bytes_ffmpeg_pipe(video_path, timestamps, max_side=1280, quality_param=2)
        for i in range(n):
            if frames[i] is None and ff[i] is not None:
                frames[i] = ff[i]

    if any(f is None for f in frames):
        mp = extract_frames_bytes_moviepy(video_path, timestamps, max_side=1280, quality=85)
        for i in range(n):
            if frames[i] is None and mp[i] is not None:
                frames[i] = mp[i]

    out: List[Tuple[int, float, bytes]] = []
    for i in range(n):
        if frames[i] is not None:
            out.append((i + 1, timestamps[i], frames[i]))

    if len(out) == 0:
        raise RuntimeError("無法擷取任何影格（記憶體流程）")

    out.sort(key=lambda x: x[0])
    return out


# -------------------------------
# OpenAI 請求封裝
# -------------------------------
def chat_vision_analyze_image_from_bytes(client: OpenAI, model: str, prompt_text: str, image_bytes: bytes, temperature: float = 0.4) -> str:
    data_url = encode_image_bytes_to_data_url(image_bytes, mime="image/jpeg")
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                }
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI 影像分析失敗: {e}")

def chat_text_only(client: OpenAI, model: str, prompt: str, temperature: float = 0.5) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI 文字處理失敗: {e}")

def transcribe_audio_bytes(client: OpenAI, wav_bytes: bytes) -> str:
    """
    以記憶體 bytes 上傳音訊做轉錄。優先 gpt-4o-transcribe，失敗回退 whisper-1。
    """
    buf = io.BytesIO(wav_bytes)
    buf.name = "audio.wav"
    buf.seek(0)
    try:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=buf
        )
        return tr.text.strip()
    except Exception:
        try:
            buf.seek(0)
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=buf
            )
            return tr.text.strip()
        except Exception as e2:
            raise RuntimeError(f"OpenAI 音訊轉文字失敗: {e2}")


# -------------------------------
# 副程式：圖像分析（記憶體影格）
# -------------------------------
def image_analysis(video_path: str, client: OpenAI, model: str = "gpt-4o") -> str:
    print_info("開始圖像分析（記憶體影格）")
    duration = get_video_duration_seconds(video_path)
    if duration is None:
        raise RuntimeError("無法取得影片時長，圖像分析中止")

    n_frames = decide_frame_count_by_duration(duration)
    print_info(f"影片時長約 {seconds_to_hms(duration)}，計畫擷取 {n_frames} 張影格（記憶體處理）")

    timestamps = linspace(0.0, max(0.001, duration), n_frames, include_end=False)
    inmem_frames = gather_frames_in_memory(video_path, timestamps, expected_count=n_frames)

    per_frame_descriptions = []
    for idx, (frame_index, t, jpeg_bytes) in enumerate(inmem_frames, start=1):
        print_info(f"分析第 {idx}/{n_frames} 張影格")
        if idx == 1:
            prompt = (
                "你是一位嚴謹的影片畫面分析助手。請針對下方的單張影格，描述："
                "可見的場景/環境、人物或物件、動作與互動、可讀取的文字、情緒氛圍、"
                "可能的故事線索或情節脈絡。避免臆測與主觀判斷，請以可觀察到的細節為主。"
            )
        else:
            prev_summary = per_frame_descriptions[-1]["desc"]
            prompt = (
                "以下為第 n-1 張影格的簡述，請先理解上下文，再描述接下來的這張影格；"
                "如果與前一張有變化，請指出關鍵變化點：\n\n"
                f"[前一張影格摘要]\n{prev_summary}\n\n"
                "現在請描述本張影格，聚焦於畫面可見的元素、行動與脈絡。"
                "避免臆測與主觀判斷。"
            )

        desc = chat_vision_analyze_image_from_bytes(client, model, prompt, jpeg_bytes, temperature=0.4)
        per_frame_descriptions.append({
            "index": idx,
            "timestamp": t,
            "desc": desc
        })

    print_info("彙整所有影格的觀察，產生整體描述")
    bullets = []
    for item in per_frame_descriptions:
        ts = seconds_to_hms(item["timestamp"]) if item["timestamp"] is not None else "N/A"
        bullets.append(f"# 影格 {item['index']} @ {ts}\n{item['desc']}")
    joined = "\n\n".join(bullets)

    final_prompt = (
        "以下是針對多張等距取樣的影格所做的觀察，請你根據它們，"
        "產出對整部影片的詳細描述。請包含：\n"
        "- 影片主題與整體脈絡\n"
        "- 主要場景、人物/物件與其關係\n"
        "- 重要事件/轉折與合理的時間脈絡\n"
        "- 可辨識的文字資訊（若有）及其可能含義\n"
        "- 畫面風格/情緒與觀感\n"
        "請避免臆測無憑據的細節，僅以影格可支持的推論為主。以下是影格觀察：\n\n"
        f"{joined}\n\n"
        "請輸出一段條理清晰、可獨立閱讀的完整描述。"
    )
    final_description = chat_text_only(client, model, final_prompt, temperature=0.4)
    print_info("圖像分析完成")
    return final_description


# -------------------------------
# 副程式：音訊分析（記憶體轉文字）
# -------------------------------
def audio_analysis_from_memory(wav_bytes: bytes, client: OpenAI) -> str:
    print_info("開始音訊分析（轉文字，記憶體）")
    if not isinstance(wav_bytes, (bytes, bytearray)) or len(wav_bytes) == 0:
        raise RuntimeError("音訊資料無效，無法轉文字")
    text = transcribe_audio_bytes(client, wav_bytes)
    print_info("音訊分析完成（轉文字，記憶體）")
    return text


# -------------------------------
# 副程式：結合圖像與音訊
# -------------------------------
def combine_image_and_audio(image_desc: str, audio_text: str, client: OpenAI, model: str = "gpt-4o") -> str:
    print_info("開始結合圖像與音訊敘述")
    prompt = (
        "我將提供兩段文字：一段是基於影格的影片畫面分析摘要（不含音訊），"
        "另一段是影片音訊的完整轉錄。請你融合兩者資訊，產生一段對整支影片的完整描述，"
        "要求包含：\n"
        "- 全片主題、結構與時間脈絡\n"
        "- 圖像與音訊共同指向的重點內容\n"
        "- 關鍵事件、角色/物件與意義\n"
        "- 可辨識文字（若音訊或畫面有）、地點、人名或特定名詞\n"
        "- 若畫面與音訊有差異或補充，請說明其對理解的影響\n"
        "請避免過度臆測，並以中立、清楚的語氣總結。\n\n"
        f"[圖像分析摘要]\n{image_desc}\n\n"
        f"[音訊轉錄全文]\n{audio_text}\n\n"
        "請輸出最終的完整描述。"
    )
    combined = chat_text_only(client, model, prompt, temperature=0.4)
    print_info("結合圖像與音訊完成")
    return combined


# -------------------------------
# 內部主流程（以檔案路徑處理，供包裝函式呼叫）
# -------------------------------
def _analyze_video_at_path(video_path: str) -> str:
    print_info("啟動主程式")
    try:
        # 確認 API Key 透過環境變數提供
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("環境變數 OPENAI_API_KEY 未設置")

        if not ensure_file_exists(video_path):
            print_info("暫存影片檔不存在或不可讀")
            return "分析失敗：暫存影片檔無效或不可讀"

        # OpenAI() 會自動讀取 OPENAI_API_KEY
        client = OpenAI()

        audio_present, wav_bytes = detect_and_extract_audio_in_memory(video_path)

        if not audio_present or wav_bytes is None:
            if not audio_present:
                print_info("流程：未偵測到音訊 → 僅執行圖像分析")
            else:
                print_info("流程：偵測到音訊但記憶體萃取失敗 → 僅執行圖像分析")
            image_summary = image_analysis(video_path, client, model="gpt-4o")
            print_info("主程式完成（僅圖像分析）")
            return image_summary

        print_info("流程：圖像分析 與 音訊分析（音訊走記憶體）")
        image_summary = image_analysis(video_path, client, model="gpt-4o")
        audio_text = audio_analysis_from_memory(wav_bytes, client)
        final_text = combine_image_and_audio(image_summary, audio_text, client, model="gpt-4o")
        print_info("主程式完成（圖像 + 音訊已融合）")
        return final_text

    except Exception as e:
        print_info(f"主程式發生錯誤：{e}")
        return f"分析失敗：{e}"


# -------------------------------
# 對外 API：直接傳檔案或 bytes
# -------------------------------
def analyze_video_file(file_obj: BinaryIO, filename: Optional[str] = None) -> str:
    """
    以檔案物件進行分析（不要求外部提供路徑）。
    - file_obj 需為二進位模式開啟（'rb'）
    - filename 僅用於推斷副檔名（如 .mp4），若不提供預設為 .mp4
    """
    tmp_path = _materialize_to_tempfile(file_obj, filename=filename)
    try:
        return _analyze_video_at_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def analyze_video_bytes(video_bytes: Union[bytes, bytearray], filename: Optional[str] = "input.mp4") -> str:
    """
    以 bytes 內容進行分析。可指定 filename 來推斷副檔名。
    """
    tmp_path = _materialize_to_tempfile(video_bytes, filename=filename)
    try:
        return _analyze_video_at_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass