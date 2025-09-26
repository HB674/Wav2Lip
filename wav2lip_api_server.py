# wav2lip_api_server_singleton.py
import os
import time
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Wav2Lip 레포 의존 ---
from models import Wav2Lip
import face_detection
import audio

app = FastAPI(title="Wav2Lip API (singleton, lazy-init, no-temp-dir)", version="1.1.0")

# ---- 경로 규약 ----
SHARED_DIR        = Path(os.getenv("SHARED_DIR", "/workspace/shared_data"))
APPLIO_OUT_DIR    = SHARED_DIR / "applio_output_queue"      # 오디오 소스
SADTALKER_OUT_DIR = SHARED_DIR / "sadtalker_output_queue"   # 얼굴(영상) 소스
WAV2LIP_OUT_DIR   = SHARED_DIR / "wav2lip_output_queue"     # 결과물
WARMUP_DIR        = SHARED_DIR / "warmup"

CHECKPOINT_PATH   = Path(os.getenv("CHECKPOINT_PATH", "/app/checkpoints/wav2lip_gan.pth"))

for d in [SHARED_DIR, APPLIO_OUT_DIR, SADTALKER_OUT_DIR, WAV2LIP_OUT_DIR, WARMUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 유틸
# ------------------------------
def _pick_latest(directory: Path, patterns: List[str]) -> Optional[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _ensure_wav(src_path: Path, dst_wav: Path) -> Path:
    """
    입력이 wav가 아니면 ffmpeg로 16k mono wav로 변환. dst_wav에 저장.
    dst_wav는 출력 디렉토리 내부(따로 temp 디렉토리 없음).
    """
    if src_path.suffix.lower() == ".wav":
        # 그대로 쓰되, 샘플레이트/채널이 다를 수 있으므로 표준화 강제
        cmd = f'ffmpeg -y -i "{src_path}" -ar 16000 -ac 1 "{dst_wav}"'
    else:
        cmd = f'ffmpeg -y -i "{src_path}" -ar 16000 -ac 1 "{dst_wav}"'
    rc = os.system(cmd)
    if rc != 0 or not dst_wav.exists():
        raise RuntimeError(f"ffmpeg 변환 실패: {cmd}")
    return dst_wav

def _read_video_frames(face_path: Path, resize_factor: int, crop: Tuple[int,int,int,int], rotate: bool) -> Tuple[List[np.ndarray], float]:
    """
    영상 파일(.mp4 등) -> 프레임 리스트, fps
    이미지(.jpg/.png 등) -> 단일 프레임 리스트, fps는 호출자가 override
    """
    if not face_path.exists():
        raise FileNotFoundError(f"face video not found: {face_path}")

    ext = face_path.suffix.lower()[1:]
    if ext in ['jpg','jpeg','png']:
        img = cv2.imread(str(face_path))
        if img is None:
            raise ValueError(f"이미지 로드 실패: {face_path}")
        if resize_factor > 1:
            img = cv2.resize(img, (img.shape[1]//resize_factor, img.shape[0]//resize_factor))
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        t, b, l, r = crop
        if r == -1: r = img.shape[1]
        if b == -1: b = img.shape[0]
        img = img[t:b, l:r]
        return [img], -1.0
    else:
        cap = cv2.VideoCapture(str(face_path))
        if not cap.isOpened():
            raise ValueError(f"비디오 오픈 실패: {face_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            t, b, l, r = crop
            if r == -1: r = frame.shape[1]
            if b == -1: b = frame.shape[0]
            frame = frame[t:b, l:r]
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("영상에서 프레임을 읽지 못했습니다.")
        return frames, float(fps)

def _get_smoothened_boxes(boxes: np.ndarray, T: int) -> np.ndarray:
    out = boxes.copy()
    for i in range(len(out)):
        if i + T > len(out):
            window = out[len(out) - T:]
        else:
            window = out[i:i+T]
        out[i] = np.mean(window, axis=0)
    return out

# ------------------------------
# 싱글톤 러너
# ------------------------------
class Wav2LipRunner:
    def __init__(self, checkpoint_path: Path, device: Optional[str] = None, img_size: int = 96):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self._lock = threading.Lock()  # 단일 GPU 동시성 제한

        # 모델 로드 (1회)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 얼굴 검출기 로드 (1회)
        self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                     flip_input=False, device=self.device)

    def _load_checkpoint(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        if self.device == 'cuda':
            return torch.load(str(checkpoint_path))
        return torch.load(str(checkpoint_path), map_location='cpu')

    def _load_model(self, checkpoint_path: Path):
        model = Wav2Lip()
        ckpt = self._load_checkpoint(checkpoint_path)
        state = ckpt["state_dict"]
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state)
        model = model.to(self.device)
        return model

    def _face_detect(self, frames: List[np.ndarray], pads: Tuple[int,int,int,int], nosmooth: bool,
                     box: Tuple[int,int,int,int], static: bool, face_det_batch_size: int,
                     debug_save_dir: Optional[Path] = None) -> List[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
        if box[0] != -1:
            y1, y2, x1, x2 = box
            results = [[f[y1:y2, x1:x2], (y1,y2,x1,x2)] for f in frames]
            return results

        bs = max(1, face_det_batch_size)
        predictions = []
        while True:
            try:
                candidate_frames = frames if not static else [frames[0]]
                for i in range(0, len(candidate_frames), bs):
                    batch = np.array(candidate_frames[i:i+bs])
                    preds = self.detector.get_detections_for_batch(batch)
                    predictions.extend(preds)
            except RuntimeError:
                if bs == 1:
                    raise RuntimeError('Face detection OOM on GPU. resize_factor를 키워보세요.')
                bs //= 2
                predictions.clear()
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        for idx, image in enumerate(frames if not static else [frames[0]]):
            rect = predictions[idx]
            if rect is None:
                if debug_save_dir:
                    (debug_save_dir / "faulty_frame.jpg").write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
                raise ValueError("얼굴 미검출. pads/resize_factor/crop/box를 조정하세요.")

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = _get_smoothened_boxes(boxes, T=5)

        out = []
        if static:
            (x1,y1,x2,y2) = boxes[0].astype(int)
            for image in frames:
                out.append([image[y1:y2, x1:x2], (y1,y2,x1,x2)])
        else:
            for image, (x1,y1,x2,y2) in zip(frames, boxes.astype(int)):
                out.append([image[y1:y2, x1:x2], (y1,y2,x1,x2)])

        return out

    def _datagen(self, frames: List[np.ndarray], mels: List[np.ndarray], face_crops, static: bool,
                 wav2lip_batch_size: int):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        for i, m in enumerate(mels):
            idx = 0 if static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_crops[idx]

            face = cv2.resize(face, (self.img_size, self.img_size))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= wav2lip_batch_size:
                ib = np.asarray(img_batch)
                mb = np.asarray(mel_batch)

                img_masked = ib.copy()
                img_masked[:, self.img_size//2:] = 0
                ib = np.concatenate((img_masked, ib), axis=3) / 255.
                mb = np.reshape(mb, [len(mb), mb.shape[1], mb.shape[2], 1])

                yield ib, mb, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            ib = np.asarray(img_batch)
            mb = np.asarray(mel_batch)
            img_masked = ib.copy()
            img_masked[:, self.img_size//2:] = 0
            ib = np.concatenate((img_masked, ib), axis=3) / 255.
            mb = np.reshape(mb, [len(mb), mb.shape[1], mb.shape[2], 1])
            yield ib, mb, frame_batch, coords_batch

    def run(self,
            face_path: Path,
            audio_path: Path,
            outfile: Path,
            *,
            fps: float = 25.0,
            static: Optional[bool] = None,
            pads: Tuple[int,int,int,int] = (0,10,0,0),
            face_det_batch_size: int = 16,
            wav2lip_batch_size: int = 128,
            resize_factor: int = 1,
            crop: Tuple[int,int,int,int] = (0,-1,0,-1),
            box: Tuple[int,int,int,int] = (-1,-1,-1,-1),
            rotate: bool = False,
            nosmooth: bool = False,
            work_dir_for_intermediate: Optional[Path] = None) -> dict:
        """
        work_dir_for_intermediate: 중간 파일을 둘 디렉토리(별도 temp 없이 출력 디렉토리 사용).
        """
        with self._lock:
            frames, vid_fps = _read_video_frames(face_path, resize_factor, crop, rotate)
            is_image = (len(frames) == 1 and vid_fps < 0)
            if static is None:
                static = is_image
            if is_image:
                if fps is None or fps <= 0:
                    fps = 25.0
            else:
                fps = vid_fps

            # 중간 파일: 출력 디렉토리 내에서만 생성
            if work_dir_for_intermediate is None:
                work_dir_for_intermediate = outfile.parent
            work_dir_for_intermediate.mkdir(parents=True, exist_ok=True)

            # 오디오 표준화
            audio_16k = work_dir_for_intermediate / f"audio_16k_{int(time.time())}.wav"
            wav_path = _ensure_wav(audio_path, audio_16k)

            # mel
            wav = audio.load_wav(str(wav_path), 16000)
            mel = audio.melspectrogram(wav)
            if np.isnan(mel.reshape(-1)).sum() > 0:
                # 정리 후 에러
                try:
                    if audio_16k.exists():
                        audio_16k.unlink()
                except Exception:
                    pass
                raise ValueError("Mel contains NaN. TTS 출력인 경우 작은 잡음을 추가해 다시 시도하세요.")

            mel_step_size = 16
            mel_chunks = []
            mel_idx_multiplier = 80.0 / float(fps)
            i = 0
            while True:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > mel.shape[1]:
                    mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
                i += 1

            frames = frames[:len(mel_chunks)]

            # 얼굴 검출/크롭 (디버그 이미지는 출력 디렉토리로 저장)
            face_crops = self._face_detect(
                frames, pads, nosmooth, box, static, face_det_batch_size,
                debug_save_dir=work_dir_for_intermediate
            )

            h, w = frames[0].shape[:2]
            avi_path = work_dir_for_intermediate / f"result_{int(time.time())}.avi"
            out = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*'DIVX'), float(fps), (w, h))

            total_batches = int(np.ceil(float(len(mel_chunks)) / float(wav2lip_batch_size)))
            gen = self._datagen(frames, mel_chunks, face_crops, static, wav2lip_batch_size)

            t0 = time.time()
            with torch.no_grad():
                for _ in range(total_batches):
                    try:
                        ib, mb, frame_batch, coords_batch = next(gen)
                    except StopIteration:
                        break

                    ib = torch.FloatTensor(np.transpose(ib, (0, 3, 1, 2))).to(self.device)
                    mb = torch.FloatTensor(np.transpose(mb, (0, 3, 1, 2))).to(self.device)

                    pred = self.model(mb, ib)
                    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.

                    for p, f, c in zip(pred, frame_batch, coords_batch):
                        y1, y2, x1, x2 = c
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        out.write(f)

            out.release()

            # mux
            cmd = (
                f'ffmpeg -y -i "{avi_path}" -i "{wav_path}" '
                f'-c:v libx264 -crf 16 -preset slow -c:a aac -b:a 192k '
                f'-vf "scale=iw:ih" "{outfile}"'
            )
            rc = os.system(cmd)
            if rc != 0 or not Path(outfile).exists():
                # 중간물 정리
                try:
                    if avi_path.exists(): avi_path.unlink()
                except Exception:
                    pass
                try:
                    if audio_16k.exists(): audio_16k.unlink()
                except Exception:
                    pass
                raise RuntimeError("ffmpeg mux 실패")

            elapsed_ms = int((time.time() - t0) * 1000)

            # 중간물 정리(디렉토리 추가 생성 없이 파일만 삭제)
            try:
                if avi_path.exists(): avi_path.unlink()
            except Exception:
                pass
            try:
                if audio_16k.exists(): audio_16k.unlink()
            except Exception:
                pass

            return {
                "fps": float(fps),
                "frame_count": len(frames),
                "elapsed_ms": elapsed_ms,
            }

# ------------------------------
# 앱 전역 싱글톤 (lazy-init)
# ------------------------------
RUNNER: Optional[Wav2LipRunner] = None
_runner_init_lock = threading.Lock()

def _ensure_runner():
    global RUNNER
    if RUNNER is None:
        with _runner_init_lock:
            if RUNNER is None:  # double-check
                if not CHECKPOINT_PATH.exists():
                    raise HTTPException(status_code=500, detail=f"checkpoint not found: {CHECKPOINT_PATH}")
                RUNNER = Wav2LipRunner(checkpoint_path=CHECKPOINT_PATH)

class HealthResp(BaseModel):
    status: str = "ok"
    checkpoint: bool = True
    device: str = "cpu"
    model_loaded: bool = False
    detector_loaded: bool = False

@app.get("/health", response_model=HealthResp)
def health():
    # lazy-init이므로 여기서 굳이 로드하지 않음
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if RUNNER is None else RUNNER.device
    return HealthResp(
        status="ok",
        checkpoint=CHECKPOINT_PATH.exists(),
        device=device,
        model_loaded=bool(RUNNER and RUNNER.model is not None),
        detector_loaded=bool(RUNNER and RUNNER.detector is not None),
    )

# ------------------------------
# 엔드포인트
# ------------------------------
@app.post("/warmup")
def warmup(
    fps: float = Form(25.0),
    resize_factor: int = Form(1),
    nosmooth: bool = Form(False),
):
    """
    예열(= lazy-init 포함):
      - 최초 호출 시 모델/검출기 로드
      - 얼굴(영상): warmup/warmup_sadtalker.mp4 또는 sadtalker_output_queue 최신 mp4
      - 오디오:     warmup/warmup_out.wav (필수)
      - 결과:       warmup/warmup_wav2lip.mp4
      - 중간 파일은 warmup 디렉토리 내에만 생성/즉시 삭제 (별도 temp 디렉토리 없음)
    """
    _ensure_runner()

    face = WARMUP_DIR / "warmup_sadtalker.mp4"
    if not face.exists():
        face = _pick_latest(SADTALKER_OUT_DIR, ["*.mp4", "*.mov", "*.mkv", "*.avi"])
        if not face:
            raise HTTPException(status_code=400, detail="No face video. Run SadTalker first (warmup or infer).")

    wav = WARMUP_DIR / "warmup_out.wav"
    if not wav.exists():
        raise HTTPException(status_code=400, detail=f"Warmup audio not found: {wav}")

    tmp_out = WARMUP_DIR / f"warmup_wav2lip_{int(time.time())}.mp4"
    stats = RUNNER.run(
        face_path=face,
        audio_path=wav,
        outfile=tmp_out,
        fps=fps,
        static=None,
        resize_factor=resize_factor,
        nosmooth=nosmooth,
        work_dir_for_intermediate=WARMUP_DIR,  # 중간 파일도 여기서만 사용 후 삭제
    )
    final_path = WARMUP_DIR / "warmup_wav2lip.mp4"
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(tmp_out), str(final_path))

    return JSONResponse({
        "status": "ok",
        "message": "warmup done (lazy-init + singleton)",
        "face": str(face),
        "audio": str(wav),
        "output": str(final_path),
        **stats
    })

@app.post("/infer")
def infer(
    input_video_path: Optional[str] = Form(None),
    input_audio_path: Optional[str] = Form(None),
    output_basename: Optional[str] = Form(None),

    fps: float = Form(25.0),
    resize_factor: int = Form(1),
    nosmooth: bool = Form(False),

    static: Optional[bool] = Form(None),
    pad_top: int = Form(0),
    pad_bottom: int = Form(10),
    pad_left: int = Form(0),
    pad_right: int = Form(0),

    face_det_batch_size: int = Form(16),
    wav2lip_batch_size: int = Form(128),

    crop_top: int = Form(0),
    crop_bottom: int = Form(-1),
    crop_left: int = Form(0),
    crop_right: int = Form(-1),

    box_top: int = Form(-1),
    box_bottom: int = Form(-1),
    box_left: int = Form(-1),
    box_right: int = Form(-1),

    rotate: bool = Form(False),
):
    """
    내부 싱글톤으로 추론 수행 (모델 재로딩 없음)
    - lazy-init: 최초 infer에서 자동 로드 가능
    - 영상 기본: sadtalker_output_queue 최신 mp4
    - 오디오 기본: applio_output_queue 최신 wav/mp3/mp4/m4a
    - 결과는 wav2lip_output_queue에 저장
    - 중간 파일은 결과 디렉토리 내부에만 생성/즉시 삭제
    """
    _ensure_runner()

    if input_video_path:
        face = Path(input_video_path)
        if not face.exists():
            raise HTTPException(status_code=400, detail=f"input_video_path not found: {face}")
    else:
        face = _pick_latest(SADTALKER_OUT_DIR, ["*.mp4", "*.mov", "*.mkv", "*.avi"])
        if not face:
            raise HTTPException(status_code=400, detail="No face video found in sadtalker_output_queue/")

    if input_audio_path:
        aud = Path(input_audio_path)
        if not aud.exists():
            raise HTTPException(status_code=400, detail=f"input_audio_path not found: {aud}")
    else:
        aud = _pick_latest(APPLIO_OUT_DIR, ["*.wav", "*.mp3", "*.m4a", "*.mp4"])
        if not aud:
            raise HTTPException(status_code=400, detail="No audio found in applio_output_queue/")

    if output_basename:
        final_name = f"{output_basename}.mp4"
    else:
        final_name = f"wav2lip_{face.stem}.mp4"

    final_path = WAV2LIP_OUT_DIR / final_name
    if final_path.exists():
        final_path = WAV2LIP_OUT_DIR / f"{final_path.stem}_{int(time.time())}.mp4"

    stats = RUNNER.run(
        face_path=face,
        audio_path=aud,
        outfile=final_path,
        fps=fps,
        static=static,
        pads=(pad_top, pad_bottom, pad_left, pad_right),
        face_det_batch_size=face_det_batch_size,
        wav2lip_batch_size=wav2lip_batch_size,
        resize_factor=resize_factor,
        crop=(crop_top, crop_bottom, crop_left, crop_right),
        box=(box_top, box_bottom, box_left, box_right),
        rotate=rotate,
        nosmooth=nosmooth,
        work_dir_for_intermediate=WAV2LIP_OUT_DIR,  # 중간 파일은 여기만 사용 후 삭제
    )

    return JSONResponse({
        "status": "ok",
        "message": "Video generated successfully by Wav2Lip (singleton, no-temp-dir)",
        "face_video": str(face),
        "audio": str(aud),
        "output": str(final_path),
        **stats
    })
