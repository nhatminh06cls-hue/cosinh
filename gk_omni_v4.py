from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import cv2, math, numpy as np, torch, uvicorn, time, os, threading, queue, base64
from fastapi.responses import FileResponse as FRespFile
import sys, importlib.util, unicodedata, glob

# Lazy load report_generator
def _get_report_gen():
    spec = importlib.util.spec_from_file_location('report_generator',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_generator.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ══════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════
try:
    from gtts import gTTS
    TTS_OK = True
except ImportError:
    TTS_OK = False

AUDIO_DIR   = "/tmp/gk_audio"
CAPTURE_DIR = "/tmp/gk_captures"
os.makedirs(AUDIO_DIR,   exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

SCRIPTS = {
    "FRONT_START":    "Xin đứng thẳng, nhìn thẳng vào camera, hai tay thả lỏng.",
    "FRONT_MOVE":     "Xin đứng yên, không di chuyển.",
    "FRONT_WRONG":    "Xin quay mặt thẳng vào camera.",
    "FRONT_GOOD":     "Tư thế tốt, giữ nguyên.",
    "FRONT_DONE":     "Đã lưu dữ liệu mặt trước. Bây giờ xin nghiêng người sang một bên.",
    "SIDE_START":     "Xin nghiêng người chín mươi độ, mặt nhìn sang một bên.",
    "SIDE_MOVE":      "Xin đứng yên, không di chuyển.",
    "SIDE_WRONG":     "Xin nghiêng thêm, không quay mặt vào camera.",
    "SIDE_GOOD":      "Tư thế tốt, giữ nguyên.",
    "SIDE_DONE":      "Đã lưu dữ liệu mặt nghiêng. Chuẩn bị kiểm tra cột sống.",
    "ADAM_BACK":      "Xin quay lưng vào camera, đứng thẳng, hai tay thả lỏng.",
    "ADAM_BACK_OK":   "Tốt. Bây giờ xin từ từ cúi người về phía trước, tay thả xuống đất.",
    "ADAM_HOLD":      "Giữ nguyên tư thế. Hệ thống đang xác nhận.",
    "ADAM_BEND_OK":   "Hoàn hảo. Giữ yên. Đang phân tích cột sống.",
    "ADAM_DONE":      "Hoàn thành kiểm tra Adam. Xem kết quả tại báo cáo.",
    "ADAM_NEAR":      "Xin bước lại gần camera hơn.",
    "ADAM_FAR":       "Xin bước ra xa camera hơn.",
    "ADAM_MORE_BEND": "Xin cúi sâu hơn.",
    "ADAM_MOVING":    "Xin giữ yên, không di chuyển.",
    "REPORT_READY":   "Kiểm tra hoàn tất. Đang hiển thị báo cáo.",
    "WARN_SHOULDER":  "Cảnh báo, phát hiện lệch vai.",
    "WARN_KYPHOSIS":  "Cảnh báo, phát hiện gù lưng.",
}

_tts_queue    = queue.Queue()
_playing      = False
_playing_lock = threading.Lock()
_pending_file = ""
_pending_lock = threading.Lock()
_cooldown: dict[str, float] = {}

def _tts_worker():
    global _playing, _pending_file
    while True:
        key = _tts_queue.get()
        if not TTS_OK:
            _tts_queue.task_done(); continue
        text = SCRIPTS.get(key, "")
        if not text:
            _tts_queue.task_done(); continue
        fpath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        try:
            if not os.path.exists(fpath):
                gTTS(text=text, lang="vi").save(fpath)
            with _playing_lock:  _playing = True
            with _pending_lock:  _pending_file = f"{key}.mp3"
            time.sleep(max(2.0, len(text) * 0.07))
        except Exception as e:
            print(f"TTS error [{key}]: {e}")
        finally:
            with _playing_lock: _playing = False
            _tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(key: str, force: bool = False):
    now = time.time()
    if not force:
        if _cooldown.get(key, 0) + 5.0 > now: return
        with _playing_lock:
            if _playing: return
    _cooldown[key] = now
    if force:
        try:
            while True: _tts_queue.get_nowait(); _tts_queue.task_done()
        except queue.Empty: pass
    _tts_queue.put(key)

def _pregenerate_all():
    for key, text in SCRIPTS.items():
        fpath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        if not os.path.exists(fpath):
            try: gTTS(text=text, lang="vi").save(fpath)
            except: pass

# ══════════════════════════════════════════════════════
#  VẼ TIẾNG VIỆT
# ══════════════════════════════════════════════════════
def putVn(img, text, pos, color=(0,255,0), sz=20):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:    font = ImageFont.truetype("DejaVuSans.ttf", sz)
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════
#  MIDAS POSE DETECTOR
# ══════════════════════════════════════════════════════
class MidasPoseDetector:
    # Thực tế đo khi lưng nằm ngang ~90°: blob_w/blob_h ≈ 1.8–2.5
    # BLOB_RATIO_BENT = 1.5: ngưỡng cần vượt qua (rộng hơn cao 50%) → xem như đã cúi đủ sâu
    # Chú ý: giá trị này phải nhất quán với MidasPoseDetector.analyze() → prog_midas
    BLOB_RATIO_BENT  = 1.50   
    DEPTH_DIFF_STILL = 18.0   
    NEAR_PCT         = 25     
    YOLO_CONF_OK     = 0.35   

    def __init__(self, midas_model, transforms, device):
        self.midas      = midas_model
        self.transforms = transforms
        self.device     = device
        self._prev_dm   = None
        self._baseline_ratio = None
        self._diff_buf  = []        
        self._DIFF_BUF  = 5   

    def reset(self):
        self._prev_dm        = None
        self._baseline_ratio = None
        self._diff_buf       = []

    def calibrate(self, blob_ratio: float):
        self._baseline_ratio = blob_ratio

    def _run(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp = self.transforms(rgb).to(self.device)
        with torch.no_grad():
            pred = self.midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        dm = pred.cpu().numpy().astype(np.float32)
        mn, mx = dm.min(), dm.max()
        return (dm - mn) / (mx - mn + 1e-6) * 255.0

    def _blob_ratio(self, dm: np.ndarray):
        thresh = np.percentile(dm, 100 - self.NEAR_PCT)
        mask = ((dm >= thresh).astype(np.uint8)) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 1.0, 0.0
        c = max(cnts, key=cv2.contourArea)
        area_frac = cv2.contourArea(c) / (dm.shape[0] * dm.shape[1])
        _, _, w, h = cv2.boundingRect(c)
        return w / max(h, 1), area_frac

    def analyze(self, frame, yolo_box=None, yolo_kp_conf=0.0,
                yolo_bend_r=1.0, yolo_is_still=False):
        H, W = frame.shape[:2]
        if yolo_box:
            x1, y1, x2, y2 = yolo_box
            pad = 30
            crop = frame[max(0,y1-pad):min(H,y2+pad), max(0,x1-pad):min(W,x2+pad)]
            if crop.size == 0: crop = frame
        else:
            crop = frame

        dm = self._run(crop)
        if self._prev_dm is not None and self._prev_dm.shape == dm.shape:
            raw_diff = float(np.median(np.abs(dm.astype(np.float32) - self._prev_dm)))
        else:
            raw_diff = 999.0
        self._prev_dm = dm.copy()

        self._diff_buf.append(raw_diff if raw_diff < 500 else 50.0)
        if len(self._diff_buf) > self._DIFF_BUF:
            self._diff_buf.pop(0)
        depth_diff   = float(np.median(self._diff_buf))
        is_still_midas = depth_diff < self.DEPTH_DIFF_STILL

        blob_ratio, blob_area = self._blob_ratio(dm)
        stand_r = self._baseline_ratio if self._baseline_ratio else 0.55
        prog_midas = (blob_ratio - stand_r) / max(self.BLOB_RATIO_BENT - stand_r, 0.01)
        prog_midas = max(0.0, min(1.0, prog_midas))

        use_yolo = yolo_kp_conf >= self.YOLO_CONF_OK
        if use_yolo:
            prog_yolo = max(0.0, min(1.0, (1.0 - yolo_bend_r) / max(1.0 - 0.68, 0.01)))
            bend_progress = prog_yolo * 0.65 + prog_midas * 0.35
            is_bent  = bend_progress >= 1.0
            is_still = yolo_is_still or is_still_midas
            source   = "yolo"
        else:
            bend_progress = prog_midas
            is_bent  = prog_midas >= 1.0
            is_still = is_still_midas
            source   = "midas"

        return dict(
            source=source, is_bent=is_bent, is_still=is_still,
            bend_progress=bend_progress, blob_ratio=blob_ratio,
            depth_diff=depth_diff, dm=dm,
        )

    def make_depth_images(self, frame, yolo_box=None, spine_hint_x=None):
        H, W = frame.shape[:2]
        if yolo_box:
            x1, y1, x2, y2 = yolo_box
            ch_box = y2 - y1
            y2_safe = min(H, y1 + int(ch_box * 0.85))
            crop = frame[max(0,y1):y2_safe, max(0,x1):min(W,x2)]
            if crop.size == 0: crop = frame
            orig_crop = crop.copy()
        else:
            crop = frame
            orig_crop = frame

        # ── CLAHE: cân bằng sáng cục bộ TRƯỚC khi cho MiDaS chạy.
        # MiDaS dùng shading để suy ra depth — nếu ánh sáng lệch thì
        # depth map lệch ngay từ đầu. CLAHE loại bỏ gradient sáng 1 chiều
        # trước khi model thấy ảnh → kết quả depth ổn định hơn rất nhiều.
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            crop_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            crop_eq = crop   # fallback nếu CLAHE lỗi vì ảnh quá nhỏ

        dm = self._run(crop_eq)
        dm8 = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(dm8, cv2.COLORMAP_MAGMA)
        orig_r = cv2.resize(orig_crop, (colored.shape[1], colored.shape[0]))
        overlay = cv2.addWeighted(colored, 0.65, orig_r, 0.35, 0)

        dm_h, dm_w = dm.shape
        margin     = int(dm_w * 0.08)
        back_rows  = dm[int(dm_h * 0.20):, :]

        # ── Row-wise normalization: mỗi hàng normalize về [0,1] độc lập.
        # Loại bỏ bias độ sáng theo chiều dọc (ví dụ: đèn trên cao chiếu
        # xuống làm phần lưng trên sáng hơn phần lưng dưới).
        # ⚠ CHÚ Ý: row-norm triệt tiêu chênh lệch chiều cao tuyệt đối
        # giữa 2 bên — cần kết hợp với raw analysis bên dưới.
        back_rows_norm = np.zeros_like(back_rows, dtype=np.float32)
        for r in range(back_rows.shape[0]):
            row = back_rows[r, margin:dm_w-margin]
            if len(row) == 0: continue
            mn_r, mx_r = float(row.min()), float(row.max())
            if mx_r - mn_r > 1e-3:
                back_rows_norm[r, margin:dm_w-margin] = (row - mn_r) / (mx_r - mn_r)
            else:
                back_rows_norm[r, margin:dm_w-margin] = 0.0

        # Dùng back_rows_norm (đã chuẩn hóa hàng) thay cho back_rows gốc
        # trong toàn bộ phân tích bên dưới.
        dm_fg = np.zeros_like(back_rows_norm)
        for r in range(back_rows_norm.shape[0]):
            row = back_rows_norm[r, margin:dm_w-margin]
            if len(row) == 0: continue
            baseline = np.percentile(row, 20)
            dm_fg[r, margin:dm_w-margin] = np.maximum(0, back_rows_norm[r, margin:dm_w-margin] - baseline)

        # ─────────────────────────────────────────────────────────────────
        # TÌM VỊ TRÍ CỘT SỐNG
        #
        # Giải phẫu học: khi người cúi 90°, sống lưng là ĐƯỜNG TRŨNG
        # giữa 2 cơ cạnh sống lưng (paraspinal muscles) - KHÔNG phải đỉnh.
        # → argmax của depth profile = đỉnh cơ = SAI về giải phẫu.
        #
        # Phương pháp đúng nhất:
        # 1. YOLO shoulder midpoint: điểm giữa 2 vai = vị trí cột sống (tin cậy cao)
        # 2. Depth bilateral minimum: tìm đường trũng đối xứng nhất (nếu ko có YOLO)
        # 3. Fallback: giữa crop
        # ─────────────────────────────────────────────────────────────────
        col_profile = np.mean(dm_fg, axis=0)
        kernel_size = max(3, dm_w // 10) | 1
        col_smooth  = np.convolve(col_profile, np.ones(kernel_size)/kernel_size, mode='same')

        search_lo = margin
        search_hi = dm_w - margin

        # ── Convert YOLO hint sang depth-map coordinates (nếu có)
        hint_dm = None
        if spine_hint_x is not None:
            crop_h, crop_w = (orig_crop.shape[:2] if orig_crop is not None else (dm_h, dm_w))
            scale_x = dm_w / max(crop_w, 1)
            _hint = int(spine_hint_x * scale_x)
            if search_lo < _hint < search_hi:
                hint_dm = _hint

        # ── LUÔN chạy depth valley detection (không bypass khi có YOLO hint)
        # Sống lưng = local minimum nằm giữa 2 đỉnh cơ lưng đối xứng nhau.
        inv_seg = float(np.max(col_smooth[search_lo:search_hi])) - col_smooth[search_lo:search_hi]
        inv_seg = np.maximum(inv_seg, 0)

        search_range = search_hi - search_lo
        min_half = max(8, search_range // 6)

        best_col   = (search_lo + search_hi) // 2
        best_score = -2.0
        max_inv    = float(np.max(inv_seg)) + 1e-5
        max_smooth = float(np.max(col_smooth[search_lo:search_hi]))

        for c in range(search_lo + min_half, search_hi - min_half):
            # Chỉ xét các điểm là valley (giá trị thấp trong profile gốc)
            if col_smooth[c] > max_smooth * 0.80:
                continue
            half_w = min(c - search_lo, search_hi - c, search_range // 4)
            if half_w < 5: continue

            lv = col_smooth[c - half_w : c]
            rv = col_smooth[c : c + half_w][::-1]
            l_std = float(np.std(lv))
            r_std = float(np.std(rv))
            if l_std < 1e-4 or r_std < 1e-4: continue

            score = float(np.corrcoef(lv, rv)[0, 1])
            # Cộng thêm điểm cho valley sâu hơn (sống lưng rõ hơn)
            valley_bonus = (max_inv - float(inv_seg[c - search_lo])) / max_inv * 0.3
            score += valley_bonus

            # ── Nếu có YOLO hint: cộng thêm proximity bonus
            # Valley gần YOLO hint được ưu tiên hơn (nhưng KHÔNG bắt buộc)
            if hint_dm is not None:
                dist_to_hint = abs(c - hint_dm) / max(search_range, 1)
                # Bonus tối đa 0.2 khi dist=0, giảm dần về 0 khi dist=0.5+
                proximity_bonus = max(0.0, 0.2 * (1.0 - dist_to_hint * 2))
                score += proximity_bonus

            if score > best_score:
                best_score = score
                best_col   = c

        # ── Quyết định final: dùng valley nếu tìm được valley tốt,
        # fallback sang YOLO hint hoặc center nếu valley quá kém
        if best_score > 0.1:
            spine_col    = best_col
            spine_source = "valley_sym"
        elif hint_dm is not None:
            # Valley detection kém (profile phẳng / ồn) → dùng YOLO hint
            spine_col    = hint_dm
            spine_source = "YOLO_fb"
        else:
            spine_col    = (search_lo + search_hi) // 2
            spine_source = "center_fb"

        print(f"[SPINE] valley_score={best_score:.3f} valley@{best_col} hint@{hint_dm} → {spine_source}@{spine_col}")

        # ─────────────────────────────────────────────────────────────────
        # PHÂN TÍCH 1: NORMALIZED HUMP (row-normalized) — shape-based
        # Đo hình dạng profile mỗi hàng, nhưng MẤT chênh lệch chiều cao.
        # ─────────────────────────────────────────────────────────────────
        row_hump   = []

        for r in range(back_rows_norm.shape[0]):
            l_v = dm_fg[r, margin:spine_col]
            r_v = dm_fg[r, spine_col:dm_w-margin]
            if len(l_v) < 5 or len(r_v) < 5: continue

            # Cắt bằng độ dài nhau để so sánh công bằng
            cmp_len = min(len(l_v), len(r_v))
            # Lấy phần gần sống lưng nhất (bên trong) để so sánh
            l_inner = l_v[-cmp_len:]   # phần phải của left (gần spine)
            r_inner = r_v[:cmp_len]    # phần trái của right (gần spine)

            l_std = float(np.std(l_inner))
            r_std = float(np.std(r_inner))

            # Đo hump: nếu 2 bên đều flat → bỏ qua hàng đó
            if l_std < 1e-3 and r_std < 1e-3:
                continue

            lmax = float(np.max(l_inner))
            rmax = float(np.max(r_inner))
            hump_total = lmax + rmax
            if hump_total > 1e-4:
                row_hump.append(abs(lmax - rmax) / hump_total * 100)

        def trimmed_median(arr, trim_frac=1/6):
            if len(arr) < 4: return 0.0
            t = max(1, int(len(arr) * trim_frac))
            return float(np.median(sorted(arr)[t:-t]))

        hump_score = trimmed_median(row_hump)

        # ─────────────────────────────────────────────────────────────────
        # PHÂN TÍCH 2: RAW DEPTH PEAK ASYMMETRY — chênh lệch chiều cao
        #
        # VẤN ĐỀ: row-norm ở trên normalize mỗi hàng về [0,1] độc lập
        # → nếu bên phải cao hơn bên trái rõ ràng, sau normalize 2 bên
        # đều thành ~1.0 → mất hết chênh lệch.
        #
        # GIẢI PHÁP: dùng depth map GỐC (back_rows), chia thành các band
        # ngang, mỗi band tính percentile 90 bên L vs R. Loại bỏ bias
        # dọc bằng cách normalize TỪNG BAND (không phải từng hàng) —
        # band rộng hơn nên vẫn giữ được chênh lệch ngang.
        # ─────────────────────────────────────────────────────────────────
        n_bands = max(4, back_rows.shape[0] // 20)
        band_h  = max(1, back_rows.shape[0] // n_bands)
        raw_asym_bands = []

        for b in range(n_bands):
            r_start = b * band_h
            r_end   = min((b + 1) * band_h, back_rows.shape[0])
            if r_end - r_start < 3: continue

            band = back_rows[r_start:r_end, :]

            # Lấy vùng L và R trên depth map GỐC
            l_band_raw = band[:, margin:spine_col]
            r_band_raw = band[:, spine_col:dm_w-margin]
            if l_band_raw.size < 10 or r_band_raw.size < 10: continue

            # Percentile 85 = đo "đỉnh" của mỗi bên (robust hơn max)
            l_peak = float(np.percentile(l_band_raw, 85))
            r_peak = float(np.percentile(r_band_raw, 85))

            # Normalize band: loại bỏ offset toàn band (bias dọc)
            band_min = float(np.percentile(band[:, margin:dm_w-margin], 10))
            l_peak_n = l_peak - band_min
            r_peak_n = r_peak - band_min

            peak_sum = l_peak_n + r_peak_n
            if peak_sum > 1e-3:
                band_asym = abs(l_peak_n - r_peak_n) / peak_sum * 100
                raw_asym_bands.append(band_asym)

        raw_asym = trimmed_median(raw_asym_bands) if raw_asym_bands else 0.0

        # ─────────────────────────────────────────────────────────────────
        # PHÂN TÍCH 3: COLUMN PROFILE PEAK ASYMMETRY
        #
        # Dùng depth GỐC (back_rows), tính column-mean profile, rồi so
        # sánh đỉnh bên L vs R — đo chênh lệch "vùng" thay vì từng hàng.
        # ─────────────────────────────────────────────────────────────────
        raw_col_profile = np.mean(back_rows[:, margin:dm_w-margin], axis=0)
        raw_col_smooth  = np.convolve(
            raw_col_profile,
            np.ones(max(3, len(raw_col_profile)//8) | 1) / (max(3, len(raw_col_profile)//8) | 1),
            mode='same')
        spine_in_margin = spine_col - margin  # spine position relative to margin
        if 2 < spine_in_margin < len(raw_col_smooth) - 2:
            l_profile = raw_col_smooth[:spine_in_margin]
            r_profile = raw_col_smooth[spine_in_margin:]
            if len(l_profile) > 3 and len(r_profile) > 3:
                l_peak_col = float(np.percentile(l_profile, 90))
                r_peak_col = float(np.percentile(r_profile, 90))
                col_base   = float(np.percentile(raw_col_smooth, 15))
                l_pc = l_peak_col - col_base
                r_pc = r_peak_col - col_base
                pc_sum = l_pc + r_pc
                col_asym = abs(l_pc - r_pc) / pc_sum * 100 if pc_sum > 1e-3 else 0.0
            else:
                col_asym = 0.0
        else:
            col_asym = 0.0

        # ─────────────────────────────────────────────────────────────────
        # KẾT HỢP: lấy MAX của 3 phương pháp
        #
        # - hump_score:  nhạy với shape chênh lệch (sau row-norm)
        # - raw_asym:    nhạy với chênh lệch chiều cao tuyệt đối (band)
        # - col_asym:    nhạy với chênh lệch profile tổng thể
        #
        # Lấy MAX vì: nếu BẤT KỲ phương pháp nào phát hiện lệch thì
        # nên báo — tốt hơn là bỏ sót (scoliosis screening).
        # ─────────────────────────────────────────────────────────────────
        asym = max(hump_score, raw_asym, col_asym)

        lm = float(np.mean(dm_fg[:, margin:spine_col])) if spine_col > margin else 0.0
        rm = float(np.mean(dm_fg[:, spine_col:dm_w-margin])) if dm_w-margin > spine_col else 0.0

        img_h = colored.shape[0]

        # ── Debug visualization: vẽ vùng L (xanh dương) và R (đỏ) để xác minh
        cv2.rectangle(colored, (margin, 0), (spine_col, img_h), (255, 150, 0), 1)  # L = blue-ish
        cv2.rectangle(colored, (spine_col, 0), (dm_w-margin, img_h), (0, 0, 255), 1)  # R = red
        cv2.rectangle(overlay, (margin, 0), (spine_col, img_h), (255, 150, 0), 1)
        cv2.rectangle(overlay, (spine_col, 0), (dm_w-margin, img_h), (0, 0, 255), 1)

        # ── Nếu có YOLO hint, vẽ thêm đường hint để so sánh
        if hint_dm is not None and hint_dm != spine_col:
            for seg_y in range(0, img_h, 10):
                cv2.line(colored, (hint_dm, seg_y), (hint_dm, min(seg_y+5, img_h)), (0, 200, 200), 1)
                cv2.line(overlay, (hint_dm, seg_y), (hint_dm, min(seg_y+5, img_h)), (0, 200, 200), 1)

        cv2.line(colored, (spine_col,0),(spine_col,img_h),(255,255,255),2)
        cv2.line(overlay,  (spine_col,0),(spine_col,img_h),(255,255,0),  2)
        label  = f"Asym: {asym:.1f}%  hump:{hump_score:.1f} raw:{raw_asym:.1f} col:{col_asym:.1f}  [{spine_source}]"
        label2 = f"L:{lm:.3f}  R:{rm:.3f}  spine@{spine_col}  rows:{len(row_hump)}  bands:{len(raw_asym_bands)}"
        cv2.putText(colored, label, (8,img_h-20),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
        cv2.putText(colored, label2,(8,img_h-6), cv2.FONT_HERSHEY_SIMPLEX,0.33,(200,200,200),1)
        cv2.putText(overlay,  label, (8,img_h-20),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,0),  1)
        cv2.putText(overlay,  label2,(8,img_h-6), cv2.FONT_HERSHEY_SIMPLEX,0.33,(200,200,0),  1)
        print(f"[DEPTH DEBUG] hump={hump_score:.2f}% raw_band={raw_asym:.2f}% col={col_asym:.2f}% → final={asym:.2f}%  spine={spine_source}@{spine_col}")
        return colored, overlay, asym


def _b64(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode()

def _kp_conf(res_kp):
    try:
        c = res_kp.conf[0].cpu().numpy()
        vals = [float(c[i]) for i in [5,6,11,12] if i<len(c)]
        return sum(vals)/len(vals) if vals else 0.0
    except: return 0.0

# ══════════════════════════════════════════════════════
#  AI MODELS (global)
# ══════════════════════════════════════════════════════
yolo_model = midas_model = device = midas_transforms = None
pose_detector: MidasPoseDetector = None

# ══════════════════════════════════════════════════════
#  CAMERA THREAD - tách capture khỏi inference loop
#  Thread chạy độc lập 30fps, luôn giữ frame mới nhất.
#  Inference loop đọc _frame không bao giờ block chờ camera.
# ══════════════════════════════════════════════════════
class CameraThread:
    def __init__(self):
        self._cap     = None
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self.opened   = False

    def start(self):
        cap = self._open_camera()
        if cap is None:
            print("[CAMERA] Khong mo duoc camera!")
            return False
        self._cap     = cap
        self.opened   = True
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[CAMERA] CameraThread da khoi dong.")
        return True

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened(): self._cap.release()
        self.opened = False

    def read(self):
        """Tra ve (ok, frame) giong cap.read() nhung khong block."""
        with self._lock:
            if self._frame is None: return False, None
            return True, self._frame.copy()

    def _open_camera(self):
        def warm(c, n=15):
            for _ in range(n):
                r, f = c.read()
                if r and f is not None: return True
                time.sleep(0.03)
            return False
        for idx, backend in [(1, cv2.CAP_V4L2), (1, cv2.CAP_ANY), (0, cv2.CAP_ANY)]:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    c.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    c.set(cv2.CAP_PROP_FPS, 30)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if warm(c): return c
                c.release()
            except Exception: pass
        return None

    def _loop(self):
        """Vong capture chay lien tuc trong thread rieng.
        Luon doc frame moi nhat vao bo nho - inference loop
        chi viec lay ra khong can cho."""
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

# ══════════════════════════════════════════════════════
#  LIFESPAN (AUTO-DETECT LOCAL CACHE)
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    global cam_thread, yolo_model, midas_model, device, midas_transforms, pose_detector
    print("=== KHỞI TẠO AI ===")
    yolo_model = YOLO("yolov8m-pose.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đoạn này sẽ tự quét trong máy tính xem có MiDaS chưa
    hub_dir = torch.hub.get_dir()
    midas_repos = glob.glob(os.path.join(hub_dir, "intel-isl_MiDaS_*"))
    
    if midas_repos:
        local_repo = midas_repos[0] # Lấy ngay thư mục đầu tiên tìm thấy
        print(f"[INFO] Tìm thấy cache local! Đang tải MiDaS Offline từ: {local_repo}")
        midas_model = torch.hub.load(local_repo, "MiDaS_small", source="local")
        midas_transforms = torch.hub.load(local_repo, "transforms", source="local").small_transform
    else:
        print("[INFO] Không tìm thấy cache. Đang thử tải từ GitHub (Online)...")
        try:
            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        except Exception as e:
            print(f"[ERROR] Lỗi tải từ GitHub: {e}")
            print("=> Hướng giải quyết: Hãy kết nối mạng ổn định hoặc chờ GitHub hết nghẽn.")
            
    midas_model.to(device).eval()
    pose_detector = MidasPoseDetector(midas_model, midas_transforms, device)
    
    cam_thread = CameraThread()
    cam_thread.start()
    if TTS_OK: threading.Thread(target=_pregenerate_all, daemon=True).start()
    yield
    cam_thread.stop()

app = FastAPI(title="BME HUST", lifespan=lifespan)
cam_thread = None  # type: CameraThread | None

# ══════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════
current_mode   = "FRONT"
stable_counter = 0
pathology_ctr  = 0
prev_pelvis    = None
adam_state     = "WAIT_BACK"
baseline_h     = 0
baseline_fill  = 0.0   
auto_advance   = True
kyphosis_streak = 0   # Frame liên tiếp vượt ngưỡng KYPHOSIS_THRESH

adam_bend_stable = 0
adam_prev_kp     = None
adam_vel_buf     = []
adam_bend_r_buf  = []          
adam_smooth_prog = 0.0         
VEL_BUF          = 8
BEND_R_BUF       = 10          
EMA_ALPHA        = 0.15        
VEL_THRESH        = 3.5
BEND_STABLE_NEED  = 45   # Tăng 25→45: cần giữ cúi ~1.5s mới chụp Adam
# BENT_ACCEPT: ngưỡng progress cần vượt qua để bắt đầu đếm giữ yên
# 0.75 = cần cúi 75% của lộ trình từ đứng thẳng → lưng ngang (safe margin)
BENT_ACCEPT       = 0.75

# ── Keypoint EMA smoothing (giảm jitter đo góc)
# Mỗi midpoint (me, ms, mh) được làm mượt theo EMA trước khi tính góc.
# KP_EMA = 0.0 → dùng nguyên raw; KP_EMA = 0.7 → rất mượt nhưng lag nhẹ.
KP_EMA   = 0.40
_kp_me_s = None  # EMA state cho midpoint mat/tai (me) — tuple(x,y) | None
_kp_ms_s = None  # EMA state cho midpoint vai (ms)
_kp_mh_s = None  # EMA state cho midpoint hong (mh)

def _ema_kp(prev, cur, alpha=KP_EMA):
    """EMA 2D: trả về tuple (x, y) đã làm mượt."""
    if prev is None: return (float(cur[0]), float(cur[1]))
    return (alpha * float(cur[0]) + (1-alpha) * prev[0],
            alpha * float(cur[1]) + (1-alpha) * prev[1])   

report_data = {
    "FRONT": {"sh_angle":0,"shift_ratio":0,"status":"Chua do","cap_b64":""},
    "SIDE":  {"torso_tilt":0,"neck_ratio":0,"status":"Chua do","cap_b64":""},
    "ADAM":  {"asym_index":0,"status":"Chua do","depth_b64":"","overlay_b64":"","raw_b64":""},
}

MOVE_LIM  = 7.0;  LOCK_TIME  = 180   # Tăng 120→180: cần giữ ~6s mới lưu bệnh lý
TWIST_LIM = 4.0;  FRONT_TILT = 5.0;  SH_RATIO_MIN = 0.18;  LAT_SHIFT  = 0.10
SIDE_TILT = 15.0; NECK_MIN   = 0.25; SH_SIDE_MAX  = 0.12
KYPHOSIS_THRESH = 8.0  # Sau trừ baseline 8°: tương đương raw 16° (gù vừa, có ý nghĩa lâm sàng)
KYPHOSIS_STREAK = 10   # Số frame liên tiếp vượt ngưỡng mới cảnh báo (chống false positive)
MIN_FILL  = 0.45; MAX_FILL   = 0.92
STABLE_NEED  = 120  # Tăng 75→120: cần giữ ~4s mới lưu tư thế bình thường
STAND_FRAMES = 120  # Tăng 90→120: cần giữ ~4s để xác nhận đứng thẳng Adam

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def _switch_mode(new):
    global current_mode, stable_counter, pathology_ctr, prev_pelvis, kyphosis_streak
    global adam_state, baseline_h, baseline_fill, adam_bend_stable, adam_prev_kp, adam_vel_buf
    global adam_bend_r_buf, adam_smooth_prog
    current_mode     = new
    stable_counter   = 0
    pathology_ctr    = 0
    kyphosis_streak  = 0
    prev_pelvis      = None
    adam_bend_stable = 0
    adam_prev_kp     = None
    adam_vel_buf     = []
    adam_bend_r_buf  = []
    adam_smooth_prog = 0.0
    if new == "ADAM":
        adam_state    = "WAIT_BACK"
        baseline_h    = 0
        baseline_fill = 0.0
        pose_detector.reset()

def _yolo_velocity(kp_now, kp_prev):
    if kp_prev is None: return 999.0
    total, n = 0.0, 0
    for i in [5,6,11,12]:
        if i < len(kp_now) and i < len(kp_prev):
            total += math.hypot(float(kp_now[i][0]-kp_prev[i][0]),
                                float(kp_now[i][1]-kp_prev[i][1]))
            n += 1
    return total/n if n else 999.0

# ══════════════════════════════════════════════════════
#  VIDEO STREAM
# ══════════════════════════════════════════════════════
def generate_frames():
    global current_mode, stable_counter, pathology_ctr, prev_pelvis, kyphosis_streak
    global _kp_me_s, _kp_ms_s, _kp_mh_s, cam_thread
    global adam_state, baseline_h, baseline_fill, adam_bend_stable, adam_prev_kp, adam_vel_buf
    global adam_bend_r_buf, adam_smooth_prog

    if cam_thread is None or not cam_thread.opened:
        blank = np.zeros((480,640,3), np.uint8)
        cv2.putText(blank,"CAMERA OFFLINE",(60,240),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,255),3)
        _, buf = cv2.imencode('.jpg', blank)
        while True:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(1)
        return

    while True:
        # CameraThread luôn giữ frame mới nhất — không bao giờ block
        ok, frame = cam_thread.read()
        if not ok or frame is None: time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]

        res = yolo_model(frame, verbose=False)
        smsg, scol = "Không thấy bệnh nhân", (0,0,255)
        has_box = has_kp = False
        x1=y1=x2=y2=ch=cw=0
        kp_raw = None

        if res[0].boxes is not None and len(res[0].boxes.xyxy):
            b = res[0].boxes.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int, b[:4])
            ch, cw = y2-y1, x2-x1
            has_box = True

        if res[0].keypoints is not None and len(res[0].keypoints.xy):
            kp_raw = res[0].keypoints.xy[0]
            if len(kp_raw) >= 13:
                le, re  = kp_raw[3], kp_raw[4]
                ls, rs  = kp_raw[5], kp_raw[6]
                lh, rh  = kp_raw[11], kp_raw[12]
                # ── EMA smoothing cho từng midpoint trước khi tính góc
                # Loại bỏ jitter frame-to-frame do noise detection của YOLO
                raw_me = ((float(le[0])+float(re[0]))/2, (float(le[1])+float(re[1]))/2)
                raw_ms = ((float(ls[0])+float(rs[0]))/2, (float(ls[1])+float(rs[1]))/2)
                raw_mh = ((float(lh[0])+float(rh[0]))/2, (float(lh[1])+float(rh[1]))/2)
                _kp_me_s = _ema_kp(_kp_me_s, raw_me)
                _kp_ms_s = _ema_kp(_kp_ms_s, raw_ms)
                _kp_mh_s = _ema_kp(_kp_mh_s, raw_mh)
                me, ms, mh = _kp_me_s, _kp_ms_s, _kp_mh_s
                has_kp = True
        else:
            # Không detect được người → reset EMA để tránh drift
            _kp_me_s = _kp_ms_s = _kp_mh_s = None

        # Giá trị mặc định an toàn khi không detect được keypoint
        # (tránh NameError / Pylance unbound-variable warning)
        if not has_kp:
            le = re = ls = rs = lh = rh = (0.0, 0.0)
            me = ms = mh = (0.0, 0.0)
            kp_raw = None

        # --- FRONT ---
        if current_mode == "FRONT":
            cv2.rectangle(frame,(W//6,H//10),(5*W//6,9*H//10),(100,220,255),2)
            if not (has_box and has_kp):
                speak("FRONT_START")
                smsg = "Xin đứng vào khung hình, nhìn thẳng vào camera"
            else:
                sh_ang = math.degrees(math.atan2(abs(ls[1]-rs[1]), abs(ls[0]-rs[0])))
                moved  = dist(mh, prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh
                shw  = abs(ls[0]-rs[0])
                shr  = shw/ch if ch else 0
                latr = abs(ms[0]-mh[0])/shw if shw else 0
                tdx,tdy = abs(ms[0]-mh[0]), abs(ms[1]-mh[1])
                ttilt = math.degrees(math.atan2(tdx,tdy)) if tdy else 90

                if moved > MOVE_LIM:
                    smsg,scol = "Xin đứng yên!",(0,0,255)
                    stable_counter = pathology_ctr = 0; speak("FRONT_MOVE")
                elif shr < SH_RATIO_MIN or latr > LAT_SHIFT:
                    smsg,scol = "Quay mặt thẳng vào camera!",(0,0,255)
                    stable_counter = pathology_ctr = 0; speak("FRONT_WRONG")
                elif ttilt > FRONT_TILT or sh_ang > TWIST_LIM:
                    pathology_ctr += 1; stable_counter = 0
                    pct = int(pathology_ctr/LOCK_TIME*100)
                    smsg,scol = f"⚠ Phát hiện lệch vai! {pct}%",(0,140,255)
                    if pct==5: speak("WARN_SHOULDER")
                else:
                    stable_counter += 1; pathology_ctr = 0
                    pct = int(stable_counter/STABLE_NEED*100)
                    smsg,scol = f"✓ Tư thế chuẩn... {pct}%",(0,220,100)
                    if stable_counter==12: speak("FRONT_GOOD")

                if stable_counter>=STABLE_NEED or pathology_ctr>=LOCK_TIME:
                    if report_data["FRONT"]["status"]=="Chua do":
                        _, buf_cap = cv2.imencode('.jpg', frame)
                        report_data["FRONT"] = {
                            "sh_angle":sh_ang,"shift_ratio":latr*100,
                            "status":"Da hoan thanh",
                            "cap_b64": base64.b64encode(buf_cap).decode()
                        }
                        speak("FRONT_DONE", force=True)
                    smsg,scol = "✅ ĐÃ LƯU! Đang chuyển...",(0,255,120)
                    if stable_counter>=STABLE_NEED: stable_counter+=1
                    if pathology_ctr>=LOCK_TIME: pathology_ctr+=1
                    if auto_advance and (stable_counter>STABLE_NEED+90 or pathology_ctr>LOCK_TIME+90):
                        _switch_mode("SIDE")

                cv2.line(frame,(int(ls[0]),int(ls[1])),(int(rs[0]),int(rs[1])),(0,80,255),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,255),2)

        # --- SIDE ---
        elif current_mode == "SIDE":
            cv2.rectangle(frame,(W//6,H//10),(5*W//6,9*H//10),(255,200,0),2)
            if not (has_box and has_kp):
                speak("SIDE_START")
                smsg = "Nghiêng người 90°, mặt nhìn sang một bên"
            else:
                nkl  = dist(me,ms); tsl = dist(ms,mh)
                nkr  = nkl/tsl if tsl else 0
                shw  = abs(ls[0]-rs[0])
                shr  = shw/ch if ch else 0
                moved = dist(mh, prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh

                # ── KYPHOTIC ANGLE: góc uốn cong tại vai (ms) nhìn từ cạnh
                # Phương pháp: đo góc lệch của vector cổ→vai so với trục đứng (Y)
                # Khi người đứng thẳng: vector vai→hông gần thẳng đứng → góc ≈ 0°
                # Khi người gù: phần trên lưng đổ ra trước → góc tăng
                #
                # QUAN TRỌNG: Giải phẫu bình thường — tai luôn nhô trước vai ~8-10°
                # (tư thế "neutral head position"). Phải trừ baseline này ra
                # để người dáng chuẩn đọc ~0-3° thay vì 10-12°.
                #
                # Dùng vector vai→hông làm trục tham chiếu (ổn định hơn so với trục Y tuyệt đối)
                # rồi đo độ lệch của vector cổ→vai so với trục đó.
                KYPHOSIS_BASELINE = 8.0  # Offset giải phẫu bình thường (độ)

                v_neck = (float(me[0]-ms[0]), float(me[1]-ms[1]))   # vai → cổ
                v_hip  = (float(mh[0]-ms[0]), float(mh[1]-ms[1]))   # vai → hông

                mag_n = math.hypot(*v_neck)
                mag_h = math.hypot(*v_hip)
                if mag_n > 1e-3 and mag_h > 1e-3:
                    cos_a = max(-1.0, min(1.0,
                        (v_neck[0]*v_hip[0] + v_neck[1]*v_hip[1]) / (mag_n * mag_h)))
                    angle_at_shoulder = math.degrees(math.acos(cos_a))  # 180° = thẳng
                else:
                    angle_at_shoulder = 180.0

                # kyphotic_angle: lệch khỏi đường thẳng (0° = hoàn toàn thẳng)
                raw_kyphotic = 180.0 - angle_at_shoulder

                # ── Bổ sung: đo độ nghiêng tuyệt đối của phần lưng trên (cổ→vai)
                # so với trục đứng thực (pixel Y xuống = dương)
                # Khi gù: đầu/cổ nhô ra trước (X thay đổi nhiều hơn Y)
                if mag_n > 1e-3:
                    # góc của vector cổ→vai so với trục Y âm (hướng lên)
                    # atan2(dx, -dy): = 0° khi hoàn toàn thẳng đứng lên
                    lean_angle = abs(math.degrees(math.atan2(v_neck[0], -v_neck[1])))
                else:
                    lean_angle = 0.0

                # Kết hợp 2 chỉ số: lấy max rồi trừ baseline giải phẫu
                # lean_angle * 0.5: giảm hệ số vì lean tự nhiên ~10-15° đã bao gồm
                # trong KYPHOSIS_BASELINE, không nên đếm double
                raw_combined = max(raw_kyphotic, lean_angle * 0.5)
                kyphotic_angle = max(0.0, raw_combined - KYPHOSIS_BASELINE)

                # Dự phòng: ttilt vẫn dùng để kiểm tra tư thế nghiêng đúng
                tdx,tdy = abs(ms[0]-mh[0]), abs(ms[1]-mh[1])
                ttilt = math.degrees(math.atan2(tdx,tdy)) if tdy else 90

                if moved > MOVE_LIM:
                    smsg,scol = "Xin đứng yên!",(0,0,255)
                    stable_counter = pathology_ctr = 0
                    kyphosis_streak = 0
                    speak("SIDE_MOVE")
                elif shr > SH_SIDE_MAX or nkr < NECK_MIN:
                    smsg,scol = "Nghiêng thêm, không quay mặt vào camera!",(0,0,255)
                    stable_counter = pathology_ctr = 0
                    kyphosis_streak = 0
                    speak("SIDE_WRONG")
                elif kyphotic_angle > KYPHOSIS_THRESH:
                    # Yêu cầu phải duy trì liên tục KYPHOSIS_STREAK frames mới cảnh báo
                    # Tránh false positive do người dùng vô tình đổi tư thế 1-2 frames
                    kyphosis_streak = min(KYPHOSIS_STREAK, kyphosis_streak + 1)
                    if kyphosis_streak >= KYPHOSIS_STREAK:
                        pathology_ctr += 1; stable_counter = 0
                        pct = int(pathology_ctr/LOCK_TIME*100)
                        smsg,scol = f"⚠ Phát hiện gù lưng ({kyphotic_angle:.0f}°)! {pct}%",(0,140,255)
                        if pct==5: speak("WARN_KYPHOSIS")
                    else:
                        smsg,scol = f"⚠ Gù lưng ({kyphotic_angle:.0f}°)? Đang xác nhận...",(0,180,255)
                else:
                    kyphosis_streak = max(0, kyphosis_streak - 1)
                    stable_counter += 1; pathology_ctr = 0
                    pct = int(stable_counter/STABLE_NEED*100)
                    smsg,scol = f"✓ Tư thế chuẩn ({kyphotic_angle:.0f}°)... {pct}%",(0,220,100)
                    if stable_counter==12: speak("SIDE_GOOD")

                if stable_counter>=STABLE_NEED or pathology_ctr>=LOCK_TIME:
                    if report_data["SIDE"]["status"]=="Chua do":
                        _, buf_cap = cv2.imencode('.jpg', frame)
                        report_data["SIDE"] = {
                            "torso_tilt":ttilt,"kyphotic_angle":kyphotic_angle,
                            "neck_ratio":nkr*100,
                            "status":"Da hoan thanh",
                            "cap_b64": base64.b64encode(buf_cap).decode()
                        }
                        speak("SIDE_DONE", force=True)
                    smsg,scol = "✅ ĐÃ LƯU! Đang chuyển...",(0,255,120)
                    if stable_counter>=STABLE_NEED: stable_counter+=1
                    if pathology_ctr>=LOCK_TIME: pathology_ctr+=1
                    if auto_advance and (stable_counter>STABLE_NEED+90 or pathology_ctr>LOCK_TIME+90):
                        _switch_mode("ADAM"); speak("ADAM_BACK", force=True)

                # Vẽ đường cong cột sống: cổ → vai → hông (3 điểm)
                cv2.line(frame,(int(me[0]),int(me[1])),(int(ms[0]),int(ms[1])),(255,80,200),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,255),3)
                # Hiển thị góc gù tại điểm vai
                ang_col = (0,80,255) if kyphotic_angle > KYPHOSIS_THRESH else (0,220,100)
                cv2.putText(frame, f"{kyphotic_angle:.0f}deg",
                    (int(ms[0])+8, int(ms[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ang_col, 2)

        # --- ADAM ---
        elif current_mode == "ADAM":
            if not has_box:
                speak("ADAM_BACK")
                smsg,scol = "Xin quay LƯNG vào camera và đứng vào khung hình",(0,180,255)
            else:
                fill = ch/H
                if fill < MIN_FILL:
                    speak("ADAM_NEAR"); smsg,scol="Lại GẦN camera hơn",(0,140,255)
                    adam_bend_stable = stable_counter = 0
                elif fill > MAX_FILL:
                    speak("ADAM_FAR"); smsg,scol="Lui RA XA camera hơn",(0,140,255)
                    adam_bend_stable = stable_counter = 0
                else:
                    box_col = (0,255,120) if adam_state in ("WAIT_BEND","SCANNING","DONE") else (100,180,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),box_col,2)

                    if adam_state == "WAIT_BACK":
                        speak("ADAM_BACK")
                        smsg,scol = "Quay LƯNG vào camera, đứng thẳng...",(100,200,255)
                        if has_kp:
                            shw = abs(ls[0]-rs[0])
                            shr = shw/ch if ch else 0
                            tdx,tdy = abs(ms[0]-mh[0]),abs(ms[1]-mh[1])
                            ttilt = math.degrees(math.atan2(tdx,tdy)) if tdy else 90

                            vel = _yolo_velocity(kp_raw, adam_prev_kp)
                            adam_vel_buf.append(min(vel,50.0))
                            if len(adam_vel_buf)>VEL_BUF: adam_vel_buf.pop(0)
                            avg_vel = sum(adam_vel_buf)/len(adam_vel_buf)
                            adam_prev_kp = kp_raw

                            kp_conf_all = res[0].keypoints.conf
                            if kp_conf_all is not None and len(kp_conf_all):
                                c = kp_conf_all[0].cpu().numpy()
                                face_conf = float(np.mean([c[0], c[1], c[2]]))
                            else:
                                face_conf = 1.0  
                            is_back_facing = face_conf < 0.40
                            is_straight = shr > 0.16 and ttilt < 20
                            is_still    = avg_vel < VEL_THRESH

                            if is_back_facing and is_straight and is_still:
                                stable_counter += 1
                                pct = int(stable_counter/STAND_FRAMES*100)
                                smsg,scol = f"✓ Xác nhận lưng vào camera... {pct}%",(0,220,100)
                                if stable_counter >= STAND_FRAMES:
                                    baseline_h    = ch
                                    baseline_fill = ch / H  
                                    pd_result = pose_detector.analyze(
                                        frame, yolo_box=(x1,y1,x2,y2),
                                        yolo_kp_conf=_kp_conf(res[0].keypoints),
                                        yolo_bend_r=1.0, yolo_is_still=True)
                                    pose_detector.calibrate(pd_result["blob_ratio"])
                                    adam_state = "WAIT_BEND"
                                    stable_counter = 0; adam_bend_stable = 0
                                    adam_vel_buf = []
                                    speak("ADAM_BACK_OK", force=True)
                            else:
                                stable_counter = max(0, stable_counter-1)
                                if not is_back_facing:
                                    hint = "→ Quay LƯNG vào camera (đừng nhìn vào cam)"
                                elif not is_still:
                                    hint = "→ Đứng yên, không di chuyển"
                                else:
                                    hint = "→ Đứng THẲNG hơn"
                                frame = putVn(frame, hint, (W//2-190, H-80), (0,200,255), 22)

                    elif adam_state == "WAIT_BEND":
                        vel = _yolo_velocity(kp_raw, adam_prev_kp) if has_kp else 999.0
                        adam_vel_buf.append(min(vel,50.0))
                        if len(adam_vel_buf)>VEL_BUF: adam_vel_buf.pop(0)
                        avg_vel = sum(adam_vel_buf)/len(adam_vel_buf)
                        if has_kp: adam_prev_kp = kp_raw

                        cur_fill = ch / H
                        adam_bend_r_buf.append(cur_fill)
                        if len(adam_bend_r_buf) > BEND_R_BUF: adam_bend_r_buf.pop(0)
                        smooth_fill = float(np.median(adam_bend_r_buf))

                        FILL_STAND = 0.78   
                        FILL_BENT  = 0.52   
                        raw_prog = (FILL_STAND - smooth_fill) / (FILL_STAND - FILL_BENT)
                        raw_prog = max(0.0, min(1.2, raw_prog))

                        yolo_bend_r  = 1.0 - raw_prog
                        yolo_conf    = _kp_conf(res[0].keypoints) if res[0].keypoints else 0.0
                        yolo_still   = avg_vel < VEL_THRESH

                        frame = putVn(frame, f"fill:{smooth_fill:.2f}", (W-90, H-90), (180,180,180), 16)

                        pd = pose_detector.analyze(
                            frame,
                            yolo_box      = (x1,y1,x2,y2) if has_box else None,
                            yolo_kp_conf  = yolo_conf,
                            yolo_bend_r   = yolo_bend_r,
                            yolo_is_still = yolo_still,
                        )

                        adam_smooth_prog = (EMA_ALPHA * min(raw_prog, 1.2) + (1 - EMA_ALPHA) * adam_smooth_prog)
                        display_prog = adam_smooth_prog

                        pb_w = int(W * 0.42)
                        pb_x = W//2 - pb_w//2
                        pb_y = H - 62
                        cv2.rectangle(frame,(pb_x,pb_y),(pb_x+pb_w,pb_y+16),(30,30,30),-1)
                        fill_px = int(pb_w * min(display_prog, 1.0))
                        bar_col = (0,200,80) if display_prog >= BENT_ACCEPT else (0,140,255)
                        cv2.rectangle(frame,(pb_x,pb_y),(pb_x+fill_px,pb_y+16),bar_col,-1)
                        src_tag = f"[{pd['source'].upper()}]"
                        frame = putVn(frame, f"Góc cúi: {int(display_prog*100)}%  {src_tag}", (pb_x, pb_y-24), (255,220,100), 18)

                        ax = W-62
                        cv2.arrowedLine(frame,(ax,H//4),(ax,3*H//4),(0,220,255),4,tipLength=0.25)
                        frame = putVn(frame,"CÚI",(ax-25,3*H//4+8),(0,220,255),26)

                        crop_lum = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                        lum = float(np.mean(cv2.cvtColor(crop_lum, cv2.COLOR_BGR2GRAY))) if crop_lum.size > 0 else 0
                        lum_col = (0,200,80) if lum >= 60 else (0,60,255)
                        frame = putVn(frame, f"Sáng: {lum:.0f}/255 {'✓' if lum>=60 else '⚠ Cần thêm sáng!'}",
                                      (pb_x, pb_y - 46), lum_col, 16)

                        src_col = (0,220,100) if pd["source"]=="yolo" else (200,80,255)
                        cv2.circle(frame,(W-18,52),7,src_col,-1)
                        frame = putVn(frame, f"blob:{pd['blob_ratio']:.2f}  diff:{pd['depth_diff']:.1f}", (W-160,70),(150,150,150),13)

                        if not (display_prog >= BENT_ACCEPT):
                            smsg,scol = "Cúi sâu hơn — lưng nghiêng ~50°",(0,140,255)
                            adam_bend_stable = max(0, adam_bend_stable-2)
                            if display_prog < 0.3: speak("ADAM_MORE_BEND")
                        elif not pd["is_still"]:
                            smsg,scol = f"Giữ tư thế...",(0,200,150)
                            adam_bend_stable = max(0, adam_bend_stable-1)
                        else:
                            adam_bend_stable = min(BEND_STABLE_NEED, adam_bend_stable+2)
                            pct = int(adam_bend_stable/BEND_STABLE_NEED*100)
                            smsg,scol = f"✓ Giữ tư thế... {pct}%",(0,220,100)
                            if adam_bend_stable == 6: speak("ADAM_HOLD")
                            if adam_bend_stable >= BEND_STABLE_NEED:
                                adam_state = "SCANNING"
                                stable_counter = 0
                                speak("ADAM_BEND_OK", force=True)

                    elif adam_state == "SCANNING":
                        bh = y2 - y1
                        bw = x2 - x1
                        ry1 = max(0, y1 + int(bh * 0.05))
                        ry2 = min(H, y1 + int(bh * 0.68))
                        rx1 = max(0, x1 - int(bw * 0.05))
                        rx2 = min(W, x2 + int(bw * 0.05))
                        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(200,0,255),3)
                        mx = (x1+x2)//2
                        # Vẽ đường guide dạng nét đứt để phân biệt với đường spine kết quả
                        for seg_y in range(ry1, ry2, 12):
                            cv2.line(frame,(mx,seg_y),(mx,min(seg_y+7,ry2)),(255,200,0),2)

                        if report_data["ADAM"]["status"]=="Chua do":
                            crop_check = frame[ry1:ry2, rx1:rx2]
                            brightness = float(np.mean(cv2.cvtColor(crop_check, cv2.COLOR_BGR2GRAY))) if crop_check.size > 0 else 0
                            if brightness < 60:
                                smsg,scol = f"⚠ Ánh sáng quá tối ({brightness:.0f}) — Bật đèn thêm!",(0,60,255)
                                frame = putVn(frame, "MiDaS cần đủ sáng để phân tích chính xác", (W//2-260, H//2), (0,100,255), 22)
                            else:
                                smsg,scol = "🔬 MiDaS đang phân tích cột sống...",(200,80,255)

                                # Dùng center của bounding box làm spine hint.
                                # Đây là ước lượng ổn định và nhất quán với đường guide vàng.
                                # Shoulder midpoint hay bị sai khi người cúi 90 độ.
                                spine_hint_x = float(mx - rx1)

                                colored, overlay, asym = pose_detector.make_depth_images(
                                    frame,
                                    yolo_box=(rx1,ry1,rx2,ry2) if has_box else None,
                                    spine_hint_x=spine_hint_x)
                                _, raw_buf = cv2.imencode('.jpg', frame)
                                report_data["ADAM"] = {
                                    "asym_index": asym,
                                    "status":     "Da hoan thanh",
                                    "depth_b64":  _b64(colored),
                                    "overlay_b64":_b64(overlay),
                                    "raw_b64":    base64.b64encode(raw_buf).decode(),
                                }
                                adam_state = "DONE"
                                stable_counter = 0
                                speak("ADAM_DONE", force=True)
                        else:
                            adam_state = "DONE"

                    elif adam_state == "DONE":
                        asym = report_data["ADAM"].get("asym_index",0)
                        warn = asym > 10
                        smsg = f"✅ Hoàn thành! Độ lệch: {asym:.1f}% — {'⚠ CẢNH BÁO VẸO!' if warn else 'Bình thường'}"
                        scol = (0,60,255) if warn else (0,220,100)
                        stable_counter += 1
                        if auto_advance and stable_counter > 90:
                            _switch_mode("REPORT"); speak("REPORT_READY", force=True)

        # ──────────────────────────────────────────────
        #  REPORT
        # ──────────────────────────────────────────────
        elif current_mode == "REPORT":
            ov = frame.copy()
            cv2.rectangle(ov,(30,30),(W-30,H-30),(5,10,20),-1)
            frame = cv2.addWeighted(ov,0.88,frame,0.12,0)
            def rline(txt,y,col=(220,220,220),sz=20):
                nonlocal frame
                frame = putVn(frame,txt,(60,y),col,sz)
            rline("BÁO CÁO LÂM SÀNG — BME HUST",50,(80,220,255),24)
            cv2.line(frame,(60,85),(W-60,85),(80,80,100),1)
            f=report_data["FRONT"]
            rline(f"[FRONT]  {f['status']}",105,(180,180,255),20)
            if f["status"]=="Da hoan thanh":
                kl="⚠ CẢNH BÁO LỆCH VAI" if f["sh_angle"]>TWIST_LIM else "Bình thường"
                rline(f"  Góc lệch vai: {f['sh_angle']:.1f}°",130,(200,200,200),18)
                rline(f"  Kết luận: {kl}",152,(0,60,220) if "CẢNH" in kl else (0,200,80),18)
            s=report_data["SIDE"]
            rline(f"[SIDE]   {s['status']}",190,(180,180,255),20)
            if s["status"]=="Da hoan thanh":
                kyph_angle = s.get("kyphotic_angle", 0.0)
                kl="⚠ CẢNH BÁO GÙ LƯNG" if kyph_angle > KYPHOSIS_THRESH else "Bình thường"
                rline(f"  Góc gù lưng: {kyph_angle:.1f}° (ngưỡng: {KYPHOSIS_THRESH:.0f}°)",215,(200,200,200),18)
                rline(f"  Kết luận: {kl}",237,(0,60,220) if "CẢNH" in kl else (0,200,80),18)
            a=report_data["ADAM"]
            rline(f"[ADAM]   {a['status']}",275,(180,180,255),20)
            if a["status"]=="Da hoan thanh":
                kl="⚠ NGUY CƠ VẸO CỘT SỐNG!" if a["asym_index"]>10 else "Bình thường"
                rline(f"  Độ lệch 2 bên: {a['asym_index']:.1f}%",300,(200,200,200),18)
                rline(f"  Kết luận: {kl}",322,(0,60,220) if "NGUY" in kl else (0,200,80),18)

        # ── HUD status bar ─────────────────────────────
        pad = 8
        cv2.rectangle(frame,(10,8),(len(smsg)*11+20+pad*2,38),(10,10,25),-1)
        frame = putVn(frame, smsg, (14+pad,12), scol, 20)

        # ── HUD guide bar ──────────────────────────────
        guide = ""
        if current_mode=="FRONT":
            guide="HƯỚNG DẪN: Đứng thẳng, thả lỏng 2 tay, mắt nhìn vào camera"
        elif current_mode=="SIDE":
            guide="HƯỚNG DẪN: Xoay nghiêng người 90°, đứng thẳng, 2 tay thả lỏng"
        elif current_mode=="ADAM":
            guide={
                "WAIT_BACK":"HƯỚNG DẪN: Xoay LƯNG về phía camera, đứng thẳng, 2 tay thả lỏng",
                "WAIT_BEND":"HƯỚNG DẪN: Cúi gập người về trước cho đến khi lưng song song mặt đất",
                "SCANNING": "HƯỚNG DẪN: Giữ nguyên tư thế cúi, AI đang phân tích...",
                "DONE":     "Phân tích hoàn tất. Xem ảnh depth map trong dashboard.",
            }.get(adam_state,"")
        if guide:
            cv2.rectangle(frame,(10,H-44),(len(guide)*9+30,H-10),(10,10,25),-1)
            frame = putVn(frame, guide, (18,H-38), (255,220,100), 17)

        _, buf = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

# ══════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/poll_audio")
def poll_audio():
    global _pending_file
    with _pending_lock:
        f, _pending_file = _pending_file, ""
    return {"file": f}

@app.get("/audio/{fn}")
def serve_audio(fn: str):
    p = os.path.join(AUDIO_DIR, fn)
    if os.path.exists(p):
        with open(p,"rb") as f: return Response(content=f.read(), media_type="audio/mpeg")
    return {"error":"not found"}

@app.get("/set_mode/{mode}")
def set_mode(mode: str):
    _switch_mode(mode)
    if mode=="ADAM": speak("ADAM_BACK", force=True)
    return {"mode": mode}

@app.get("/reset")
def reset():
    global report_data
    report_data = {
        "FRONT":{"sh_angle":0,"shift_ratio":0,"status":"Chua do","cap_b64":""},
        "SIDE": {"torso_tilt":0,"neck_ratio":0,"status":"Chua do","cap_b64":""},
        "ADAM": {"asym_index":0,"status":"Chua do","depth_b64":"","overlay_b64":"","raw_b64":""},
    }
    _switch_mode("FRONT")
    return {"status":"reset"}

@app.get("/status")
def status():
    vel = round(sum(adam_vel_buf)/len(adam_vel_buf),1) if adam_vel_buf else 0
    return {
        "mode":current_mode, "adam_state":adam_state,
        "adam_bend_stable":adam_bend_stable, "adam_velocity":vel,
        "front":report_data["FRONT"]["status"],
        "side": report_data["SIDE"]["status"],
        "adam": report_data["ADAM"]["status"],
        "adam_asym": report_data["ADAM"].get("asym_index",0),
        "camera": cam_thread is not None and cam_thread.opened,
    }

@app.get("/report_data")
def get_report_data():
    return report_data

@app.get("/generate_report")
def generate_report(
    name: str = "", age: str = "", gender: str = "",
    height: str = "", weight: str = ""
):
    """Tạo và trả về PDF báo cáo lâm sàng."""
    try:
        import io as _io
        from datetime import datetime as _dt
        from fastapi.responses import StreamingResponse as _SR
        
        # --- XÓA DẤU TIẾNG VIỆT ĐỂ TẠO TÊN FILE AN TOÀN (ASCII only) ---
        def to_ascii_filename(input_str):
            if not input_str: return "patient"
            import unicodedata as _ud
            # Bước 1: thay Đ/đ trước (không phân tách được qua NFKD)
            s = input_str.replace("Đ", "D").replace("đ", "d")
            # Bước 2: NFKD để tách dấu khỏi chữ
            s = _ud.normalize('NFKD', s)
            # Bước 3: encode ASCII, bỏ mọi ký tự không phải ASCII
            s = s.encode('ascii', 'ignore').decode('ascii')
            # Bước 4: thay khoảng trắng bằng _
            s = s.replace(" ", "_").strip("_")
            return s if s else "patient"

        safe_gender_text = to_ascii_filename(gender)
        file_prefix      = to_ascii_filename(name)
        
        rg = _get_report_gen()
        # Truyền tên gốc tiếng Việt vào PDF để hiển thị đúng
        patient = {
            "name":   name    or "—", 
            "age":    age     or "—", 
            "gender": gender  or "—",
            "height": height  or "—", 
            "weight": weight  or "—",
        }
        
        pdf_bytes = rg.generate_pdf_report(report_data, patient)
        buf = _io.BytesIO(pdf_bytes)
        ts  = _dt.now().strftime("%Y%m%d_%H%M%S")
        
        safe_filename = f"bme_hust_{file_prefix}_{ts}.pdf"
        
        # Dùng RFC 5987 để hỗ trợ filename UTF-8 nếu cần
        encoded_name = safe_filename.encode('ascii', 'ignore').decode('ascii')
        return _SR(buf, media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={encoded_name}"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}


# ══════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BME HUST</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  :root{--accent:#38bdf8;--ok:#22c55e;--warn:#f97316;--danger:#ef4444}
  body{background:#080c14;font-family:'Segoe UI',sans-serif}
  .glass{background:rgba(255,255,255,.04);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.08)}
  .btn{transition:all .2s}.btn:hover{transform:translateY(-2px);filter:brightness(1.15)}
  .step-active{border-color:#38bdf8!important;background:rgba(56,189,248,.12)!important}
  .step-done{border-color:#22c55e!important;background:rgba(34,197,94,.10)!important}
  .step-pending{border-color:rgba(255,255,255,.1)}
  @keyframes ping2{0%,100%{opacity:1}50%{opacity:.4}}.blink{animation:ping2 1.2s infinite}
  #depth-panel{display:none}#depth-panel.show{display:block}
  .dtab{cursor:pointer;transition:all .2s}.dtab.active{background:rgba(56,189,248,.2);border-color:#38bdf8!important;color:#38bdf8}
  .metric-card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:12px 16px}
  #adam-modal{transition:opacity .25s}
  .vel-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
  #patient-modal{transition:opacity .25s}
  .form-input{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);border-radius:8px;padding:8px 12px;color:#e2e8f0;font-size:13px;width:100%;outline:none;transition:border .2s}
  .form-input:focus{border-color:#38bdf8}
  .form-label{font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:4px;display:block}
</style>
</head>
<body class="text-gray-100 h-screen flex flex-col overflow-hidden">

<div id="start-overlay" class="fixed inset-0 z-[100] bg-gray-900/95 flex flex-col items-center justify-center backdrop-blur-sm">
  <div class="w-16 h-16 rounded-2xl bg-sky-500/20 border border-sky-500/50 flex items-center justify-center mb-6">
    <svg class="w-8 h-8 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
    </svg>
  </div>
  <h1 class="text-3xl font-bold text-white mb-2">BME HUST</h1>
  <p class="text-gray-400 mb-8 text-sm">Hệ thống cần quyền phát âm thanh cảnh báo y tế</p>
  <button onclick="unlockSystem()" class="px-8 py-3.5 bg-sky-600 hover:bg-sky-500 text-white rounded-xl font-bold text-sm shadow-lg transition-all hover:-translate-y-1">
    BẤM VÀO ĐÂY ĐỂ BẮT ĐẦU
  </button>
</div>

<div id="adam-modal" class="hidden fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
 <div class="glass rounded-2xl max-w-xl w-full p-6 border border-purple-500/30 shadow-2xl">
  <div class="flex items-center gap-3 mb-5">
   <div class="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/40 flex items-center justify-center text-purple-300 font-bold text-lg">A</div>
   <div>
    <h2 class="font-bold text-purple-200 text-lg">Adam Forward Bending Test</h2>
    <p class="text-xs text-gray-400">MiDaS depth map • phát hiện vẹo cột sống</p>
   </div>
  </div>
  <div class="grid grid-cols-3 gap-3 mb-5">
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 1</div>
    <p class="text-xs text-gray-300 font-medium">Quay lưng vào camera</p>
    <p class="text-xs text-gray-500 mt-1">Đứng thẳng, tay thả lỏng</p>
   </div>
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 2</div>
    <p class="text-xs text-gray-300 font-medium">Cúi ~90° — lưng nằm ngang</p>
    <p class="text-xs text-gray-500 mt-1">AI dùng MiDaS khi YOLO mất tracking</p>
   </div>
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 3</div>
    <p class="text-xs text-gray-300 font-medium">AI phân tích depth map</p>
    <p class="text-xs text-gray-500 mt-1">Đo độ lệch trái/phải lưng</p>
   </div>
  </div>
  <div class="bg-yellow-900/30 border border-yellow-600/30 rounded-xl p-3 mb-5 text-xs text-yellow-200/80">
   <div class="font-semibold text-yellow-400 mb-1">⚠ Trước khi bắt đầu</div>
   <div class="grid grid-cols-2 gap-1">
    <div>• Cuốn áo lên, để lộ lưng</div><div>• Đứng cách camera 1.5–2m</div>
    <div>• Cúi đến khi lưng nằm ngang</div><div>• Giữ yên đến khi AI xác nhận</div>
   </div>
  </div>
  <div class="flex gap-3">
   <button onclick="closeModal()" class="btn flex-1 py-2.5 rounded-xl border border-gray-600/50 text-gray-400 hover:bg-white/5 text-sm font-semibold">Hủy</button>
   <button onclick="startAdam()" class="btn flex-[2] py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-sm shadow-lg">BẮT ĐẦU →</button>
  </div>
 </div>
</div>

<div id="patient-modal" class="hidden fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
 <div class="glass rounded-2xl max-w-md w-full p-6 border border-emerald-500/30 shadow-2xl">
  <div class="flex items-center gap-3 mb-5">
   <div class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-300 text-lg">📋</div>
   <div>
    <h2 class="font-bold text-emerald-200 text-lg">Thông tin bệnh nhân</h2>
    <p class="text-xs text-gray-400">Điền thông tin trước khi xuất báo cáo PDF</p>
   </div>
  </div>

  <div class="flex flex-col gap-3 mb-5">
   <div>
    <label class="form-label">Họ và tên *</label>
    <input id="pt-name" type="text" class="form-input" placeholder="Nguyễn Văn A">
   </div>
   <div class="grid grid-cols-2 gap-3">
    <div>
     <label class="form-label">Tuổi</label>
     <input id="pt-age" type="number" class="form-input" placeholder="25" min="1" max="120">
    </div>
    <div>
     <label class="form-label">Giới tính</label>
     <select id="pt-gender" class="form-input">
      <option value="">— Chọn —</option>
      <option value="Nam">Nam</option>
      <option value="Nữ">Nữ</option>
      <option value="Khác">Khác</option>
     </select>
    </div>
   </div>
   <div class="grid grid-cols-2 gap-3">
    <div>
     <label class="form-label">Chiều cao (cm)</label>
     <input id="pt-height" type="number" class="form-input" placeholder="170" min="50" max="250">
    </div>
    <div>
     <label class="form-label">Cân nặng (kg)</label>
     <input id="pt-weight" type="number" class="form-input" placeholder="65" min="10" max="300">
    </div>
   </div>
  </div>

  <div class="bg-sky-900/20 border border-sky-500/20 rounded-xl p-3 mb-4 text-xs text-sky-200/70">
   Thông tin chỉ dùng để in báo cáo, không được lưu lại trên hệ thống. (Tên sẽ tự động chuyển thành không dấu để tương thích đa nền tảng).
  </div>

  <div class="flex gap-3">
   <button onclick="closePatientModal()" class="btn flex-1 py-2.5 rounded-xl border border-gray-600/50 text-gray-400 hover:bg-white/5 text-sm font-semibold">Hủy</button>
   <button onclick="submitPatientAndDownload()" class="btn flex-[2] py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-sm shadow-lg">
    Xác nhận &amp; Tải PDF →
   </button>
  </div>
 </div>
</div>

<header class="flex items-center justify-between px-5 py-3 glass border-b border-white/5 flex-shrink-0">
  <div class="flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg bg-sky-500/20 border border-sky-500/40 flex items-center justify-center">
      <svg class="w-4 h-4 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
      </svg>
    </div>
    <span class="font-bold tracking-wider text-sky-300 text-sm uppercase">BME HUST — Clinical AI</span>
  </div>
  <div class="flex items-center gap-2 text-xs">
    <span id="cam-badge" class="px-2 py-1 rounded-md font-semibold bg-yellow-500/20 text-yellow-300 border border-yellow-500/30 blink">CAM...</span>
    <span id="src-badge" class="px-2 py-1 rounded-md font-semibold bg-gray-500/20 text-gray-400 border border-gray-600/30 hidden">SRC —</span>
    <span id="tts-badge" class="px-2 py-1 rounded-md font-semibold bg-gray-500/20 text-gray-400 border border-gray-600/30">TTS</span>
  </div>
</header>

<audio id="tts-player" autoplay style="display:none"></audio>

<div class="flex flex-1 overflow-hidden p-3 gap-3 min-h-0">

 <div class="flex-1 flex flex-col gap-3 min-w-0">

  <div class="flex-1 glass rounded-2xl overflow-hidden relative min-h-0">
   <img id="vid" src="/video_feed" class="w-full h-full object-contain">
   <div class="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2 flex-nowrap">
    <div id="s-front"  class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all whitespace-nowrap">① MẶT TRƯỚC</div>
    <div class="text-gray-600 flex items-center">›</div>
    <div id="s-side"   class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all whitespace-nowrap">② MẶT NGHIÊNG</div>
    <div class="text-gray-600 flex items-center">›</div>
    <div id="s-adam"   class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all whitespace-nowrap">③ ADAM TEST</div>
    <div class="text-gray-600 flex items-center">›</div>
    <div id="s-report" class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all whitespace-nowrap">④ BÁO CÁO</div>
   </div>
  </div>

  <div id="depth-panel" class="glass rounded-2xl p-4 flex-shrink-0">
   <div class="flex items-center justify-between mb-3">
    <div class="text-xs text-gray-500 font-semibold uppercase tracking-widest">Kết quả MiDaS — Adam Test</div>
    <div class="flex gap-2">
     <button class="dtab active text-xs px-3 py-1 rounded-lg border border-white/10" onclick="showTab('depth')"   id="tab-depth">Depth Map</button>
     <button class="dtab        text-xs px-3 py-1 rounded-lg border border-white/10" onclick="showTab('overlay')" id="tab-overlay">Overlay</button>
     <button class="dtab        text-xs px-3 py-1 rounded-lg border border-white/10" onclick="showTab('raw')"     id="tab-raw">Ảnh gốc</button>
    </div>
   </div>
   <div class="flex gap-3">
    <div class="flex-1 bg-black/30 rounded-xl overflow-hidden" style="max-height:200px">
     <img id="depth-img" src="" class="w-full h-full object-contain" style="max-height:200px">
    </div>
    <div class="w-44 flex flex-col gap-2 flex-shrink-0">
     <div class="metric-card">
      <div class="text-xs text-gray-500 mb-1">Độ lệch trái/phải</div>
      <div id="metric-asym" class="text-2xl font-bold text-white">—</div>
      <div id="metric-kl"   class="text-xs mt-1 text-gray-400">—</div>
     </div>
     <div class="metric-card">
      <div class="text-xs text-gray-500 mb-1">Ngưỡng cảnh báo</div>
      <div class="text-xs font-semibold text-amber-400">&gt; 10% → nguy cơ vẹo</div>
     </div>
     <div class="metric-card">
      <div class="text-xs text-gray-500 mb-1">Phương pháp</div>
      <div class="text-xs text-gray-300">Tri-Signal Scoring<br>Dynamic Spine Detection</div>
     </div>
    </div>
   </div>
   <div class="mt-3 grid grid-cols-2 gap-3">
    <div><div class="text-xs text-gray-500 mb-1">Depth map</div>
     <img id="depth-small" src="" class="w-full rounded-lg object-contain" style="max-height:100px"></div>
    <div><div class="text-xs text-gray-500 mb-1">Overlay</div>
     <img id="overlay-small" src="" class="w-full rounded-lg object-contain" style="max-height:100px"></div>
   </div>
  </div>

 </div>

 <div class="w-72 flex flex-col gap-3 flex-shrink-0">

  <div class="glass rounded-2xl p-4 flex flex-col gap-2 flex-shrink-0">
   <div class="text-xs text-gray-500 font-semibold uppercase tracking-widest mb-1">Điều khiển</div>
   <button onclick="setMode('FRONT')" class="btn w-full py-2.5 px-4 rounded-xl bg-sky-600/80 hover:bg-sky-500 text-sm font-bold text-left flex items-center gap-2">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">①</span>Đo mặt trước
   </button>
   <button onclick="setMode('SIDE')" class="btn w-full py-2.5 px-4 rounded-xl bg-sky-600/80 hover:bg-sky-500 text-sm font-bold text-left flex items-center gap-2">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">②</span>Đo mặt nghiêng
   </button>
   <button onclick="openModal()" class="btn w-full py-2.5 px-4 rounded-xl bg-purple-600/80 hover:bg-purple-500 text-sm font-bold text-left flex items-center gap-2 border border-purple-500/30">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">③</span>
    <span>Quét lưng — Adam Test<br><span class="text-xs font-normal opacity-60">Xem hướng dẫn trước</span></span>
   </button>
   <button onclick="setMode('REPORT')" class="btn w-full py-2.5 px-4 rounded-xl bg-amber-500/80 hover:bg-amber-400 text-sm font-bold text-left flex items-center gap-2 text-black mt-1">
    <span class="w-5 h-5 rounded-full bg-black/20 flex items-center justify-center text-xs">④</span>Xem báo cáo nhanh
   </button>
   <button onclick="openPatientModal()" id="btn-pdf" class="btn w-full py-2.5 px-4 rounded-xl bg-emerald-600/80 hover:bg-emerald-500 text-sm font-bold text-left flex items-center gap-2 mt-1" disabled>
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">⬇</span>
    <span id="btn-pdf-txt">Tải báo cáo PDF (Lâm sàng)</span>
   </button>
   <button onclick="resetAll()" class="btn w-full py-2 px-4 rounded-xl border border-red-500/40 text-red-400 hover:bg-red-500/10 text-xs font-semibold mt-1">
    ↺ Reset dữ liệu
   </button>
  </div>

  <div class="glass rounded-2xl p-4 flex-1 overflow-y-auto">
   <div class="text-xs text-gray-500 font-semibold uppercase tracking-widest mb-3">Tiến trình</div>
   <div class="flex flex-col gap-2.5">
    <div class="flex items-center gap-2">
     <div id="dot-front" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div><div class="text-xs font-semibold text-gray-300">Mặt trước</div>
      <div id="txt-front" class="text-xs text-gray-500">Chưa đo</div></div>
    </div>
    <div class="flex items-center gap-2">
     <div id="dot-side" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div><div class="text-xs font-semibold text-gray-300">Mặt nghiêng</div>
      <div id="txt-side" class="text-xs text-gray-500">Chưa đo</div></div>
    </div>
    <div class="flex items-center gap-2">
     <div id="dot-adam" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div><div class="text-xs font-semibold text-gray-300">Adam Test</div>
      <div id="txt-adam" class="text-xs text-gray-500">Chưa đo</div></div>
    </div>
   </div>

   <div id="adam-debug" class="hidden mt-4 p-3 rounded-xl bg-purple-900/20 border border-purple-500/20 text-xs">
    <div class="font-semibold text-purple-300 mb-2">Adam — realtime</div>
    <div class="flex flex-col gap-1 text-gray-400">
     <div>State: <span id="dbg-state" class="text-purple-300 font-semibold">—</span></div>
     <div class="flex items-center gap-1">
      <span class="vel-dot bg-gray-500" id="vel-dot"></span>
      Source: <span id="dbg-src" class="ml-1 font-semibold">—</span>
     </div>
     <div>Depth diff: <span id="dbg-diff">—</span></div>
     <div>Giữ yên: <span id="dbg-stable">—</span> / 45f</div>
    </div>
   </div>

   <div id="adam-guide" class="hidden mt-3 p-3 rounded-xl bg-purple-900/30 border border-purple-500/20 text-xs text-purple-200/80">
    <div class="font-semibold text-purple-300 mb-1">Hướng dẫn</div>
    <div id="adam-step-txt" class="leading-relaxed"></div>
   </div>

   <div class="mt-4 pt-3 border-t border-white/5">
    <div class="text-xs text-gray-500">Mode hiện tại</div>
    <div id="cur-mode" class="text-sm font-bold text-sky-400 mt-0.5">FRONT</div>
   </div>
  </div>

 </div>
</div>

<script>
let audioEnabled=false, depthData={}, curTab='depth';
const player=document.getElementById('tts-player');

function unlockSystem(){
  audioEnabled=true;
  const ov=document.getElementById('start-overlay');
  ov.style.opacity='0';ov.style.transition='opacity .5s';
  setTimeout(()=>ov.remove(),500);
  player.src="data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=";
  player.play().catch(()=>{});
}

setInterval(()=>{
  fetch('/poll_audio').then(r=>r.json()).then(d=>{
    if(d.file&&audioEnabled){
      player.src='/audio/'+d.file+'?t='+Date.now();
      player.play().catch(()=>{});
      const b=document.getElementById('tts-badge');
      b.textContent='🔊 TTS';
      b.className='px-2 py-1 rounded-md font-semibold bg-green-500/20 text-green-300 border border-green-500/30';
    }
  });
},500);

function updateStatus(){
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('cur-mode').textContent=d.mode;
    ['front','side','adam'].forEach((s,i)=>{
      const keys=[d.front,d.side,d.adam];
      const done=keys[i]==='Da hoan thanh';
      const active=d.mode.toLowerCase()===s||(s==='adam'&&d.mode==='ADAM');
      document.getElementById('dot-'+s).className='w-2.5 h-2.5 rounded-full flex-shrink-0 '+(done?'bg-green-400':active?'bg-sky-400 blink':'bg-gray-600');
      document.getElementById('s-'+s).className='px-3 py-1.5 rounded-full border text-xs font-semibold transition-all whitespace-nowrap '+(done?'step-done':active?'step-active':'step-pending');
    });
    document.getElementById('txt-front').textContent=d.front==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';
    document.getElementById('txt-side').textContent =d.side ==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';
    document.getElementById('txt-adam').textContent =d.adam ==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';

    const cb=document.getElementById('cam-badge');
    cb.textContent=d.camera?'CAM ✓':'CAM ✗';
    cb.className=d.camera
      ?'px-2 py-1 rounded-md font-semibold bg-green-500/20 text-green-300 border border-green-500/30'
      :'px-2 py-1 rounded-md font-semibold bg-red-500/20 text-red-300 border border-red-500/30 blink';

    if(d.mode==='ADAM'){
      document.getElementById('adam-debug').classList.remove('hidden');
      document.getElementById('adam-guide').classList.remove('hidden');
      document.getElementById('src-badge').classList.remove('hidden');
      document.getElementById('dbg-state').textContent  =d.adam_state;
      document.getElementById('dbg-stable').textContent =d.adam_bend_stable;
      const guides={
        WAIT_BACK:'📷 Quay LƯNG vào camera, đứng thẳng. Giữ yên để AI xác nhận tư thế.',
        WAIT_BEND:'⬇ Cúi người về phía trước cho đến khi lưng song song mặt đất. Hệ thống dùng MiDaS depth khi YOLO mất tracking.',
        SCANNING: '🔬 Đang phân tích... Giữ nguyên tư thế cúi.',
        DONE:     '✅ Hoàn thành! Xem ảnh depth map bên trái.',
      };
      document.getElementById('adam-step-txt').textContent=guides[d.adam_state]||'';
    } else {
      document.getElementById('adam-debug').classList.add('hidden');
      document.getElementById('adam-guide').classList.add('hidden');
      document.getElementById('src-badge').classList.add('hidden');
    }

    // Enable PDF button khi có ít nhất 1 kết quả
    const anyDone = d.front==='Da hoan thanh' || d.side==='Da hoan thanh' || d.adam==='Da hoan thanh';
    const pdfBtn = document.getElementById('btn-pdf');
    if (pdfBtn) pdfBtn.disabled = !anyDone;

    if(d.adam==='Da hoan thanh'){
      if(!depthData.loaded) loadDepthResults();
      // Đảm bảo panel luôn hiển thị nếu đã có kết quả (trường hợp refresh trang)
      document.getElementById('depth-panel').classList.add('show');
    }
  });
}
setInterval(updateStatus,800);
updateStatus();

function loadDepthResults(){
  fetch('/report_data').then(r=>r.json()).then(data=>{
    const a=data.ADAM;
    if(!a.depth_b64) return;
    depthData={
      depth:  'data:image/png;base64,'+a.depth_b64,
      overlay:'data:image/png;base64,'+a.overlay_b64,
      raw:    'data:image/png;base64,'+a.raw_b64,
      asym:   a.asym_index, loaded:true,
    };
    document.getElementById('depth-panel').classList.add('show');
    document.getElementById('depth-img').src    =depthData.depth;
    document.getElementById('depth-small').src  =depthData.depth;
    document.getElementById('overlay-small').src=depthData.overlay;
    const warn=(a.asym_index||0)>10;
    const el=document.getElementById('metric-asym');
    el.textContent=(a.asym_index||0).toFixed(1)+'%';
    el.className='text-2xl font-bold '+(warn?'text-red-400':'text-green-400');
    const kl=document.getElementById('metric-kl');
    kl.textContent=warn?'⚠ Nguy cơ vẹo cột sống cao':'✓ Trong giới hạn bình thường';
    kl.className='text-xs mt-1 '+(warn?'text-red-300':'text-green-300');
  });
}

function showTab(tab){
  curTab=tab;
  if(!depthData.loaded) return;
  const map={depth:depthData.depth,overlay:depthData.overlay,raw:depthData.raw};
  document.getElementById('depth-img').src=map[tab];
  ['depth','overlay','raw'].forEach(t=>document.getElementById('tab-'+t).classList.toggle('active',t===tab));
}

function resetAll(){
  fetch('/reset').then(()=>{
    updateStatus();
    depthData = {};
    document.getElementById('btn-pdf').disabled = true;
    const dp = document.getElementById('depth-panel');
    if(dp){ dp.classList.remove('show'); }
    document.getElementById('depth-img').src    = '';
    document.getElementById('depth-small').src  = '';
    document.getElementById('overlay-small').src= '';
    document.getElementById('metric-asym').textContent = '—';
    document.getElementById('metric-kl').textContent   = '—';
    document.getElementById('metric-asym').className   = 'text-2xl font-bold text-white';
    ['pt-name','pt-age','pt-height','pt-weight'].forEach(id=>{
      const el = document.getElementById(id);
      if(el) el.value = '';
    });
    const sel = document.getElementById('pt-gender');
    if(sel) sel.value = '';
  });
}

function openPatientModal(){
  document.getElementById('patient-modal').classList.remove('hidden');
  setTimeout(()=>document.getElementById('pt-name').focus(), 100);
}
function closePatientModal(){
  document.getElementById('patient-modal').classList.add('hidden');
}
function submitPatientAndDownload(){
  const name   = document.getElementById('pt-name').value.trim();
  const age    = document.getElementById('pt-age').value.trim();
  const gender = document.getElementById('pt-gender').value;
  const height = document.getElementById('pt-height').value.trim();
  const weight = document.getElementById('pt-weight').value.trim();

  if(!name){
    document.getElementById('pt-name').focus();
    document.getElementById('pt-name').style.borderColor='#ef4444';
    return;
  }
  document.getElementById('pt-name').style.borderColor='';
  closePatientModal();

  const params = new URLSearchParams({name,age,gender,height,weight});
  const btn = document.getElementById('btn-pdf');
  const txt = document.getElementById('btn-pdf-txt');
  txt.textContent = 'Đang tạo PDF...';
  btn.disabled = true;

  fetch('/generate_report?' + params.toString())
    .then(r=>{ if(!r.ok) throw new Error('Server error'); return r.blob(); })
    .then(blob=>{
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      // Khử dấu ở tên file tải xuống bên phía client luôn cho chắc ăn
      let safe_client_name = (name||'patient').normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/\s+/g,'_');
      a.download = `bme_hust_${safe_client_name}_report.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      txt.textContent = 'Tải báo cáo PDF (Lâm sàng)';
      btn.disabled = false;
    })
    .catch(e=>{
      txt.textContent = 'Lỗi: '+e.message;
      btn.disabled = false;
    });
}

document.addEventListener('DOMContentLoaded', ()=>{
  document.getElementById('patient-modal').addEventListener('click', e=>{
    if(e.target===e.currentTarget) closePatientModal();
  });
});

function setMode(m){
  audioEnabled=true;
  // Reset rõ ràng: xoá cache depth data và ẩn panel cũ
  depthData = { loaded: false };
  document.getElementById('depth-panel').classList.remove('show');
  document.getElementById('depth-img').src     = '';
  document.getElementById('depth-small').src   = '';
  document.getElementById('overlay-small').src = '';
  fetch('/set_mode/'+m).then(()=>updateStatus());
}
function openModal(){audioEnabled=true;document.getElementById('adam-modal').classList.remove('hidden');}
function closeModal(){document.getElementById('adam-modal').classList.add('hidden');}
function startAdam(){closeModal();setMode('ADAM');}
document.getElementById('adam-modal').addEventListener('click',e=>{if(e.target===e.currentTarget)closeModal();});
</script>
</body>
</html>"""

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, reload=False)