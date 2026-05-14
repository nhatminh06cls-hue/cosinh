from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
from contextlib import asynccontextmanager
from ultralytics import YOLO  
from PIL import ImageFont, ImageDraw, Image 
import cv2, math, numpy as np, torch, uvicorn, time, os, threading, queue

# ══════════════════════════════════════════════════════
#  TTS  —  Queue-based, nói hết câu rồi mới nói tiếp
# ══════════════════════════════════════════════════════
try:
    from gtts import gTTS
    TTS_OK = True
except ImportError:
    TTS_OK = False
    print("WARNING: pip install gTTS")

AUDIO_DIR = "/tmp/gk_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

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
    "ADAM_BACK":      "Xin quay lưng vào camera, đứng thẳng, hai tay thả lỏng hai bên.",
    "ADAM_BACK_OK":   "Tốt. Bây giờ xin từ từ cúi người về phía trước, hai tay thả thẳng xuống đất.",
    "ADAM_BEND_OK":   "Giữ nguyên tư thế. Hệ thống đang phân tích cột sống.",
    "ADAM_DONE":      "Hoàn thành kiểm tra Adam. Xem kết quả tại báo cáo.",
    "ADAM_NEAR":      "Xin bước lại gần camera hơn.",
    "ADAM_FAR":       "Xin bước ra xa camera hơn.",
    "REPORT_READY":   "Kiểm tra hoàn tất. Đang hiển thị báo cáo lâm sàng.",
    "WARN_SHOULDER":  "Cảnh báo, phát hiện lệch vai.",
    "WARN_KYPHOSIS":  "Cảnh báo, phát hiện ngã lưng.",
}

_tts_queue   = queue.Queue()   
_playing     = False           
_playing_lock = threading.Lock()
_pending_file = ""             
_pending_lock = threading.Lock()
_cooldown: dict[str, float] = {}   

def _tts_worker():
    global _playing, _pending_file
    while True:
        key = _tts_queue.get()
        if not TTS_OK:
            _tts_queue.task_done()
            continue
        text = SCRIPTS.get(key, "")
        if not text:
            _tts_queue.task_done()
            continue
        fpath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        try:
            if not os.path.exists(fpath):
                gTTS(text=text, lang="vi").save(fpath)
            with _playing_lock:
                _playing = True
            with _pending_lock:
                _pending_file = f"{key}.mp3"
            duration = max(2.0, len(text) * 0.07)
            time.sleep(duration)
        except Exception as e:
            print(f"TTS error [{key}]: {e}")
        finally:
            with _playing_lock:
                _playing = False
            _tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(key: str, force: bool = False):
    now = time.time()
    if not force:
        if _cooldown.get(key, 0) + 5.0 > now:
            return
        with _playing_lock:
            if _playing:          
                return
    _cooldown[key] = now
    if force:
        try:
            while True: _tts_queue.get_nowait(); _tts_queue.task_done()
        except queue.Empty:
            pass
    _tts_queue.put(key)

def _pregenenrate_all():
    for key, text in SCRIPTS.items():
        fpath = os.path.join(AUDIO_DIR, f"{key}.mp3")
        if not os.path.exists(fpath):
            try: gTTS(text=text, lang="vi").save(fpath)
            except: pass

# ══════════════════════════════════════════════════════
#  HÀM VẼ TIẾNG VIỆT
# ══════════════════════════════════════════════════════
def cv2_putText_vn(img, text, position, color=(0, 255, 0), font_size=20):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════
#  AI MODELS
# ══════════════════════════════════════════════════════
model = midas = device = midas_transforms = None

# ══════════════════════════════════════════════════════
#  CAMERA INIT 
# ══════════════════════════════════════════════════════
def init_camera():
    def warm(cap, n=10):
        for _ in range(n):
            r, f = cap.read()
            if r and f is not None: return True
            time.sleep(0.1)
        return False

    print("[CAM 1/3] V4L2 + MJPEG 1280×720 @30fps …")
    try:
        c = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            if warm(c):
                print(f"[CAM OK] V4L2+MJPEG {int(c.get(3))}×{int(c.get(4))}")
                return c
        c.release()
    except Exception as e: print(f"  V4L2 err: {e}")

    print("[CAM 2/3] GStreamer MJPEG …")
    for gst in [
        "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false",
        "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false",
    ]:
        try:
            c = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if c.isOpened() and warm(c):
                print("[CAM OK] GStreamer MJPEG"); return c
            c.release()
        except Exception as e: print(f"  GST err: {e}")

    print("[CAM 3/3] CAP_ANY …")
    try:
        c = cv2.VideoCapture(0, cv2.CAP_ANY)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            if warm(c): print("[CAM OK] CAP_ANY"); return c
        c.release()
    except Exception as e: print(f"  ANY err: {e}")

    print("[CAM FAIL] sudo fuser -k /dev/video0 rồi thử lại")
    return None

# ══════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    global cap, model, midas, device, midas_transforms
    print("=== KHỞI TẠO HỆ THỐNG ===")
    model = YOLO("yolov8m-pose.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device); midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    cap = init_camera()
    if TTS_OK:
        print("Pre-generating TTS …")
        threading.Thread(target=_pregenenrate_all, daemon=True).start()
    yield
    if cap and cap.isOpened(): cap.release(); print("[CAM] released")

app = FastAPI(title="Gatekeeper Omni", lifespan=lifespan)
cap = None

# ══════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════
current_mode   = "FRONT"
stable_counter = 0
pathology_ctr  = 0
prev_pelvis    = None
adam_state     = "WAIT_BACK"   
baseline_h     = 0
auto_advance   = True          

report_data = {
    "FRONT": {"sh_angle": 0, "shift_ratio": 0, "status": "Chua do"},
    "SIDE":  {"torso_tilt": 0, "neck_ratio": 0, "status": "Chua do"},
    "ADAM":  {"asym_index": 0,                  "status": "Chua do"},
}

MOVE_LIM   = 7.0
LOCK_TIME  = 90
TWIST_LIM  = 4.0;  FRONT_TILT = 5.0;  SH_RATIO_MIN = 0.18;  LAT_SHIFT  = 0.10
SIDE_TILT  = 15.0; NECK_MIN   = 0.25; SH_SIDE_MAX  = 0.12
MIN_FILL   = 0.45; MAX_FILL   = 0.92; BEND_TGT     = 0.58
STAND_FRAMES = 60   
BEND_FRAMES  = 45   

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def _switch_mode(new_mode):
    global current_mode, stable_counter, pathology_ctr, prev_pelvis, adam_state, baseline_h
    current_mode   = new_mode
    stable_counter = 0
    pathology_ctr  = 0
    prev_pelvis    = None
    if new_mode == "ADAM":
        adam_state = "WAIT_BACK"
        baseline_h = 0

# ══════════════════════════════════════════════════════
#  VIDEO STREAM
# ══════════════════════════════════════════════════════
def generate_frames():
    global current_mode, stable_counter, pathology_ctr, prev_pelvis
    global adam_state, baseline_h, report_data

    if cap is None or not cap.isOpened():
        blank = np.zeros((480,640,3), np.uint8)
        cv2.putText(blank,"CAMERA OFFLINE",(60,240),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,255),3)
        _, buf = cv2.imencode('.jpg', blank)
        while True:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(1)
        return

    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.05); continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        res = model(frame, verbose=False)
        smsg, scol = "Không thấy bệnh nhân", (0,0,255)
        has_box = has_kp = False
        x1=y1=x2=y2=ch=cw=0

        if res[0].boxes is not None and len(res[0].boxes.xyxy):
            b = res[0].boxes.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int,b[:4])
            ch,cw = y2-y1, x2-x1
            has_box = True

        if res[0].keypoints is not None and len(res[0].keypoints.xy):
            kp = res[0].keypoints.xy[0]
            if len(kp) >= 13:
                le,re = kp[3],kp[4]
                ls,rs = kp[5],kp[6]
                lh,rh = kp[11],kp[12]
                me = ((le[0]+re[0])/2,(le[1]+re[1])/2)
                ms = ((ls[0]+rs[0])/2,(ls[1]+rs[1])/2)
                mh = ((lh[0]+rh[0])/2,(lh[1]+rh[1])/2)
                has_kp = True

        # ─── FRONT ───────────────────────────────────────
        if current_mode == "FRONT":
            cv2.rectangle(frame,(W//6,H//10),(5*W//6,9*H//10),(100,220,255),2)
            if not (has_box and has_kp):
                speak("FRONT_START")
                smsg = "Xin đứng vào khung hình, nhìn thẳng vào camera"
            else:
                sh_ang = math.degrees(math.atan2(abs(ls[1]-rs[1]), abs(ls[0]-rs[0])))
                moved  = dist(mh, prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh
                shw    = abs(ls[0]-rs[0])
                shr    = shw/ch if ch else 0
                latr   = abs(ms[0]-mh[0])/shw if shw else 0
                tdx,tdy = abs(ms[0]-mh[0]), abs(ms[1]-mh[1])
                ttilt  = math.degrees(math.atan2(tdx,tdy)) if tdy else 90

                if moved > MOVE_LIM:
                    smsg,scol = "Xin đứng yên!",  (0,0,255)
                    stable_counter = pathology_ctr = 0; speak("FRONT_MOVE")
                elif shr < SH_RATIO_MIN or latr > LAT_SHIFT:
                    smsg,scol = "Quay mặt thẳng vào camera!", (0,0,255)
                    stable_counter = pathology_ctr = 0; speak("FRONT_WRONG")
                elif ttilt > FRONT_TILT or sh_ang > TWIST_LIM:
                    pathology_ctr += 1; stable_counter = 0
                    pct = int(pathology_ctr/LOCK_TIME*100)
                    smsg,scol = f"⚠ Phát hiện lệch vai! Xác nhận {pct}%", (0,140,255)
                    if pct == 5:
                        speak("WARN_SHOULDER")
                else:
                    stable_counter += 1; pathology_ctr = 0
                    pct = int(stable_counter/30*100)
                    smsg,scol = f"✓ Tư thế chuẩn... {pct}%", (0,220,100)
                    if stable_counter == 8: speak("FRONT_GOOD")

                if stable_counter >= 30 or pathology_ctr >= LOCK_TIME:
                    if report_data["FRONT"]["status"] == "Chua do":
                        report_data["FRONT"] = {"sh_angle":sh_ang,"shift_ratio":latr*100,"status":"Da hoan thanh"}
                        speak("FRONT_DONE", force=True)
                        if stable_counter >= 30: stable_counter = 30
                        if pathology_ctr >= LOCK_TIME: pathology_ctr = LOCK_TIME

                    smsg,scol = "✅ ĐÃ LƯU DỮ LIỆU! Đang chuyển sang bước tiếp theo...", (0,255,120)
                    
                    if stable_counter >= 30: stable_counter += 1
                    if pathology_ctr >= LOCK_TIME: pathology_ctr += 1

                    if auto_advance and (stable_counter > 30 + 90 or pathology_ctr > LOCK_TIME + 90):
                        _switch_mode("SIDE")

                cv2.line(frame,(int(ls[0]),int(ls[1])),(int(rs[0]),int(rs[1])),(0,80,255),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,255),2)

        # ─── SIDE ────────────────────────────────────────
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
                tdx,tdy = abs(ms[0]-mh[0]), abs(ms[1]-mh[1])
                ttilt = math.degrees(math.atan2(tdx,tdy)) if tdy else 90
                moved = dist(mh, prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh

                if moved > MOVE_LIM:
                    smsg,scol = "Xin đứng yên!", (0,0,255)
                    stable_counter = pathology_ctr = 0; speak("SIDE_MOVE")
                elif shr > SH_SIDE_MAX or nkr < NECK_MIN:
                    smsg,scol = "Nghiêng thêm, không quay mặt vào camera!", (0,0,255)
                    stable_counter = pathology_ctr = 0; speak("SIDE_WRONG")
                elif ttilt > SIDE_TILT:
                    pathology_ctr += 1; stable_counter = 0
                    pct = int(pathology_ctr/LOCK_TIME*100)
                    smsg,scol = f"⚠ Phát hiện gù/ngã lưng ({ttilt:.0f}°)! {pct}%", (0,140,255)
                    if pct == 5:
                        speak("WARN_KYPHOSIS")
                else:
                    stable_counter += 1; pathology_ctr = 0
                    pct = int(stable_counter/30*100)
                    smsg,scol = f"✓ Tư thế chuẩn... {pct}%", (0,220,100)
                    if stable_counter == 8: speak("SIDE_GOOD")

                if stable_counter >= 30 or pathology_ctr >= LOCK_TIME:
                    if report_data["SIDE"]["status"] == "Chua do":
                        report_data["SIDE"] = {"torso_tilt":ttilt,"neck_ratio":nkr*100,"status":"Da hoan thanh"}
                        speak("SIDE_DONE", force=True)
                        if stable_counter >= 30: stable_counter = 30
                        if pathology_ctr >= LOCK_TIME: pathology_ctr = LOCK_TIME

                    smsg,scol = "✅ ĐÃ LƯU DỮ LIỆU! Đang chuyển sang bước tiếp theo...", (0,255,120)
                    
                    if stable_counter >= 30: stable_counter += 1
                    if pathology_ctr >= LOCK_TIME: pathology_ctr += 1

                    if auto_advance and (stable_counter > 30 + 90 or pathology_ctr > LOCK_TIME + 90):
                        _switch_mode("ADAM"); speak("ADAM_BACK", force=True)

                cv2.line(frame,(int(me[0]),int(me[1])),(int(ms[0]),int(ms[1])),(255,80,200),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,255),3)

        # ─── ADAM ────────────────────────────────────────
        elif current_mode == "ADAM":
            if not has_box:
                speak("ADAM_BACK")
                smsg,scol = "Xin quay LƯNG vào camera và đứng vào khung hình", (0,180,255)
            else:
                fill = ch / H
                if fill < MIN_FILL:
                    speak("ADAM_NEAR"); smsg,scol="Lại GẦN camera hơn",(0,140,255)
                    stable_counter = 0
                elif fill > MAX_FILL:
                    speak("ADAM_FAR");  smsg,scol="Lui RA XA camera hơn",(0,140,255)
                    stable_counter = 0
                else:
                    box_col = (0,255,120) if adam_state in ("WAIT_BEND","SCANNING","DONE") else (100,180,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),box_col,2)

                    if adam_state == "WAIT_BACK":
                        speak("ADAM_BACK")
                        smsg,scol = "Quay LƯNG vào camera, đứng thẳng...", (100,200,255)
                        if has_kp:
                            shw = abs(ls[0]-rs[0])
                            shr = shw/ch if ch else 0
                            tdx,tdy = abs(ms[0]-mh[0]), abs(ms[1]-mh[1])
                            ttilt = math.degrees(math.atan2(tdx,tdy)) if tdy else 90
                            
                            if shr > 0.16 and ttilt < 20:
                                stable_counter += 1
                                pct = int(stable_counter/STAND_FRAMES*100)
                                smsg,scol = f"✓ Xác nhận lưng vào camera... {pct}%", (0,220,100)
                                if stable_counter >= STAND_FRAMES:
                                    baseline_h = ch
                                    adam_state = "WAIT_BEND"
                                    stable_counter = 0
                                    speak("ADAM_BACK_OK", force=True)
                            else:
                                stable_counter = max(0, stable_counter-2)
                                hint = "→ Xoay lưng thẳng vào camera" if shr <= 0.16 else "→ Đứng THẲNG hơn"
                                frame = cv2_putText_vn(frame, hint, (W//2-180, H-80), (0,200,255), 24)

                    elif adam_state == "WAIT_BEND":
                        smsg,scol = "Từ từ CÚI người về phía trước, tay thả xuống...", (255,220,0)
                        bend_r = ch/baseline_h if baseline_h else 1.0
                        ax = W - 70
                        cv2.arrowedLine(frame,(ax,H//4),(ax,3*H//4),(0,220,255),4,tipLength=0.25)
                        frame = cv2_putText_vn(frame, "CÚI", (ax-30, 3*H//4+10), (0,220,255), 28)

                        if bend_r < BEND_TGT:
                            stable_counter += 1
                            pct = int(stable_counter/BEND_FRAMES*100)
                            smsg,scol = f"✓ Xác nhận góc cúi... {pct}%", (0,220,100)
                            if stable_counter >= BEND_FRAMES:
                                adam_state = "SCANNING"
                                stable_counter = 0
                                speak("ADAM_BEND_OK", force=True)
                        else:
                            stable_counter = max(0, stable_counter-1)

                    elif adam_state == "SCANNING":
                        ry1 = max(0, y1)
                        ry2 = min(H, y1+int(ch*0.75))
                        cv2.rectangle(frame,(x1,ry1),(x2,ry2),(200,0,255),4)
                        mx = (x1+x2)//2
                        cv2.line(frame,(mx,ry1),(mx,ry2),(255,255,0),2)

                        if report_data["ADAM"]["status"] == "Chua do":
                            smsg,scol = "🔬 MiDaS đang phân tích cột sống...", (200,80,255)
                            crop = frame[ry1:ry2, x1:x2]
                            if crop.size > 0:
                                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                inp  = midas_transforms(rgb).to(device)
                                with torch.no_grad():
                                    pred = midas(inp)
                                    pred = torch.nn.functional.interpolate(
                                        pred.unsqueeze(1), size=rgb.shape[:2],
                                        mode="bicubic", align_corners=False).squeeze()
                                dm   = pred.cpu().numpy()
                                mid  = dm.shape[1]//2
                                lm,rm = np.mean(dm[:,:mid]), np.mean(dm[:,mid:])
                                asym = abs(lm-rm)/(max(lm,rm)+1e-6)*100
                                report_data["ADAM"] = {"asym_index":asym,"status":"Da hoan thanh"}
                                adam_state = "DONE"
                                stable_counter = 0
                                speak("ADAM_DONE", force=True)
                        else:
                            if adam_state != "DONE": stable_counter = 0
                            adam_state = "DONE"

                    elif adam_state == "DONE":
                        asym = report_data["ADAM"].get("asym_index",0)
                        warn = asym > 10
                        smsg = f"✅ Hoàn thành! Độ lệch: {asym:.1f}% — {'⚠ CẢNH BÁO VẸO!' if warn else 'Bình thường'} (Đang chuyển...)"
                        scol = (0,60,255) if warn else (0,220,100)
                        stable_counter += 1
                        if auto_advance and stable_counter > 90:
                            _switch_mode("REPORT"); speak("REPORT_READY",force=True)

        # ─── REPORT ──────────────────────────────────────
        elif current_mode == "REPORT":
            ov = frame.copy()
            cv2.rectangle(ov,(30,30),(W-30,H-30),(5,10,20),-1)
            frame = cv2.addWeighted(ov,0.88,frame,0.12,0)

            def rline(txt, y, col=(220,220,220), sz=20):
                nonlocal frame
                frame = cv2_putText_vn(frame, txt, (60, y), col, font_size=sz)

            rline("BÁO CÁO LÂM SÀNG — GATEKEEPER OMNI", 50, (80,220,255), 26)
            cv2.line(frame,(60,90),(W-60,90),(80,80,100),1)

            f=report_data["FRONT"]
            rline(f"[FRONT]  {f['status']}", 110, (180,180,255), 22)
            if f["status"]=="Da hoan thanh":
                kl = "CẢNH BÁO VẸO/LỆCH" if f["sh_angle"]>TWIST_LIM else "Bình thường"
                rline(f"  Lệch vai: {f['sh_angle']:.1f}°", 140, (200,200,200), 20)
                rline(f"  Kết luận: {kl}", 170, (0,60,220) if "CẢNH" in kl else (0,200,80), 20)

            s=report_data["SIDE"]
            rline(f"[SIDE]   {s['status']}", 220, (180,180,255), 22)
            if s["status"]=="Da hoan thanh":
                kl = "CẢNH BÁO GÙ/NGÃ LƯNG" if s["torso_tilt"]>SIDE_TILT else "Bình thường"
                rline(f"  Ngã lưng: {s['torso_tilt']:.1f}°", 250, (200,200,200), 20)
                rline(f"  Kết luận: {kl}", 280, (0,60,220) if "CẢNH" in kl else (0,200,80), 20)

            a=report_data["ADAM"]
            rline(f"[ADAM]   {a['status']}", 330, (180,180,255), 22)
            if a["status"]=="Da hoan thanh":
                kl = "NGUY CƠ VẸO CAO!" if a["asym_index"]>10 else "Bình thường"
                rline(f"  Độ lệch 2 bên: {a['asym_index']:.1f}%", 360, (200,200,200), 20)
                rline(f"  Kết luận: {kl}", 390, (0,60,220) if "NGUY" in kl else (0,200,80), 20)

        # ─── HUD STATUS BAR ────────────────────────────────
        pad=8
        tw, th = len(smsg) * 11, 24
        cv2.rectangle(frame,(10,8),(tw+20+pad*2, th+20),(10,10,25),-1)
        frame = cv2_putText_vn(frame, smsg, (14+pad, 14), scol, font_size=20)

        # ─── HUD GUIDE ─────────────────────────────────────
        guide = ""
        if current_mode == "FRONT": guide = "HƯỚNG DẪN: Đứng thẳng người, thả lỏng 2 tay, mắt nhìn vào camera"
        elif current_mode == "SIDE": guide = "HƯỚNG DẪN: Xoay nghiêng người 90°, đứng thẳng, 2 tay thả lỏng"
        elif current_mode == "ADAM":
            if adam_state == "WAIT_BACK": guide = "HƯỚNG DẪN: Xoay lưng về phía camera, đứng thẳng, 2 tay thả lỏng"
            elif adam_state == "WAIT_BEND": guide = "HƯỚNG DẪN: Từ từ cúi gập người về trước, thả lỏng 2 tay xuống"
            elif adam_state == "SCANNING": guide = "HƯỚNG DẪN: Giữ nguyên tư thế cúi để AI phân tích..."
        if guide:
            gw = len(guide) * 10
            cv2.rectangle(frame,(10, H-45),(gw+40, H-10),(10,10,25),-1)
            frame = cv2_putText_vn(frame, guide, (20, H-38), (255,220,100), font_size=18)

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
        with open(p, "rb") as f:
            return Response(content=f.read(), media_type="audio/mpeg")
    return {"error":"not found"}

@app.get("/set_mode/{mode}")
def set_mode(mode: str):
    _switch_mode(mode)
    if mode == "ADAM": speak("ADAM_BACK", force=True)
    return {"mode": mode}

@app.get("/reset")
def reset():
    global report_data
    report_data = {
        "FRONT": {"sh_angle":0,"shift_ratio":0,"status":"Chua do"},
        "SIDE":  {"torso_tilt":0,"neck_ratio":0,"status":"Chua do"},
        "ADAM":  {"asym_index":0,               "status":"Chua do"},
    }
    _switch_mode("FRONT")
    return {"status":"reset"}

@app.get("/status")
def status():
    return {
        "mode": current_mode,
        "adam_state": adam_state,
        "front": report_data["FRONT"]["status"],
        "side":  report_data["SIDE"]["status"],
        "adam":  report_data["ADAM"]["status"],
        "camera": cap is not None and cap.isOpened(),
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gatekeeper Omni</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  :root { --accent:#38bdf8; --ok:#22c55e; --warn:#f97316; --danger:#ef4444; }
  body { background:#080c14; font-family:'Segoe UI',sans-serif; }
  .glass { background:rgba(255,255,255,.04); backdrop-filter:blur(12px);
           border:1px solid rgba(255,255,255,.08); }
  .glow  { box-shadow:0 0 18px rgba(56,189,248,.25); }
  .btn   { transition:all .2s; }
  .btn:hover { transform:translateY(-2px); filter:brightness(1.15); }
  .btn:active { transform:translateY(0); }
  .step-active   { border-color:#38bdf8 !important; background:rgba(56,189,248,.12) !important; }
  .step-done     { border-color:#22c55e !important; background:rgba(34,197,94,.10) !important; }
  .step-pending  { border-color:rgba(255,255,255,.1); }
  .progress-bar  { transition:width .4s ease; }
  @keyframes ping2 { 0%,100%{opacity:1} 50%{opacity:.4} }
  .blink { animation:ping2 1.2s infinite; }
  #adam-modal { transition:opacity .25s; }
</style>
</head>
<body class="text-gray-100 h-screen flex flex-col overflow-hidden relative">

<div id="start-overlay" class="fixed inset-0 z-[100] bg-gray-900/95 flex flex-col items-center justify-center backdrop-blur-sm transition-opacity duration-500">
  <div class="w-16 h-16 rounded-2xl bg-sky-500/20 border border-sky-500/50 flex items-center justify-center mb-6 glow">
    <svg class="w-8 h-8 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
    </svg>
  </div>
  <h1 class="text-3xl font-bold text-white mb-2">Gatekeeper Omni</h1>
  <p class="text-gray-400 mb-8 text-sm">Hệ thống cần quyền phát âm thanh cảnh báo y tế</p>
  <button onclick="unlockSystem()" class="px-8 py-3.5 bg-sky-600 hover:bg-sky-500 text-white rounded-xl font-bold text-sm tracking-wide shadow-lg shadow-sky-900/50 transition-all hover:-translate-y-1">
    BẤM VÀO ĐÂY ĐỂ BẮT ĐẦU
  </button>
</div>

<header class="flex items-center justify-between px-5 py-3 glass border-b border-white/5">
  <div class="flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg bg-sky-500/20 border border-sky-500/40 flex items-center justify-center">
      <svg class="w-4 h-4 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944
                 a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003
                 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332
                 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
      </svg>
    </div>
    <span class="font-bold tracking-wider text-sky-300 text-sm uppercase">Gatekeeper Omni — Clinical AI</span>
  </div>
  <div class="flex items-center gap-2 text-xs">
    <span id="cam-badge" class="px-2 py-1 rounded-md font-semibold bg-yellow-500/20 text-yellow-300 border border-yellow-500/30 blink">CAM...</span>
    <span id="tts-badge" class="px-2 py-1 rounded-md font-semibold bg-gray-500/20 text-gray-400 border border-gray-600/30">TTS</span>
    <span class="px-2 py-1 rounded-md font-semibold bg-green-500/20 text-green-300 border border-green-500/30">RTX 3070Ti ✓</span>
  </div>
</header>

<audio id="tts-player" autoplay style="display: none;"></audio>

<div id="adam-modal" class="hidden fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
 <div class="glass rounded-2xl max-w-xl w-full p-6 border border-purple-500/30 shadow-2xl">
  <div class="flex items-center gap-3 mb-5">
   <div class="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/40 flex items-center justify-center text-purple-300 font-bold text-lg">A</div>
   <div>
    <h2 class="font-bold text-purple-200 text-lg">Adam Forward Bending Test</h2>
    <p class="text-xs text-gray-400">Phát hiện vẹo cột sống bằng phân tích độ lệch lưng</p>
   </div>
  </div>

  <div class="grid grid-cols-3 gap-3 mb-5">
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 1</div>
    <svg viewBox="0 0 70 130" class="w-12 mx-auto mb-2" fill="none">
     <circle cx="35" cy="16" r="9" fill="#94a3b8"/>
     <line x1="35" y1="25" x2="35" y2="80" stroke="#94a3b8" stroke-width="5" stroke-linecap="round"/>
     <line x1="35" y1="44" x2="14" y2="64" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="35" y1="44" x2="56" y2="64" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="35" y1="80" x2="22" y2="122" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="35" y1="80" x2="48" y2="122" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <text x="35" y="140" font-size="8" fill="#38bdf8" text-anchor="middle">LƯNG vào CAM</text>
    </svg>
    <p class="text-xs text-gray-300 font-medium">Quay lưng vào camera</p>
    <p class="text-xs text-gray-500 mt-1">Đứng thẳng, tay thả lỏng</p>
   </div>
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 2</div>
    <svg viewBox="0 0 90 90" class="w-16 mx-auto mb-2" fill="none">
     <circle cx="12" cy="45" r="8" fill="#94a3b8"/>
     <line x1="19" y1="48" x2="68" y2="35" stroke="#94a3b8" stroke-width="5" stroke-linecap="round"/>
     <line x1="38" y1="42" x2="30" y2="65" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="52" y1="38" x2="48" y2="65" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="68" y1="35" x2="60" y2="80" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <line x1="68" y1="35" x2="78" y2="80" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
     <path d="M20 48 Q44 33 68 35" stroke="#f97316" stroke-width="2" stroke-dasharray="4,3" fill="none"/>
    </svg>
    <p class="text-xs text-gray-300 font-medium">Cúi người 90°</p>
    <p class="text-xs text-gray-500 mt-1">Tay thả thẳng xuống</p>
   </div>
   <div class="rounded-xl p-3 border border-white/10 text-center">
    <div class="text-xs text-purple-400 font-bold mb-2">BƯỚC 3</div>
    <div class="w-14 h-14 mx-auto mb-2 rounded-lg overflow-hidden border border-white/10 flex">
     <div class="flex-1 bg-green-500/50"></div>
     <div class="w-px bg-white/30"></div>
     <div class="flex-1 bg-orange-500/60"></div>
    </div>
    <p class="text-xs text-gray-300 font-medium">AI phân tích</p>
    <p class="text-xs text-gray-500 mt-1">MiDaS đo độ lệch lưng</p>
   </div>
  </div>

  <div class="bg-yellow-900/30 border border-yellow-600/30 rounded-xl p-3 mb-5 text-xs text-yellow-200/80">
   <div class="font-semibold text-yellow-400 mb-1">⚠ Lưu ý trước khi bắt đầu</div>
   <div class="grid grid-cols-2 gap-1">
    <div>• Cuốn áo lên, để lộ lưng</div><div>• Đứng cách camera 1.5–2m</div>
    <div>• Không đeo ba lô, áo rộng</div><div>• Ánh sáng đủ sáng</div>
   </div>
  </div>

  <div class="flex gap-3">
   <button onclick="closeModal()" class="btn flex-1 py-2.5 rounded-xl border border-gray-600/50 text-gray-400 hover:bg-white/5 text-sm font-semibold">Hủy</button>
   <button onclick="startAdam()" class="btn flex-[2] py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-sm shadow-lg shadow-purple-900/40">BẮT ĐẦU KIỂM TRA →</button>
  </div>
 </div>
</div>

<div class="flex flex-1 overflow-hidden p-3 gap-3">

 <div class="flex-1 glass rounded-2xl overflow-hidden relative">
  <img id="vid" src="/video_feed" class="w-full h-full object-contain">
  <div class="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2">
   <div id="s-front" class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all duration-300">① MẶT TRƯỚC</div>
   <div class="text-gray-600 flex items-center">›</div>
   <div id="s-side"  class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all duration-300">② MẶT NGHIÊNG</div>
   <div class="text-gray-600 flex items-center">›</div>
   <div id="s-adam"  class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all duration-300">③ ADAM TEST</div>
   <div class="text-gray-600 flex items-center">›</div>
   <div id="s-report" class="step-pending px-3 py-1.5 rounded-full border text-xs font-semibold transition-all duration-300">④ BÁO CÁO</div>
  </div>
 </div>

 <div class="w-72 flex flex-col gap-3">

  <div class="glass rounded-2xl p-4 flex flex-col gap-2">
   <div class="text-xs text-gray-500 font-semibold uppercase tracking-widest mb-1">Điều khiển</div>

   <button id="btn-front" onclick="setMode('FRONT')"
    class="btn w-full py-2.5 px-4 rounded-xl bg-sky-600/80 hover:bg-sky-500 text-sm font-bold text-left flex items-center gap-2">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">①</span>
    Đo mặt trước
   </button>
   <button id="btn-side" onclick="setMode('SIDE')"
    class="btn w-full py-2.5 px-4 rounded-xl bg-sky-600/80 hover:bg-sky-500 text-sm font-bold text-left flex items-center gap-2">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">②</span>
    Đo mặt nghiêng
   </button>
   <button id="btn-adam" onclick="openModal()"
    class="btn w-full py-2.5 px-4 rounded-xl bg-purple-600/80 hover:bg-purple-500 text-sm font-bold text-left flex items-center gap-2 border border-purple-500/30">
    <span class="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-xs">③</span>
    <span>Quét lưng — Adam Test<br><span class="text-xs font-normal opacity-60">Xem hướng dẫn trước khi bắt đầu</span></span>
   </button>
   <button id="btn-report" onclick="setMode('REPORT')"
    class="btn w-full py-2.5 px-4 rounded-xl bg-amber-500/80 hover:bg-amber-400 text-sm font-bold text-left flex items-center gap-2 text-black mt-1">
    <span class="w-5 h-5 rounded-full bg-black/20 flex items-center justify-center text-xs">④</span>
    Xuất báo cáo
   </button>
   <button onclick="fetch('/reset').then(()=>{ updateStatus(); })"
    class="btn w-full py-2 px-4 rounded-xl border border-red-500/40 text-red-400 hover:bg-red-500/10 text-xs font-semibold mt-1">
    ↺ Reset dữ liệu
   </button>
  </div>

  <div class="glass rounded-2xl p-4 flex-1">
   <div class="text-xs text-gray-500 font-semibold uppercase tracking-widest mb-3">Tiến trình</div>
   <div class="flex flex-col gap-2.5">
    <div class="flex items-center gap-2">
     <div id="dot-front" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div class="flex-1">
      <div class="text-xs font-semibold text-gray-300">Mặt trước</div>
      <div id="txt-front" class="text-xs text-gray-500">Chưa đo</div>
     </div>
    </div>
    <div class="flex items-center gap-2">
     <div id="dot-side" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div class="flex-1">
      <div class="text-xs font-semibold text-gray-300">Mặt nghiêng</div>
      <div id="txt-side" class="text-xs text-gray-500">Chưa đo</div>
     </div>
    </div>
    <div class="flex items-center gap-2">
     <div id="dot-adam" class="w-2.5 h-2.5 rounded-full bg-gray-600 flex-shrink-0"></div>
     <div class="flex-1">
      <div class="text-xs font-semibold text-gray-300">Adam Test</div>
      <div id="txt-adam" class="text-xs text-gray-500">Chưa đo</div>
     </div>
    </div>
   </div>

   <div id="adam-guide" class="hidden mt-4 p-3 rounded-xl bg-purple-900/30 border border-purple-500/20 text-xs text-purple-200/80">
    <div class="font-semibold text-purple-300 mb-1">Hướng dẫn Adam Test</div>
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
let audioEnabled = false;
const player = document.getElementById('tts-player');
const ttsBadge = document.getElementById('tts-badge');

function unlockSystem() {
  audioEnabled = true;
  document.getElementById('start-overlay').style.opacity = '0';
  setTimeout(() => document.getElementById('start-overlay').remove(), 500);
  
  player.src = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=";
  player.play().catch(()=>{});
}

setInterval(()=>{
 fetch('/poll_audio').then(r=>r.json()).then(d=>{
  if (d.file && audioEnabled) {
   player.src = '/audio/' + d.file + '?t=' + Date.now();
   player.play().catch(()=>{});
   ttsBadge.textContent = '🔊 TTS';
   ttsBadge.className = 'px-2 py-1 rounded-md font-semibold bg-green-500/20 text-green-300 border border-green-500/30';
  }
 });
}, 500);

function updateStatus() {
 fetch('/status').then(r=>r.json()).then(d=>{
  document.getElementById('cur-mode').textContent = d.mode;

  const steps = ['front','side','adam'];
  const keys  = [d.front, d.side, d.adam];
  steps.forEach((s,i)=>{
   const done = keys[i]==='Da hoan thanh';
   const active = d.mode.toLowerCase()===s || (s==='adam' && d.mode==='ADAM');
   const dot  = document.getElementById('dot-'+s);
   const stEl = document.getElementById('s-'+s);
   dot.className  = 'w-2.5 h-2.5 rounded-full flex-shrink-0 ' + (done?'bg-green-400':active?'bg-sky-400 blink':'bg-gray-600');
   stEl.className = 'px-3 py-1.5 rounded-full border text-xs font-semibold transition-all duration-300 ' +
                    (done?'step-done':active?'step-active':'step-pending');
  });
  document.getElementById('txt-front').textContent = d.front==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';
  document.getElementById('txt-side').textContent  = d.side ==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';
  document.getElementById('txt-adam').textContent  = d.adam ==='Da hoan thanh'?'✓ Hoàn thành':'Chờ đo';

  const guide = document.getElementById('adam-guide');
  const stepTxt = document.getElementById('adam-step-txt');
  if (d.mode==='ADAM') {
   guide.classList.remove('hidden');
   const guides = {
    WAIT_BACK: '📷 Quay LƯNG vào camera, đứng thẳng. AI sẽ tự nhận diện.',
    WAIT_BEND: '⬇ Từ từ cúi người về phía trước, tay thả thẳng xuống đất.',
    SCANNING:  '🔬 Đang phân tích... Giữ nguyên tư thế.',
    DONE:      '✅ Hoàn thành! Chuyển sang báo cáo.',
   };
   stepTxt.textContent = guides[d.adam_state] || '';
  } else {
   guide.classList.add('hidden');
  }

  const cb = document.getElementById('cam-badge');
  cb.textContent = d.camera ? 'CAM ✓' : 'CAM ✗';
  cb.className = d.camera
   ? 'px-2 py-1 rounded-md font-semibold bg-green-500/20 text-green-300 border border-green-500/30'
   : 'px-2 py-1 rounded-md font-semibold bg-red-500/20 text-red-300 border border-red-500/30 blink';
 });
}
setInterval(updateStatus, 1000);
updateStatus();

function setMode(m) {
 audioEnabled = true;
 fetch('/set_mode/'+m).then(()=>updateStatus());
}
function openModal()  { audioEnabled=true; document.getElementById('adam-modal').classList.remove('hidden'); }
function closeModal() { document.getElementById('adam-modal').classList.add('hidden'); }
function startAdam()  { closeModal(); setMode('ADAM'); }
document.getElementById('adam-modal').addEventListener('click', e=>{ if(e.target===e.currentTarget) closeModal(); });
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run("doe:app", host="0.0.0.0", port=8000, workers=1, reload=False)