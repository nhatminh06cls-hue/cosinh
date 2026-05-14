from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import cv2, math, numpy as np, torch, uvicorn, time, os, threading, queue

# ══════════════════════════════════════════════════════
# TTS — QUEUE-BASED (không bao giờ ngắt câu đang nói)
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
    "FRONT_MOVING":   "Xin đứng yên, không di chuyển.",
    "FRONT_WRONG":    "Xin quay mặt thẳng vào camera.",
    "FRONT_GOOD":     "Tư thế tốt. Giữ nguyên.",
    "FRONT_DONE":     "Hoàn thành mặt trước. Xin nghiêng người chín mươi độ.",
    "SIDE_START":     "Xin nghiêng người, nhìn sang một bên.",
    "SIDE_MOVING":    "Xin đứng yên.",
    "SIDE_WRONG":     "Xin nghiêng người thêm.",
    "SIDE_GOOD":      "Tư thế tốt. Giữ nguyên.",
    "SIDE_DONE":      "Hoàn thành mặt nghiêng. Chuẩn bị kiểm tra cột sống.",
    "ADAM_INTRO":     "Xin quay lưng vào camera. Hai chân rộng bằng vai, hai tay thả lỏng.",
    "ADAM_STAND_OK":  "Tốt. Bây giờ xin từ từ cúi người về phía trước. Hai tay thả thẳng xuống.",
    "ADAM_BEND_OK":   "Giữ nguyên tư thế. Hệ thống đang phân tích cột sống.",
    "ADAM_DONE_OK":   "Kiểm tra hoàn tất. Đang chuyển sang báo cáo.",
    "ADAM_DONE_WARN": "Phát hiện bất thường. Đang chuyển sang báo cáo.",
    "ADAM_FAR":       "Xin bước lại gần camera hơn.",
    "ADAM_CLOSE":     "Xin bước ra xa camera hơn.",
    "REPORT_READY":   "Báo cáo đã sẵn sàng.",
}

# Queue chứa các key cần phát — thread riêng phát lần lượt
_tts_queue: queue.Queue = queue.Queue()
_is_speaking = False          # True khi đang phát, block enqueue trùng
_last_key = ""
_last_key_time = 0.0
_pending_file = ""            # file mp3 mới nhất cho browser fetch
_pending_lock = threading.Lock()


def _tts_worker():
    """Thread chạy ngầm, phát audio tuần tự — không bao giờ ngắt câu."""
    global _is_speaking, _pending_file
    while True:
        key = _tts_queue.get()          # chờ item
        _is_speaking = True
        try:
            fname  = f"{key}.mp3"
            fpath  = os.path.join(AUDIO_DIR, fname)
            if not os.path.exists(fpath) and TTS_OK:
                gTTS(text=SCRIPTS[key], lang="vi").save(fpath)
            if os.path.exists(fpath):
                with _pending_lock:
                    _pending_file = fname
                # Ước tính thời gian phát (độ dài text ÷ 3 ký tự/giây, min 2s)
                est = max(2.0, len(SCRIPTS.get(key,"")) / 10)
                time.sleep(est)          # giữ _is_speaking=True suốt thời gian phát
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            _is_speaking = False
            _tts_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


def speak(key: str, force: bool = False):
    """Đưa key vào queue. Nếu force=False, không lặp lại trong 4 giây."""
    global _last_key, _last_key_time
    if key not in SCRIPTS:
        return
    now = time.time()
    if not force and key == _last_key and (now - _last_key_time) < 4.0:
        return
    if not force and _is_speaking:
        return                           # đang nói — bỏ qua thông báo ít quan trọng
    _last_key = key
    _last_key_time = now
    try:
        _tts_queue.put_nowait(key)
    except queue.Full:
        pass


def speak_force(key: str):
    """Luôn phát — dùng cho các thông báo chuyển bước quan trọng."""
    global _last_key, _last_key_time
    _last_key = key
    _last_key_time = time.time()
    _tts_queue.put(key)


# ══════════════════════════════════════════════════════
# AI MODELS (init trong lifespan)
# ══════════════════════════════════════════════════════
model = midas = device = midas_transforms = None


# ══════════════════════════════════════════════════════
# CAMERA INIT — MJPEG FIRST
# ══════════════════════════════════════════════════════
def init_camera():
    def warmup(cap, n=10):
        for _ in range(n):
            ret, f = cap.read()
            if ret and f is not None: return True
            time.sleep(0.1)
        return False

    print("[CAM 1/3] V4L2 + MJPEG 1280×720 @30fps …")
    try:
        c = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            if warmup(c):
                print(f"  Camera OK! {int(c.get(3))}×{int(c.get(4))}")
                return c
        c.release()
    except Exception as e: print(f"  {e}")

    print("[CAM 2/3] GStreamer MJPEG …")
    for gst in [
        "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false",
        "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1  ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true sync=false",
    ]:
        try:
            c = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if c.isOpened() and warmup(c):
                print("  Camera OK! GStreamer")
                return c
            c.release()
        except Exception as e: print(f"  {e}")

    print("[CAM 3/3] CAP_ANY …")
    try:
        c = cv2.VideoCapture(0, cv2.CAP_ANY)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            if warmup(c):
                print("  Camera OK! CAP_ANY")
                return c
        c.release()
    except Exception as e: print(f"  {e}")

    print("CAMERA FAIL — sudo fuser -k /dev/video0 roi thu lai")
    return None


# ══════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap, model, midas, device, midas_transforms
    print("=== KHOI TAO HE THONG AI ===")
    model = YOLO("yolov8m-pose.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device); midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    cap = init_camera()
    if TTS_OK:
        print("Pre-generating TTS …")
        for k in SCRIPTS:
            fpath = os.path.join(AUDIO_DIR, f"{k}.mp3")
            if not os.path.exists(fpath):
                try: gTTS(text=SCRIPTS[k], lang="vi").save(fpath)
                except: pass
        print("TTS ready!")
    yield
    if cap and cap.isOpened(): cap.release(); print("Camera released")


app = FastAPI(title="Gatekeeper Omni", lifespan=lifespan)
cap = None

# ══════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════
current_mode   = "FRONT"
stable_cnt     = 0
path_cnt       = 0
prev_pelvis    = None
adam_state     = "BACK_STAND"   # BACK_STAND → BACK_BEND → SCANNING → DONE
baseline_h     = 0
auto_switch_at = 0.0            # timestamp de tu dong chuyen mode
_mode_locked   = False          # True sau khi da khoa du lieu, cho cho dem auto-switch

report_data = {
    "FRONT": {"sh_angle": 0, "shift_ratio": 0, "status": "Chua do"},
    "SIDE":  {"torso_tilt": 0, "neck_ratio": 0, "status": "Chua do"},
    "ADAM":  {"asym_index": 0, "status": "Chua do"},
}

# Nguong
MOVE_LIM   = 7.0
LOCK_TIME  = 90      # frames de xac nhan benh ly
TWIST_LIM  = 4.0;  FRONT_TILT  = 5.0;  MIN_SH_R = 0.18;  LAT_SHIFT = 0.10
SIDE_TILT  = 15.0; MIN_NECK_R  = 0.25; MAX_SH_SIDE = 0.12
MIN_FILL   = 0.45; MAX_FILL = 0.92; BEND_TGT = 0.58
STAND_FRAMES = 60   # 2s @30fps
BEND_FRAMES  = 45   # 1.5s

AUTO_SWITCH_DELAY = 3.0   # giay cho truoc khi chuyen mode


def calc_dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])


def _do_auto_switch(next_mode: str):
    """Dat hen gio chuyen mode sau AUTO_SWITCH_DELAY giay."""
    global auto_switch_at, _mode_locked
    _mode_locked   = True
    auto_switch_at = time.time() + AUTO_SWITCH_DELAY


def _apply_mode(mode: str):
    global current_mode, stable_cnt, path_cnt, prev_pelvis, adam_state, baseline_h, _mode_locked
    current_mode = mode
    stable_cnt = path_cnt = 0
    prev_pelvis = None
    _mode_locked = False
    if mode == "ADAM":
        adam_state = "BACK_STAND"
        baseline_h = 0
        speak_force("ADAM_INTRO")
    elif mode == "SIDE":
        speak_force("FRONT_DONE")
    elif mode == "REPORT":
        speak_force("REPORT_READY")


# ══════════════════════════════════════════════════════
# GENERATE FRAMES
# ══════════════════════════════════════════════════════
def generate_frames():
    global current_mode, stable_cnt, path_cnt, prev_pelvis
    global adam_state, baseline_h, report_data, auto_switch_at, _mode_locked

    # Sequence: FRONT → SIDE → ADAM → REPORT
    next_mode_map = {"FRONT": "SIDE", "SIDE": "ADAM", "ADAM": "REPORT"}

    if not cap or not cap.isOpened():
        ef = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(ef, "CAMERA OFFLINE", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        _, buf = cv2.imencode('.jpg', ef)
        while True:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(1)
        return

    while True:
        # ── Auto switch ──
        if _mode_locked and auto_switch_at > 0 and time.time() >= auto_switch_at:
            nxt = next_mode_map.get(current_mode)
            if nxt:
                _apply_mode(nxt)
            auto_switch_at = 0.0

        ok, frame = cap.read()
        if not ok: time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        res = model(frame, verbose=False)
        msg, mc = "Khong tim thay benh nhan", (80, 80, 80)
        has_box = has_kpts = False
        x1=y1=x2=y2=bh=bw = 0

        if res[0].boxes is not None and len(res[0].boxes.xyxy):
            b = res[0].boxes.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int,b[:4]); bh=y2-y1; bw=x2-x1; has_box=True

        if res[0].keypoints is not None and len(res[0].keypoints.xy):
            kp = res[0].keypoints.xy[0]
            if len(kp) >= 13:
                le,re = kp[3],kp[4]; ls,rs = kp[5],kp[6]; lh,rh = kp[11],kp[12]
                me = ((le[0]+re[0])/2,(le[1]+re[1])/2)
                ms = ((ls[0]+rs[0])/2,(ls[1]+rs[1])/2)
                mh = ((lh[0]+rh[0])/2,(lh[1]+rh[1])/2)
                has_kpts = True

        # ── Overlay: thanh tien trinh + mode label ──
        _draw_hud(frame, W, H, current_mode, adam_state if current_mode=="ADAM" else "")

        # ══════════════════════════════════════════════
        # FRONT
        # ══════════════════════════════════════════════
        if current_mode == "FRONT":
            _draw_guide_box(frame, W, H)
            if _mode_locked:
                msg, mc = "Da khoa du lieu FRONT — Chuyen sang SIDE...", (0,255,120)
            elif not has_kpts or not has_box:
                speak("FRONT_START"); msg, mc = "Xin dung thang, nhin vao camera", (180,180,0)
            else:
                dy=abs(ls[1]-rs[1]); dx=abs(ls[0]-rs[0])
                sh_angle = math.degrees(math.atan2(dy,dx))
                moved = calc_dist(mh, prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh
                tdx=abs(ms[0]-mh[0]); tdy=abs(ms[1]-mh[1])
                tilt = math.degrees(math.atan2(tdx,tdy)) if tdy>0 else 90
                shw = abs(ls[0]-rs[0])
                shr = shw/bh if bh else 0
                sft = abs(ms[0]-mh[0])/shw if shw else 0

                if moved > MOVE_LIM:
                    msg,mc = "Xin dung yen!",           (0,60,255); stable_cnt=path_cnt=0; speak("FRONT_MOVING")
                elif shr < MIN_SH_R or sft > LAT_SHIFT:
                    msg,mc = "Quay mat vao camera!",    (0,60,255); stable_cnt=path_cnt=0; speak("FRONT_WRONG")
                elif tilt > FRONT_TILT or sh_angle > TWIST_LIM:
                    path_cnt+=1; stable_cnt=0
                    pct=int(path_cnt/LOCK_TIME*100)
                    msg,mc = f"Phat hien lech vai — xac minh {pct}%", (0,140,255)
                else:
                    stable_cnt+=1; path_cnt=0
                    pct=int(stable_cnt/30*100)
                    msg,mc = f"Tu the chuan — giu nguyen {pct}%", (0,220,80)
                    if stable_cnt==5: speak("FRONT_GOOD")

                cv2.line(frame,(int(ls[0]),int(ls[1])),(int(rs[0]),int(rs[1])),(0,60,255),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,200),2)
                _draw_progress(frame, W, H, stable_cnt, path_cnt)

                if stable_cnt>=30 or path_cnt>=LOCK_TIME:
                    report_data["FRONT"] = {"sh_angle":sh_angle,"shift_ratio":sft*100,"status":"Da hoan thanh"}
                    _do_auto_switch("SIDE")
                    msg,mc = "DA KHOA FRONT — Chuyen SIDE trong 3s...", (0,255,120)

        # ══════════════════════════════════════════════
        # SIDE
        # ══════════════════════════════════════════════
        elif current_mode == "SIDE":
            _draw_guide_box(frame, W, H)
            if _mode_locked:
                msg,mc = "Da khoa du lieu SIDE — Chuyen sang ADAM TEST...", (0,255,120)
            elif not has_kpts or not has_box:
                speak("SIDE_START"); msg,mc = "Xin nghieng nguoi 90 do, nhin sang mot ben", (180,180,0)
            else:
                nl = calc_dist(me,ms); tl = calc_dist(ms,mh)
                nr = nl/tl if tl else 0
                tdx=abs(ms[0]-mh[0]); tdy=abs(ms[1]-mh[1])
                tilt = math.degrees(math.atan2(tdx,tdy)) if tdy>0 else 90
                moved = calc_dist(mh,prev_pelvis) if prev_pelvis else 0
                prev_pelvis = mh
                shw=abs(ls[0]-rs[0]); shr=shw/bh if bh else 0

                if moved > MOVE_LIM:
                    msg,mc="Xin dung yen!",(0,60,255); stable_cnt=path_cnt=0; speak("SIDE_MOVING")
                elif shr>MAX_SH_SIDE or nr<MIN_NECK_R:
                    msg,mc="Nghieng nguoi them!",(0,60,255); stable_cnt=path_cnt=0; speak("SIDE_WRONG")
                elif tilt>SIDE_TILT:
                    path_cnt+=1; stable_cnt=0
                    pct=int(path_cnt/LOCK_TIME*100)
                    msg,mc=f"Phat hien gu/nga lung — xac minh {pct}%",(0,140,255)
                else:
                    stable_cnt+=1; path_cnt=0
                    pct=int(stable_cnt/30*100)
                    msg,mc=f"Tu the chuan — giu nguyen {pct}%",(0,220,80)
                    if stable_cnt==5: speak("SIDE_GOOD")

                cv2.line(frame,(int(me[0]),int(me[1])),(int(ms[0]),int(ms[1])),(200,0,200),3)
                cv2.line(frame,(int(ms[0]),int(ms[1])),(int(mh[0]),int(mh[1])),(0,220,200),3)
                _draw_progress(frame, W, H, stable_cnt, path_cnt)

                if stable_cnt>=30 or path_cnt>=LOCK_TIME:
                    report_data["SIDE"] = {"torso_tilt":tilt,"neck_ratio":nr*100,"status":"Da hoan thanh"}
                    _do_auto_switch("ADAM")
                    speak_force("SIDE_DONE")
                    msg,mc="DA KHOA SIDE — Chuyen ADAM TEST trong 3s...",(0,255,120)

        # ══════════════════════════════════════════════
        # ADAM — QUAY LUNG VAO CAM
        # ══════════════════════════════════════════════
        elif current_mode == "ADAM":
            if not has_box:
                speak("ADAM_INTRO")
                msg,mc = "Xin quay LUNG vao camera, buoc vao khung hinh", (180,180,0)
            else:
                fill = bh/H
                if fill < MIN_FILL:
                    speak("ADAM_FAR"); msg,mc="Lai GAN camera hon",(0,140,255); stable_cnt=0
                    _draw_distance_hint(frame, W, H, "GAN")
                elif fill > MAX_FILL:
                    speak("ADAM_CLOSE"); msg,mc="Lui RA XA hon",(0,140,255); stable_cnt=0
                    _draw_distance_hint(frame, W, H, "XA")
                else:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,100),2)

                    # ── BACK_STAND: cho dung thang quay lung ──
                    if adam_state == "BACK_STAND":
                        speak("ADAM_INTRO")
                        msg,mc="Quay LUNG vao camera, dung thang...",(100,200,255)
                        # Khi quay lung: keypoints vai / hong van detect duoc
                        # Detect dung thang qua torso thang dung
                        if has_kpts:
                            tdx=abs(ms[0]-mh[0]); tdy=abs(ms[1]-mh[1])
                            tilt=math.degrees(math.atan2(tdx,tdy)) if tdy>0 else 90
                            if tilt<20:
                                stable_cnt+=1
                                pct=int(stable_cnt/STAND_FRAMES*100)
                                msg,mc=f"Xac nhan tu the dung thang... {pct}%",(100,200,255)
                                _draw_progress(frame,W,H,stable_cnt,0,STAND_FRAMES)
                                if stable_cnt>=STAND_FRAMES:
                                    baseline_h=bh; adam_state="BACK_BEND"
                                    stable_cnt=0; speak_force("ADAM_STAND_OK")
                            else:
                                stable_cnt=max(0,stable_cnt-2)

                    # ── BACK_BEND: cho cui nguoi (van quay lung) ──
                    elif adam_state == "BACK_BEND":
                        msg,mc="Tu tu CUI NGUOI ve phia truoc, tay tha xuong...",(255,220,0)
                        bend_r = bh/baseline_h if baseline_h else 1.0
                        if bend_r < BEND_TGT:
                            stable_cnt+=1
                            pct=int(stable_cnt/BEND_FRAMES*100)
                            msg,mc=f"Dang xac nhan goc cui... {pct}%",(0,220,80)
                            _draw_progress(frame,W,H,stable_cnt,0,BEND_FRAMES)
                            if stable_cnt>=BEND_FRAMES:
                                adam_state="SCANNING"; stable_cnt=0
                                speak_force("ADAM_BEND_OK")
                        else:
                            stable_cnt=max(0,stable_cnt-1)
                            # Ve mui ten chi dan
                            ax=W-70
                            cv2.arrowedLine(frame,(ax,H//4),(ax,3*H//4),(0,220,255),5,tipLength=0.25)
                            cv2.putText(frame,"CUI",(ax-25,3*H//4+35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),2)

                    # ── SCANNING: MiDaS scan lung ──
                    elif adam_state == "SCANNING":
                        # Crop phan lung (60% tren cua bbox)
                        roi_y1=max(0,y1); roi_y2=min(H,y1+int(bh*0.65))
                        cv2.rectangle(frame,(x1,roi_y1),(x2,roi_y2),(180,0,255),4)
                        # Scan line animation
                        scan_y=roi_y1+int(((time.time()*60)%max(1,roi_y2-roi_y1)))
                        cv2.line(frame,(x1,scan_y),(x2,scan_y),(180,0,255),2)
                        msg,mc="MIDAS DANG PHAN TICH LUNG...",(180,0,255)

                        if report_data["ADAM"]["status"]=="Chua do":
                            crop=frame[roi_y1:roi_y2,x1:x2]
                            if crop.size>0:
                                rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                                inp=midas_transforms(rgb).to(device)
                                with torch.no_grad():
                                    pred=midas(inp)
                                    pred=torch.nn.functional.interpolate(
                                        pred.unsqueeze(1),size=rgb.shape[:2],
                                        mode="bicubic",align_corners=False).squeeze()
                                dm=pred.cpu().numpy()
                                mx=dm.shape[1]//2
                                lm=np.mean(dm[:,:mx]); rm=np.mean(dm[:,mx:])
                                asym=(abs(lm-rm)/(max(lm,rm)+1e-6))*100
                                report_data["ADAM"]={"asym_index":asym,"status":"Da hoan thanh"}
                                adam_state="DONE"
                                spk = "ADAM_DONE_WARN" if asym>10 else "ADAM_DONE_OK"
                                speak_force(spk)
                                _do_auto_switch("REPORT")
                        else:
                            adam_state="DONE"

                    # ── DONE ──
                    elif adam_state=="DONE":
                        asym=report_data["ADAM"].get("asym_index",0)
                        if _mode_locked:
                            msg,mc="Hoan thanh ADAM — Chuyen REPORT trong 3s...",(0,255,120)
                        else:
                            c=(0,60,255) if asym>10 else (0,220,80)
                            msg,mc=(f"CANH BAO VEO! Do lech: {asym:.1f}%" if asym>10
                                    else f"Binh thuong. Do lech: {asym:.1f}%"), c

        # ══════════════════════════════════════════════
        # REPORT
        # ══════════════════════════════════════════════
        elif current_mode == "REPORT":
            _draw_report(frame, W, H, report_data)
            msg,mc="BAO CAO TAM SOAT — Nhan RESET de do lai",(200,200,0)

        # ── Status bar ──
        _draw_status_bar(frame, W, H, msg, mc)

        _, buf = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


# ══════════════════════════════════════════════════════
# DRAW HELPERS
# ══════════════════════════════════════════════════════
def _draw_hud(frame, W, H, mode, sub=""):
    """Mode label góc trên phải."""
    labels = {"FRONT":"MAT TRUOC","SIDE":"MAT NGHIENG","ADAM":"QUET LUNG","REPORT":"BAO CAO"}
    colors = {"FRONT":(30,120,255),"SIDE":(200,60,200),"ADAM":(0,180,120),"REPORT":(30,200,220)}
    label = labels.get(mode, mode)
    col   = colors.get(mode, (200,200,200))
    cv2.rectangle(frame,(W-220,8),(W-8,42),(20,20,30),-1)
    cv2.rectangle(frame,(W-220,8),(W-8,42),col,1)
    cv2.putText(frame,label,(W-210,32),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    if sub:
        sub_labels = {"BACK_STAND":"B1: DUNG THANG","BACK_BEND":"B2: CUI NGUOI","SCANNING":"B3: SCAN","DONE":"XONG"}
        sl = sub_labels.get(sub,"")
        if sl:
            cv2.putText(frame,sl,(W-210,55),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)


def _draw_guide_box(frame, W, H):
    """Khung hướng dẫn đứng vào."""
    pad_x, pad_y = int(W*0.12), int(H*0.06)
    cv2.rectangle(frame,(pad_x,pad_y),(W-pad_x,H-pad_y),(60,60,80),1)
    # 4 goc noi bat
    L=30; t=2; c=(100,200,255)
    for x,y in [(pad_x,pad_y),(W-pad_x,pad_y),(pad_x,H-pad_y),(W-pad_x,H-pad_y)]:
        sx=1 if x==pad_x else -1; sy=1 if y==pad_y else -1
        cv2.line(frame,(x,y),(x+sx*L,y),c,t)
        cv2.line(frame,(x,y),(x,y+sy*L),c,t)


def _draw_progress(frame, W, H, ok_cnt, bad_cnt, total=30):
    """Thanh tiến trình ở dưới cùng."""
    bw=int(W*0.6); bx=(W-bw)//2; by=H-28; bh_bar=12
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh_bar),(40,40,50),-1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh_bar),(60,60,80),1)
    if ok_cnt>0:
        fill=min(1.0,ok_cnt/total)
        cv2.rectangle(frame,(bx,by),(bx+int(bw*fill),by+bh_bar),(0,200,80),-1)
    if bad_cnt>0:
        fill=min(1.0,bad_cnt/LOCK_TIME)
        cv2.rectangle(frame,(bx,by),(bx+int(bw*fill),by+bh_bar),(0,80,255),-1)


def _draw_distance_hint(frame, W, H, direction):
    """Mũi tên khoảng cách."""
    cy=H//2
    if direction=="GAN":
        cv2.arrowedLine(frame,(W//2,cy+60),(W//2,cy-60),(0,180,255),5,tipLength=0.3)
        cv2.putText(frame,"LAI GAN",(W//2-60,cy+100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,180,255),2)
    else:
        cv2.arrowedLine(frame,(W//2,cy-60),(W//2,cy+60),(0,180,255),5,tipLength=0.3)
        cv2.putText(frame,"LUI RA XA",(W//2-70,cy+100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,180,255),2)


def _draw_status_bar(frame, W, H, msg, color):
    """Thanh trạng thái đáy màn hình."""
    bar_h = 50
    overlay = frame[H-bar_h:H, 0:W].copy()
    cv2.rectangle(overlay,(0,0),(W,bar_h),(15,15,20),-1)
    frame[H-bar_h:H,0:W] = cv2.addWeighted(overlay,0.85,frame[H-bar_h:H,0:W],0.15,0)
    cv2.putText(frame, msg, (20, H-16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def _draw_report(frame, W, H, rd):
    """Overlay báo cáo."""
    overlay = frame.copy()
    cv2.rectangle(overlay,(30,30),(W-30,H-30),(10,12,18),-1)
    frame[:] = cv2.addWeighted(overlay,0.92,frame,0.08,0)

    title = "PHIEU TAM SOAT COT SONG"
    cv2.putText(frame,title,(W//2-len(title)*9,75),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,200,220),2)
    cv2.line(frame,(60,90),(W-60,90),(50,50,70),1)

    def row(label, val, kluan, y):
        cv2.putText(frame,label,(70,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(180,180,200),1)
        cv2.putText(frame,val,(70,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(220,220,220),1)
        c=(0,60,255) if "BAO" in kluan or "VEO" in kluan or "GU" in kluan else (0,200,80)
        cv2.putText(frame,f"→ {kluan}",(70,y+60),cv2.FONT_HERSHEY_SIMPLEX,0.75,c,2)
        cv2.line(frame,(60,y+80),(W-60,y+80),(40,40,55),1)

    f=rd["FRONT"]; s=rd["SIDE"]; a=rd["ADAM"]

    if f["status"]=="Da hoan thanh":
        kl="CANH BAO VEO/LECH" if f["sh_angle"]>TWIST_LIM else "Binh thuong"
        row("[1] MAT TRUOC",f"Goc lech vai: {f['sh_angle']:.1f} do",kl,130)
    else:
        cv2.putText(frame,"[1] MAT TRUOC — Chua do",(70,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,120),1)

    if s["status"]=="Da hoan thanh":
        kl="CANH BAO GU/NGA LUNG" if s["torso_tilt"]>SIDE_TILT else "Binh thuong"
        row("[2] MAT NGHIENG",f"Goc nga lung: {s['torso_tilt']:.1f} do",kl,310)
    else:
        cv2.putText(frame,"[2] MAT NGHIENG — Chua do",(70,340),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,120),1)

    if a["status"]=="Da hoan thanh":
        kl="NGUY CO VEO CAO — can kham chuyen khoa" if a["asym_index"]>10 else "Binh thuong"
        row("[3] QUET LUNG MIDAS",f"Chi so bat doi: {a['asym_index']:.1f}%",kl,490)
    else:
        cv2.putText(frame,"[3] QUET LUNG — Chua do",(70,520),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,120),1)

    cv2.putText(frame,"Gatekeeper Omni — Clinical AI v2",(W//2-180,H-60),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,60,80),1)


# ══════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/poll_audio")
def poll_audio():
    global _pending_file
    with _pending_lock:
        f = _pending_file; _pending_file = ""
    return {"file": f}


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    p = os.path.join(AUDIO_DIR, filename)
    return FileResponse(p, media_type="audio/mpeg") if os.path.exists(p) else {"error":"not found"}


@app.get("/set_mode/{mode}")
def set_mode(mode: str):
    _apply_mode(mode)
    return {"status": "ok", "mode": mode}


@app.get("/reset")
def reset():
    global report_data, auto_switch_at, _mode_locked
    report_data = {
        "FRONT": {"sh_angle":0,"shift_ratio":0,"status":"Chua do"},
        "SIDE":  {"torso_tilt":0,"neck_ratio":0,"status":"Chua do"},
        "ADAM":  {"asym_index":0,"status":"Chua do"},
    }
    auto_switch_at=0.0; _mode_locked=False
    _apply_mode("FRONT")
    return {"status": "reset"}


@app.get("/camera_status")
def cam_status():
    return {"ok": bool(cap and cap.isOpened())}


# ══════════════════════════════════════════════════════
# DASHBOARD HTML
# ══════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gatekeeper Omni</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  :root{--accent:#00c8ff;--warn:#ff4d4d;--ok:#00e087}
  body{background:#080b10;font-family:'Segoe UI',system-ui,sans-serif}
  .glass{background:rgba(255,255,255,.04);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.08)}
  .btn-mode{transition:all .2s;position:relative;overflow:hidden}
  .btn-mode:active{transform:scale(.97)}
  .btn-mode.active-mode::after{content:'';position:absolute;inset:0;border:2px solid var(--accent);border-radius:inherit;pointer-events:none}
  .badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:99px;font-size:.7rem;font-weight:700;letter-spacing:.05em}
  .dot{width:7px;height:7px;border-radius:50%}
  .pulse{animation:pulse 1.4s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  #step-bar .step{flex:1;text-align:center;font-size:.65rem;padding:6px 4px;border-radius:8px;transition:all .3s}
  #step-bar .step.active{background:rgba(0,200,255,.15);color:#00c8ff;border:1px solid rgba(0,200,255,.3)}
  #step-bar .step.done{background:rgba(0,224,135,.1);color:#00e087;border:1px solid rgba(0,224,135,.25)}
  #step-bar .step.pending{color:#444;border:1px solid #222}
  /* Modal */
  #modal-overlay{transition:opacity .25s}
  .modal-step{border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;text-align:center;flex:1}
</style>
</head>
<body class="text-gray-100 h-screen flex flex-col select-none">

<!-- ░░ HEADER ░░ -->
<header class="glass px-5 py-3 flex items-center justify-between shrink-0 border-b border-white/5">
  <div class="flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-black text-sm">G</div>
    <div>
      <div class="text-sm font-bold tracking-widest text-cyan-400">GATEKEEPER OMNI</div>
      <div class="text-xs text-gray-500">Clinical Spine Screening AI</div>
    </div>
  </div>
  <div class="flex items-center gap-2">
    <span id="cam-badge" class="badge bg-yellow-500/10 border border-yellow-500/30 text-yellow-400">
      <span class="dot bg-yellow-400 pulse"></span>CHECKING
    </span>
    <span id="tts-badge" class="badge bg-gray-500/10 border border-gray-500/20 text-gray-500">
      <span class="dot bg-gray-500"></span>TTS
    </span>
    <span class="badge bg-green-500/10 border border-green-500/30 text-green-400">
      <span class="dot bg-green-400 pulse"></span>RTX 3070Ti
    </span>
  </div>
</header>

<audio id="tts-player" preload="auto"></audio>

<!-- ░░ BODY ░░ -->
<div class="flex flex-1 overflow-hidden gap-3 p-3">

  <!-- VIDEO PANEL -->
  <div class="flex-1 glass rounded-2xl overflow-hidden relative flex items-center justify-center bg-black/60">
    <img id="vf" src="/video_feed" class="w-full h-full object-contain">
    <div id="vf-err" class="hidden absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-400">
      <svg class="w-12 h-12 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-width="1.5" d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14m0-4v4m0-4H5a2 2 0 00-2 2v0a2 2 0 002 2h10"/></svg>
      <span class="text-sm font-bold">Mất kết nối camera</span>
    </div>
  </div>

  <!-- CONTROL PANEL -->
  <div class="w-72 flex flex-col gap-3">

    <!-- Step indicator -->
    <div id="step-bar" class="glass rounded-xl p-3 flex gap-2">
      <div class="step pending" id="s-FRONT">① Mặt trước</div>
      <div class="step pending" id="s-SIDE">② Nghiêng</div>
      <div class="step pending" id="s-ADAM">③ Cột sống</div>
      <div class="step pending" id="s-REPORT">④ Báo cáo</div>
    </div>

    <!-- Mode buttons -->
    <div class="glass rounded-xl p-4 flex flex-col gap-2">
      <div class="text-xs text-gray-500 font-semibold tracking-widest mb-1">ĐIỀU KHIỂN</div>

      <button id="btn-FRONT" onclick="selectMode('FRONT')"
        class="btn-mode w-full py-3 rounded-xl font-bold text-sm bg-blue-600/20 border border-blue-500/30 hover:bg-blue-600/40 text-blue-300">
        ① Đo mặt trước
      </button>
      <button id="btn-SIDE" onclick="selectMode('SIDE')"
        class="btn-mode w-full py-3 rounded-xl font-bold text-sm bg-purple-600/20 border border-purple-500/30 hover:bg-purple-600/40 text-purple-300">
        ② Đo mặt nghiêng
      </button>
      <button id="btn-ADAM" onclick="openAdamModal()"
        class="btn-mode w-full py-3 rounded-xl font-bold text-sm bg-teal-600/20 border border-teal-500/30 hover:bg-teal-600/40 text-teal-300">
        ③ Quét cột sống
        <div class="text-xs opacity-60 font-normal mt-0.5">Xem hướng dẫn trước khi đo</div>
      </button>
      <button id="btn-REPORT" onclick="selectMode('REPORT')"
        class="btn-mode w-full py-3 rounded-xl font-bold text-sm bg-amber-600/20 border border-amber-500/30 hover:bg-amber-600/40 text-amber-300 mt-1">
        ④ Xem báo cáo
      </button>
      <button onclick="doReset()"
        class="w-full py-2 rounded-xl font-bold text-xs border border-red-500/30 text-red-400 hover:bg-red-500/10 mt-1">
        ↺ Reset dữ liệu
      </button>
    </div>

    <!-- Status cards -->
    <div class="glass rounded-xl p-4 flex flex-col gap-2 flex-1">
      <div class="text-xs text-gray-500 font-semibold tracking-widest mb-1">KẾT QUẢ</div>
      <div id="card-FRONT" class="result-card rounded-lg p-2.5 bg-white/3 border border-white/6">
        <div class="text-xs text-gray-400 font-semibold">MẶT TRƯỚC</div>
        <div id="val-FRONT" class="text-xs text-gray-500 mt-0.5">Chưa đo</div>
      </div>
      <div id="card-SIDE" class="result-card rounded-lg p-2.5 bg-white/3 border border-white/6">
        <div class="text-xs text-gray-400 font-semibold">MẶT NGHIÊNG</div>
        <div id="val-SIDE" class="text-xs text-gray-500 mt-0.5">Chưa đo</div>
      </div>
      <div id="card-ADAM" class="result-card rounded-lg p-2.5 bg-white/3 border border-white/6">
        <div class="text-xs text-gray-400 font-semibold">CỘT SỐNG (ADAM)</div>
        <div id="val-ADAM" class="text-xs text-gray-500 mt-0.5">Chưa đo</div>
      </div>
    </div>

  </div>
</div>

<!-- ░░ ADAM INTRO MODAL ░░ -->
<div id="modal-overlay" class="hidden fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
  <div class="glass rounded-2xl w-full max-w-xl p-6 border border-teal-500/30 shadow-2xl">
    <div class="flex items-center gap-3 mb-5">
      <div class="w-10 h-10 rounded-xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 font-black">A</div>
      <div>
        <div class="font-bold text-teal-300">Adam Forward Bending Test</div>
        <div class="text-xs text-gray-500">Kiểm tra vẹo cột sống bằng AI độ sâu (MiDaS)</div>
      </div>
    </div>

    <!-- 3 buoc SVG -->
    <div class="flex gap-3 mb-5">
      <!-- B1 -->
      <div class="modal-step bg-white/3">
        <div class="text-teal-400 text-xs font-bold mb-2">BƯỚC 1</div>
        <svg viewBox="0 0 70 130" class="w-12 mx-auto mb-2" fill="none">
          <circle cx="35" cy="14" r="9" fill="#94a3b8"/>
          <line x1="35" y1="23" x2="35" y2="78" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
          <line x1="35" y1="40" x2="14" y2="60" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <line x1="35" y1="40" x2="56" y2="60" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <line x1="35" y1="78" x2="22" y2="118" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <line x1="35" y1="78" x2="48" y2="118" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <!-- lung (back) highlight -->
          <rect x="26" y="25" width="18" height="50" rx="4" fill="#00c8ff" opacity=".2"/>
          <text x="35" y="126" font-size="6" fill="#94a3b8" text-anchor="middle">Quay lưng</text>
        </svg>
        <div class="text-xs text-gray-300 font-medium">Quay lưng vào camera</div>
        <div class="text-xs text-gray-500 mt-1">Hai chân rộng bằng vai</div>
      </div>
      <div class="flex items-center text-gray-600 text-2xl">›</div>
      <!-- B2 -->
      <div class="modal-step bg-white/3">
        <div class="text-teal-400 text-xs font-bold mb-2">BƯỚC 2</div>
        <svg viewBox="0 0 90 90" class="w-16 mx-auto mb-2" fill="none">
          <!-- Nguoi cui, nhin tu ben -->
          <circle cx="18" cy="40" r="8" fill="#94a3b8"/>
          <line x1="25" y1="43" x2="68" y2="30" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>
          <line x1="42" y1="36" x2="36" y2="62" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <line x1="55" y1="33" x2="52" y2="60" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <!-- cot song highlight -->
          <path d="M26 42 Q47 30 68 30" stroke="#00c8ff" stroke-width="2" stroke-dasharray="4,3" opacity=".8"/>
          <line x1="68" y1="30" x2="60" y2="80" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
          <line x1="68" y1="30" x2="78" y2="80" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <div class="text-xs text-gray-300 font-medium">Cúi người 90°</div>
        <div class="text-xs text-gray-500 mt-1">Tay thả xuống tự nhiên</div>
      </div>
      <div class="flex items-center text-gray-600 text-2xl">›</div>
      <!-- B3 -->
      <div class="modal-step bg-white/3">
        <div class="text-teal-400 text-xs font-bold mb-2">BƯỚC 3</div>
        <div class="w-16 h-16 mx-auto mb-2 relative">
          <svg viewBox="0 0 70 70" class="w-full">
            <rect x="8" y="8" width="54" height="54" rx="6" fill="#0f172a"/>
            <rect x="12" y="12" width="20" height="46" rx="3" fill="#00e087" opacity=".6"/>
            <rect x="38" y="12" width="20" height="46" rx="3" fill="#ff4d4d" opacity=".7"/>
            <line x1="35" y1="8" x2="35" y2="62" stroke="white" stroke-width="1" stroke-dasharray="3,2" opacity=".3"/>
            <text x="22" y="66" font-size="6" fill="#00e087" text-anchor="middle">TRÁI</text>
            <text x="48" y="66" font-size="6" fill="#ff4d4d" text-anchor="middle">PHẢI</text>
          </svg>
        </div>
        <div class="text-xs text-gray-300 font-medium">AI phân tích</div>
        <div class="text-xs text-gray-500 mt-1">Heatmap độ lệch 2 bên</div>
      </div>
    </div>

    <!-- Luu y -->
    <div class="bg-amber-500/8 border border-amber-500/20 rounded-xl p-3 mb-5 text-xs text-amber-200">
      <div class="font-bold text-amber-400 mb-1.5">⚠ Lưu ý trước khi bắt đầu</div>
      <div class="grid grid-cols-2 gap-1 text-gray-300">
        <div>• Cuốn áo lên, để lộ lưng</div>
        <div>• Cách camera khoảng 1.5 – 2m</div>
        <div>• Không đeo balo, áo rộng</div>
        <div>• Ánh sáng đủ sáng, không ngược</div>
      </div>
    </div>

    <div class="flex gap-3">
      <button onclick="closeModal()"
        class="flex-1 py-3 rounded-xl border border-white/10 text-gray-400 hover:bg-white/5 text-sm font-bold">
        Hủy
      </button>
      <button onclick="startAdam()"
        class="flex-[2] py-3 rounded-xl bg-teal-600/30 border border-teal-500/50 hover:bg-teal-600/50 text-teal-200 text-sm font-bold">
        Bắt đầu kiểm tra →
      </button>
    </div>
  </div>
</div>

<script>
const MODES = ['FRONT','SIDE','ADAM','REPORT'];
let curMode = 'FRONT';
let audioEnabled = false;
const player = document.getElementById('tts-player');

// Unlock audio on first interaction
document.body.addEventListener('click', ()=>{ audioEnabled=true; }, {once:true});

// Poll audio
setInterval(()=>{
  fetch('/poll_audio').then(r=>r.json()).then(d=>{
    if(d.file && audioEnabled){
      player.src = '/audio/'+d.file;
      player.play().catch(()=>{});
      const b=document.getElementById('tts-badge');
      b.innerHTML='<span class="dot bg-green-400 pulse"></span>TTS ON';
      b.className='badge bg-green-500/10 border border-green-500/30 text-green-400';
    }
  });
}, 500);

// Camera status
fetch('/camera_status').then(r=>r.json()).then(d=>{
  const b=document.getElementById('cam-badge');
  if(d.ok){
    b.innerHTML='<span class="dot bg-green-400"></span>CAM OK';
    b.className='badge bg-green-500/10 border border-green-500/30 text-green-400';
  } else {
    b.innerHTML='<span class="dot bg-red-400 pulse"></span>CAM OFFLINE';
    b.className='badge bg-red-500/10 border border-red-500/30 text-red-400';
    document.getElementById('vf-err').classList.remove('hidden');
  }
});

// Poll mode + results từ server mỗi 1s để sync auto-switch
setInterval(syncState, 1000);

function syncState(){
  // Sync step bar theo curMode (optimistic), server tự switch
  // Cũng refresh result cards
}

function selectMode(m){
  audioEnabled = true;
  fetch('/set_mode/'+m).then(r=>r.json()).then(()=>{ updateUI(m); });
}

function updateUI(m){
  curMode = m;
  // Buttons
  MODES.forEach(x=>{
    const b = document.getElementById('btn-'+x);
    if(!b) return;
    b.classList.toggle('active-mode', x===m);
  });
  // Step bar
  const order = ['FRONT','SIDE','ADAM','REPORT'];
  const idx = order.indexOf(m);
  order.forEach((x,i)=>{
    const el = document.getElementById('s-'+x);
    if(!el) return;
    el.className = 'step ' + (i<idx?'done': i===idx?'active':'pending');
  });
}

// Modal
function openAdamModal(){ audioEnabled=true; document.getElementById('modal-overlay').classList.remove('hidden'); }
function closeModal(){ document.getElementById('modal-overlay').classList.add('hidden'); }
function startAdam(){ closeModal(); selectMode('ADAM'); }
document.getElementById('modal-overlay').addEventListener('click', e=>{ if(e.target===e.currentTarget) closeModal(); });

function doReset(){
  fetch('/reset').then(()=>{ updateUI('FRONT'); });
}

// Init
updateUI('FRONT');
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("doe:app", host="0.0.0.0", port=8000, workers=1, reload=False)
