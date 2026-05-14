import cv2
import math
import numpy as np
import torch
from ultralytics import YOLO

print("=== GATEKEEPER OMNI V16 (FULL MIDAS INTEGRATION) ===")
print("Khoi tao YOLOv8 Pose... ")
model = YOLO("yolov8m-pose.pt")

print("Khoi tao MiDaS Depth Estimation tren RTX 3070ti... Vui long doi!")
# Tải MiDaS bản nhẹ để tiết kiệm dung lượng SSD, đưa thẳng lên GPU
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Tối ưu Linux Camera V4L2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==========================================
# BIẾN TRẠNG THÁI & LƯU TRỮ DỮ LIỆU
# ==========================================
current_mode = "FRONT"  
stable_counter = 0
pathology_confirm_counter = 0 
prev_pelvis = None

# Kho lưu trữ kết quả để làm Report
report_data = {
    "FRONT": {"sh_angle": 0, "shift_ratio": 0, "status": "Chua do"},
    "SIDE": {"torso_tilt": 0, "neck_ratio": 0, "status": "Chua do"},
    "ADAM": {"asym_index": 0, "status": "Chua do"}
}

# Biến Adam Test
adam_state = "STEP1_SIDE"
baseline_box_height = 0
prev_box_height = 0
prev_box_center = None

# Ngưỡng lâm sàng
MOVEMENT_LIMIT = 7.0      
FORCE_LOCK_TIME = 90        

TWIST_ANGLE_LIMIT = 4.0     
FRONT_TILT_LIMIT = 5.0      
MIN_SH_RATIO = 0.18         
LATERAL_SHIFT_LIMIT = 0.10  

SIDE_TILT_LIMIT = 15.0    
MIN_NECK_RATIO = 0.25     
MAX_SIDE_SH_RATIO = 0.12    

MIN_FILL, MAX_FILL = 0.50, 0.90           
BEND_TARGET = 0.60        
ADAM_STABILITY, ADAM_DRIFT = 15, 20           

def calc_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    results = model(frame, verbose=False)
    status_msg = "Khong tim thay benh nhan"
    msg_color = (0, 0, 255)
    
    has_box = False
    has_kpts = False

    if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])
        curr_h, curr_w = y2 - y1, x2 - x1
        curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        has_box = True

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kpts = results[0].keypoints.xy[0]
        if len(kpts) >= 13:
            l_ear, r_ear = kpts[3], kpts[4]
            l_sh, r_sh = kpts[5], kpts[6]
            l_hip, r_hip = kpts[11], kpts[12]
            
            mid_ear = ((l_ear[0] + r_ear[0])/2, (l_ear[1] + r_ear[1])/2)
            mid_sh = ((l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2)
            mid_hip = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
            has_kpts = True

    # ==========================================
    # CHẾ ĐỘ 1: FRONT
    # ==========================================
    if current_mode == "FRONT":
        cv2.rectangle(frame, (100, 50), (W-100, H-50), (255, 255, 0), 2) 
        if has_kpts and has_box:
            dy, dx = abs(l_sh[1] - r_sh[1]), abs(l_sh[0] - r_sh[0])
            shoulder_angle = math.degrees(math.atan2(dy, dx))
            dist_moved = calc_dist(mid_hip, prev_pelvis) if prev_pelvis else 0
            prev_pelvis = mid_hip
            torso_dx, torso_dy = abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1])
            torso_tilt = math.degrees(math.atan2(torso_dx, torso_dy)) if torso_dy > 0 else 90
            sh_w = abs(l_sh[0] - r_sh[0])
            sh_h_ratio = sh_w / curr_h if curr_h > 0 else 0
            shift_ratio = abs(mid_sh[0] - mid_hip[0]) / sh_w if sh_w > 0 else 0

            if dist_moved > MOVEMENT_LIMIT:
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Nhuc nhich!", 0, 0, (0, 0, 255)
            elif sh_h_ratio < MIN_SH_RATIO: 
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Chua huong 100% nguc!", 0, 0, (0, 0, 255)
            elif shift_ratio > LATERAL_SHIFT_LIMIT: 
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Trong tam lech / Danh hong!", 0, 0, (0, 0, 255)
            elif torso_tilt > FRONT_TILT_LIMIT or shoulder_angle > TWIST_ANGLE_LIMIT: 
                stable_counter = 0
                pathology_confirm_counter += 1
                status_msg, msg_color = f"PHAT HIEN LECH! Dang xac minh... {int((pathology_confirm_counter/FORCE_LOCK_TIME)*100)}%", (0, 165, 255) 
            else:
                pathology_confirm_counter = 0
                stable_counter += 1
                status_msg, msg_color = f"TU THE HOAN HAO... {int((stable_counter/30)*100)}%", (0, 255, 0)
            
            if stable_counter >= 30 or pathology_confirm_counter >= FORCE_LOCK_TIME: 
                status_msg, msg_color = "DA KHOA DU LIEU [FRONT]!", (0, 255, 0)
                report_data["FRONT"]["sh_angle"] = shoulder_angle
                report_data["FRONT"]["shift_ratio"] = shift_ratio * 100 
                report_data["FRONT"]["status"] = "Da hoan thanh"
            
            cv2.line(frame, (int(l_sh[0]), int(l_sh[1])), (int(r_sh[0]), int(r_sh[1])), (0, 0, 255), 3)
            cv2.line(frame, (int(mid_sh[0]), int(mid_sh[1])), (int(mid_hip[0]), int(mid_hip[1])), (0, 255, 255), 2)
            cv2.line(frame, (int(mid_sh[0]), int(mid_sh[1])), (int(mid_sh[0]), int(mid_hip[1])), (255, 255, 255), 1)

    # ==========================================
    # CHẾ ĐỘ 2: SIDE 
    # ==========================================
    elif current_mode == "SIDE":
        cv2.rectangle(frame, (100, 50), (W-100, H-50), (255, 255, 0), 2)
        if has_kpts and has_box:
            neck_len = calc_dist(mid_ear, mid_sh)
            torso_len = calc_dist(mid_sh, mid_hip)
            neck_ratio = neck_len / torso_len if torso_len > 0 else 0
            torso_dx, torso_dy = abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1])
            torso_tilt = math.degrees(math.atan2(torso_dx, torso_dy)) if torso_dy > 0 else 90
            dist_moved = calc_dist(mid_hip, prev_pelvis) if prev_pelvis else 0
            prev_pelvis = mid_hip
            sh_w = abs(l_sh[0] - r_sh[0])
            sh_h_ratio = sh_w / curr_h if curr_h > 0 else 0

            if dist_moved > MOVEMENT_LIMIT:
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Nhuc nhich!", 0, 0, (0, 0, 255)
            elif sh_h_ratio > MAX_SIDE_SH_RATIO: 
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Chua quay nghieng 100%!", 0, 0, (0, 0, 255)
            elif torso_tilt > SIDE_TILT_LIMIT:
                stable_counter = 0
                pathology_confirm_counter += 1
                status_msg, msg_color = f"NGA LUNG/GU ({torso_tilt:.1f}o)! Xac minh... {int((pathology_confirm_counter/FORCE_LOCK_TIME)*100)}%", (0, 165, 255)
            elif neck_ratio < MIN_NECK_RATIO:
                status_msg, stable_counter, pathology_confirm_counter, msg_color = "LOI: Rut co / Bo vai", 0, 0, (0, 0, 255)
            else:
                pathology_confirm_counter = 0
                stable_counter += 1
                status_msg, msg_color = f"TU THE SIDE CHUAN... {int((stable_counter/30)*100)}%", (0, 255, 0)
                
            if stable_counter >= 30 or pathology_confirm_counter >= FORCE_LOCK_TIME: 
                status_msg, msg_color = "DA KHOA DU LIEU [SIDE]!", (0, 255, 0)
                report_data["SIDE"]["torso_tilt"] = torso_tilt
                report_data["SIDE"]["neck_ratio"] = neck_ratio * 100
                report_data["SIDE"]["status"] = "Da hoan thanh"

            cv2.line(frame, (int(mid_ear[0]), int(mid_ear[1])), (int(mid_sh[0]), int(mid_sh[1])), (255, 0, 255), 3)
            cv2.line(frame, (int(mid_sh[0]), int(mid_sh[1])), (int(mid_hip[0]), int(mid_hip[1])), (0, 255, 255), 3)

    # ==========================================
    # CHẾ ĐỘ 3: ADAM TEST & MIDAS INFERENCE
    # ==========================================
    elif current_mode == "ADAM":
        cv2.rectangle(frame, (50, int(H*(1-MAX_FILL))), (W-50, int(H*(1-MIN_FILL))), (255, 255, 255), 1)
        if has_box:
            curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            fill_ratio = curr_h / H
            is_dist_ok = MIN_FILL <= fill_ratio <= MAX_FILL
            bbox_color = (0, 255, 0) if is_dist_ok else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

            if not is_dist_ok:
                status_msg, stable_counter, msg_color = f"B0: SAI KHOANG CACH", 0, (0, 165, 255) 
            else:
                if adam_state == "STEP1_SIDE":
                    stable_counter += 1
                    status_msg, msg_color = f"B1: DANG LUU MOC CHIEU CAO... {int((stable_counter/25)*100)}%", (0, 255, 255)
                    if stable_counter >= 25:
                        baseline_box_height, adam_state, stable_counter = curr_h, "STEP2_BEND", 0
                elif adam_state == "STEP2_BEND":
                    bend_ratio = curr_h / baseline_box_height if baseline_box_height > 0 else 1.0
                    if bend_ratio < BEND_TARGET:
                        stable_counter += 1
                        status_msg, msg_color = f"B2: XAC NHAN GOC GAP... {int((stable_counter/15)*100)}%", (0, 255, 0)
                        if stable_counter >= 15: adam_state, stable_counter = "STEP3_TURN", 0
                    else: status_msg, stable_counter, msg_color = "B2: CUI GAP NGUOI XUONG!", 0, (0, 165, 255)
                elif adam_state == "STEP3_TURN":
                    stable_counter += 1
                    status_msg, msg_color = f"B3: CHUAN XAC! Dang khoa... {int((stable_counter/30)*100)}%", (0, 255, 0)
                    if stable_counter >= 30: 
                        adam_state = "READY"
                        
                elif adam_state == "READY":
                    roi_y1 = max(0, y1)
                    roi_y2 = min(H, y1 + int(curr_h * 0.70)) 
                    cv2.rectangle(frame, (x1, roi_y1), (x2, roi_y2), (255, 0, 255), 4)
                    
                    # LOGIC MIDAS XỬ LÝ ẢNH CROP NGAY TẠI ĐÂY
                    if report_data["ADAM"]["status"] == "Chua do":
                        status_msg, msg_color = "MIDAS DANG PHAN TICH HEATMAP...", (255, 0, 255)
                        back_crop = frame[roi_y1:roi_y2, x1:x2]
                        
                        if back_crop.size > 0:
                            # Chuyển ảnh sang dạng MiDaS đọc được
                            img_rgb = cv2.cvtColor(back_crop, cv2.COLOR_BGR2RGB)
                            input_batch = midas_transforms(img_rgb).to(device)

                            with torch.no_grad():
                                prediction = midas(input_batch)
                                prediction = torch.nn.functional.interpolate(
                                    prediction.unsqueeze(1),
                                    size=img_rgb.shape[:2],
                                    mode="bicubic",
                                    align_corners=False,
                                ).squeeze()
                            
                            # Lấy mảng Depth Map
                            depth_map = prediction.cpu().numpy()
                            
                            # Thuật toán chia đôi lưng so sánh chênh lệch
                            h_depth, w_depth = depth_map.shape
                            mid_x = w_depth // 2
                            left_side = depth_map[:, :mid_x]
                            right_side = depth_map[:, mid_x:]
                            
                            left_mean = np.mean(left_side)
                            right_mean = np.mean(right_side)
                            
                            diff = abs(left_mean - right_mean)
                            max_val = max(left_mean, right_mean) + 1e-6 # Chống lỗi chia cho 0
                            asym_index = (diff / max_val) * 100
                            
                            # Cập nhật kết quả vào Report
                            report_data["ADAM"]["asym_index"] = asym_index
                            report_data["ADAM"]["status"] = "Da hoan thanh"
                    else:
                        status_msg, msg_color = "DA TINH TOAN XONG! Bam [4] de xem Report", (0, 255, 0)

    # ==========================================
    # CHẾ ĐỘ 4: REPORT 
    # ==========================================
    elif current_mode == "REPORT":
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (W-50, H-50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        cv2.putText(frame, "--- BAO CAO TAM SOAT LAM SANG ---", (W//2 - 250, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

        # 1. FRONT
        f_data = report_data["FRONT"]
        cv2.putText(frame, f"[FRONT] Trang thai: {f_data['status']}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if f_data['status'] == "Da hoan thanh":
            cv2.putText(frame, f"- Lech vai: {f_data['sh_angle']:.1f} do", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, f"- Lech hong: {f_data['shift_ratio']:.1f}%", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            if f_data['sh_angle'] > TWIST_ANGLE_LIMIT:
                cv2.putText(frame, "=> KET LUAN: Canh bao Veo Cot Song Nguc", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif f_data['shift_ratio'] > (LATERAL_SHIFT_LIMIT*100):
                cv2.putText(frame, "=> KET LUAN: Canh bao Lech Khung Chau", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "=> KET LUAN: Truc Co The Binh Thuong", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2. SIDE
        s_data = report_data["SIDE"]
        cv2.putText(frame, f"[SIDE] Trang thai: {s_data['status']}", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if s_data['status'] == "Da hoan thanh":
            cv2.putText(frame, f"- Nga lung: {s_data['torso_tilt']:.1f} do", (150, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            if s_data['torso_tilt'] > SIDE_TILT_LIMIT:
                cv2.putText(frame, "=> KET LUAN: Canh bao Tat Gu Lung (Kyphosis)", (150, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "=> KET LUAN: Do cong Sinh ly Binh thuong", (150, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. ADAM TEST 
        a_data = report_data["ADAM"]
        cv2.putText(frame, f"[ADAM] Trang thai: {a_data['status']}", (100, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if a_data['status'] == "Da hoan thanh":
            asym = a_data['asym_index']
            cv2.putText(frame, f"- Do lech 2 ben lung: {asym:.1f}%", (150, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            if asym > 10.0:
                cv2.putText(frame, "=> KET LUAN: Nguy co VEO COT SONG CAO! Chi dinh X-Quang", (150, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif asym > 5.0:
                cv2.putText(frame, "=> KET LUAN: Nguy co nhe. Can theo doi", (150, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "=> KET LUAN: Lung can doi, Khong phat hien khoi go", (150, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ==========================================
    # GIAO DIỆN CHUNG & MENU PHÍM BẤM
    # ==========================================
    cv2.rectangle(frame, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(frame, "[1] FRONT", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if current_mode=="FRONT" else (150, 150, 150), 2)
    cv2.putText(frame, "[2] SIDE", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if current_mode=="SIDE" else (150, 150, 150), 2)
    cv2.putText(frame, "[3] ADAM", (280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if current_mode=="ADAM" else (150, 150, 150), 2)
    cv2.putText(frame, "[4] REPORT", (410, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if current_mode=="REPORT" else (150, 150, 150), 2)
    
    cv2.putText(frame, status_msg, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
    cv2.putText(frame, "R: Reset | Q: Quit", (W-220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Gatekeeper Omni V16 - Full MiDaS Integration", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): current_mode, stable_counter, pathology_confirm_counter, prev_pelvis = "FRONT", 0, 0, None
    elif key == ord('2'): current_mode, stable_counter, pathology_confirm_counter, prev_pelvis = "SIDE", 0, 0, None
    elif key == ord('3'): 
        current_mode = "ADAM"
        adam_state, stable_counter, prev_box_center = "STEP1_SIDE", 0, None
    elif key == ord('4'):
        current_mode = "REPORT"
        status_msg, msg_color = "BANG TONG HOP KET QUA", (255, 255, 0)
    elif key == ord('r'):
        stable_counter, pathology_confirm_counter, prev_pelvis = 0, 0, None
        if current_mode == "ADAM": adam_state, prev_box_center = "STEP1_SIDE", None
        report_data = {"FRONT": {"status": "Chua do"}, "SIDE": {"status": "Chua do"}, "ADAM": {"status": "Chua do"}}

cap.release()
cv2.destroyAllWindows()