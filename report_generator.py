"""
report_generator.py — Intelligent Medical Measurement
Tạo PDF báo cáo lâm sàng với font DejaVuSans (hỗ trợ tiếng Việt đầy đủ).
"""
import cv2, numpy as np, base64, io, os
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

# ── Font ──────────────────────────────────────────────────────────────────────
_F  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if os.path.exists(_F):
    pdfmetrics.registerFont(TTFont("VN",      _F))
    pdfmetrics.registerFont(TTFont("VN-Bold", _FB if os.path.exists(_FB) else _F))
    FN, FB = "VN", "VN-Bold"
else:
    FN, FB = "Helvetica", "Helvetica-Bold"

# ── Màu sắc (bảng màu sáng / y tế chuyên nghiệp) ────────────────────────────
C_BG       = colors.HexColor("#F8FAFC")   # nền trang
C_WHITE    = colors.white
C_PRIMARY  = colors.HexColor("#1A56DB")   # xanh dương đậm
C_PRIMARY2 = colors.HexColor("#1E429F")   # xanh navy
C_ACCENT   = colors.HexColor("#0694A2")   # teal
C_OK       = colors.HexColor("#057A55")   # xanh lá đậm
C_OK_BG    = colors.HexColor("#F3FAF7")
C_WARN     = colors.HexColor("#C81E1E")   # đỏ đậm
C_WARN_BG  = colors.HexColor("#FDF2F2")
C_AMBER    = colors.HexColor("#C27803")
C_HEADER   = colors.HexColor("#1E3A5F")   # header bảng
C_ROW1     = colors.HexColor("#EFF6FF")
C_ROW2     = colors.HexColor("#FFFFFF")
C_BORDER   = colors.HexColor("#CBD5E1")
C_BORDER2  = colors.HexColor("#93C5FD")
C_TEXT     = colors.HexColor("#1E293B")
C_MUTED    = colors.HexColor("#64748B")
C_LABEL    = colors.HexColor("#374151")
C_CARD     = colors.HexColor("#F1F5F9")
C_CARD2    = colors.HexColor("#DBEAFE")

# ── Helpers ───────────────────────────────────────────────────────────────────
_DONE_VALUES = {"Đã hoàn thành", "Da hoan thanh", "done", "DONE", "completed"}

def _done(section: dict) -> bool:
    return section.get("status", "") in _DONE_VALUES

def b64_to_np(b64: str) -> Optional[np.ndarray]:
    if not b64: return None
    try:
        data = base64.b64decode(b64)
        arr  = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None

def np_to_rl(img_bgr: np.ndarray, max_w: float, max_h: float) -> Optional[RLImage]:
    if img_bgr is None: return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        h, w = img_bgr.shape[:2]
        r = min(max_w / w, max_h / h)
        return RLImage(buf, width=w * r, height=h * r)
    except: return None

def score_bar(val: float, max_val: float, warn_val: float,
              w: float = 140, h: float = 8) -> Drawing:
    d = Drawing(w, h + 8)
    # nền
    d.add(Rect(0, 4, w, h, fillColor=colors.HexColor("#E2E8F0"),
               strokeColor=C_BORDER, strokeWidth=0.5, rx=2, ry=2))
    # fill
    fw = w * min(val / max_val, 1.0)
    if fw > 0:
        fc = C_WARN if val > warn_val else C_OK
        d.add(Rect(0, 4, fw, h, fillColor=fc, strokeColor=None, rx=2, ry=2))
    # vạch ngưỡng
    tx = w * (warn_val / max_val)
    d.add(Line(tx, 0, tx, h + 8, strokeColor=C_AMBER, strokeWidth=1.5))
    return d

# ── Style factory ─────────────────────────────────────────────────────────────
def S(name, **kw):
    base = dict(fontName=FN, fontSize=9, textColor=C_TEXT, leading=13)
    base.update(kw)
    return ParagraphStyle(name, **base)

def P(text, style):
    """Tạo Paragraph với font VN để đảm bảo tiếng Việt hiển thị đúng."""
    return Paragraph(text, style)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def generate_pdf_report(report_data: dict, patient_info: dict = None) -> bytes:
    patient_info = patient_info or {}
    buf = io.BytesIO()
    W, H = A4
    CW = W - 3.2 * cm

    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm)

    # ── Styles ────────────────────────────────────────────────────────────────
    s_title   = S("tt", fontSize=22, fontName=FB, textColor=C_PRIMARY2,
                  alignment=TA_CENTER, spaceBefore=2, spaceAfter=6)
    s_brand   = S("br", fontSize=11, fontName=FB, textColor=C_ACCENT,
                  alignment=TA_CENTER, spaceBefore=0, spaceAfter=3)
    s_sub     = S("sb", fontSize=8, textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=6)
    s_h2      = S("h2", fontSize=11, fontName=FB, textColor=C_PRIMARY2,
                  spaceBefore=8, spaceAfter=3)
    s_label   = S("lb", fontSize=7, fontName=FB, textColor=C_MUTED,
                  spaceAfter=1, leading=10)
    s_val_ok  = S("vo", fontSize=14, fontName=FB, textColor=C_OK, leading=18)
    s_val_wn  = S("vw", fontSize=14, fontName=FB, textColor=C_WARN, leading=18)
    s_body    = S("bd", fontSize=9, textColor=C_TEXT, leading=13)
    s_small   = S("sm", fontSize=7.5, textColor=C_MUTED, leading=11)
    s_center  = S("ct", fontSize=7.5, textColor=C_MUTED, alignment=TA_CENTER)
    s_info_lbl= S("il", fontSize=8, fontName=FB, textColor=C_PRIMARY2, leading=11)
    s_info_val= S("iv", fontSize=9, fontName=FN, textColor=C_TEXT, leading=13)
    s_expl    = S("ex", fontSize=7.5, textColor=C_TEXT, leading=11,
                  backColor=colors.HexColor("#EFF6FF"),
                  borderColor=C_BORDER2, borderWidth=0.8,
                  borderPad=5, spaceAfter=3)
    s_warn_box = S("wb", fontSize=9, fontName=FB, textColor=C_WARN,
                   backColor=C_WARN_BG, borderColor=C_WARN,
                   borderWidth=1, borderPad=8)
    s_ok_box   = S("ob", fontSize=9, fontName=FB, textColor=C_OK,
                   backColor=C_OK_BG, borderColor=C_OK,
                   borderWidth=1, borderPad=8)
    s_footer   = S("ft", fontSize=6.5, textColor=C_MUTED, alignment=TA_CENTER)
    s_tag_warn = S("tw", fontSize=8, fontName=FB, textColor=C_WARN)
    s_tag_ok   = S("to", fontSize=8, fontName=FB, textColor=C_OK)
    s_tbl_hdr  = S("th", fontSize=8, fontName=FB, textColor=C_WHITE)
    s_tbl_cell = S("tc", fontSize=8.5, fontName=FN, textColor=C_TEXT)

    story = []
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── HEADER ────────────────────────────────────────────────────────────────
    story.append(P("BÁO CÁO LÂM SÀNG", s_title))
    story.append(HRFlowable(width="60%", thickness=2, color=C_ACCENT,
                            hAlign="CENTER", spaceAfter=4))
    story.append(P("Intelligent Medical Measurement", s_brand))
    story.append(P("AI Scoliosis Screening System — Phân tích tư thế cột sống", s_sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_PRIMARY, spaceAfter=8))

    # ── Thông tin bệnh nhân ───────────────────────────────────────────────────
    name   = patient_info.get("name",   "—")
    age    = patient_info.get("age",    "—")
    gender = patient_info.get("gender", "—")
    height = patient_info.get("height", "—")
    weight = patient_info.get("weight", "—")
    bmi = "—"
    try:
        h_m = float(height) / 100
        bmi = f"{float(weight) / (h_m**2):.1f} kg/m²"
    except: pass

    # Bảng thông tin: 2 cột (label | value) x2 side-by-side
    LW = 2.6*cm   # label col
    VW = (CW - 2*LW - 0.4*cm) / 2  # value col
    def mk(txt, st): return P(txt, st)

    info_data = [
        [mk("Họ và tên",  s_info_lbl), mk(str(name),        s_info_val),
         mk("Ngày đo",    s_info_lbl), mk(now,               s_info_val)],
        [mk("Tuổi",       s_info_lbl), mk(str(age),          s_info_val),
         mk("Giới tính",  s_info_lbl), mk(str(gender),       s_info_val)],
        [mk("Chiều cao",  s_info_lbl), mk(f"{height} cm",    s_info_val),
         mk("Cân nặng",   s_info_lbl), mk(f"{weight} kg",    s_info_val)],
        [mk("BMI",        s_info_lbl), mk(bmi,               s_info_val),
         mk("", s_small), mk("", s_small)],
    ]

    info_tbl = Table(info_data, colWidths=[LW, VW, LW, VW])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_WHITE),
        ("BACKGROUND",    (0,0), (0,-1),  colors.HexColor("#EFF6FF")),
        ("BACKGROUND",    (2,0), (2,-1),  colors.HexColor("#EFF6FF")),
        ("BOX",           (0,0), (-1,-1), 1.0, C_PRIMARY),
        ("LINEBELOW",     (0,0), (-1,-2), 0.4, C_BORDER),
        ("LINEAFTER",     (1,0), (1,-1),  0.8, C_PRIMARY),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))

    half = CW / 2 - 0.3 * cm

    # ── Section helper ────────────────────────────────────────────────────────
    def make_section(title, img_bgr, right_cells):
        # Tiêu đề section với thanh màu bên trái
        story.append(P(title, s_h2))
        left = []
        if img_bgr is not None:
            rl = np_to_rl(img_bgr, half - 0.5*cm, 8*cm)
            if rl:
                left.append(rl)
        left += [Spacer(1, 3), P("Ảnh chụp tại thời điểm đo", s_center)]

        tbl = Table([[left, right_cells]], colWidths=[half + 0.1*cm, half + 0.5*cm])
        tbl.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BACKGROUND",    (0,0), (0,0),   C_CARD),
            ("BOX",           (0,0), (0,0),   0.5, C_BORDER),
        ]))
        story.append(tbl)

    # ── 1. FRONT ──────────────────────────────────────────────────────────────
    f = report_data.get("FRONT", {})
    if _done(f):
        sh   = f.get("sh_angle", 0)
        warn = sh > 4.0
        vs   = s_val_wn if warn else s_val_ok
        make_section(
            "① ĐO MẶT TRƯỚC — Phân tích đường vai",
            b64_to_np(f.get("cap_b64", "")),
            [
                P("CHỈ SỐ ĐO ĐƯỢC", s_label),
                P(f"Góc lệch vai: <b>{sh:.1f}°</b>", vs),
                Spacer(1, 4),
                score_bar(sh, 20, 4, CW * 0.37),
                Spacer(1, 2),
                P("Ngưỡng cảnh báo: 4° &nbsp;|&nbsp; Vạch vàng = giới hạn", s_small),
                Spacer(1, 8),
                P("KẾT LUẬN", s_label),
                P("CẢNH BÁO LỆCH VAI" if warn else "Bình thường", vs),
                Spacer(1, 8),
                P("HỆ THỐNG ĐO NHƯ THẾ NÀO?", s_label),
                P("Camera ghi lại ảnh người dùng từ phía trước. "
                  "AI tự động xác định vị trí hai vai và vẽ đường thẳng nối chúng. "
                  "Nếu đường đó bị nghiêng — vai bên cao bên thấp — hệ thống đo góc lệch đó. "
                  "Góc càng lớn, vai càng bị lệch.", s_expl),
            ]
        )
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))

    # ── 2. SIDE ───────────────────────────────────────────────────────────────
    s = report_data.get("SIDE", {})
    if _done(s):
        kyph = s.get("kyphotic_angle", s.get("torso_tilt", 0))  # fallback for old data
        warn = kyph > 8.0
        vs   = s_val_wn if warn else s_val_ok
        make_section(
            "② ĐO MẶT NGHIÊNG — Phân tích đường lưng",
            b64_to_np(s.get("cap_b64", "")),
            [
                P("CHỈ SỐ ĐO ĐƯỢC", s_label),
                P(f"Góc gù lưng: <b>{kyph:.1f}°</b>", vs),
                Spacer(1, 4),
                score_bar(kyph, 30, 8, CW * 0.37),
                Spacer(1, 2),
                P("Ngưỡng cảnh báo: 8° &nbsp;|&nbsp; Vạch vàng = giới hạn", s_small),
                Spacer(1, 8),
                P("KẾT LUẬN", s_label),
                P("CẢNH BÁO GÙ LƯNG" if warn else "Bình thường", vs),
                Spacer(1, 8),
                P("HỆ THỐNG ĐO NHƯ THẾ NÀO?", s_label),
                P("Camera ghi lại ảnh người dùng từ phía bên cạnh. "
                  "AI xác định 3 điểm: cổ, vai và hông, rồi đo góc uốn của cột sống tại điểm vai. "
                  "Người đứng thẳng hoàn toàn góc này = 0°. "
                  "Góc càng lớn = cột sống càng được cuộn về phía trước (gù lưng).", s_expl),
            ]
        )
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))

    # ── 3. ADAM ───────────────────────────────────────────────────────────────
    a = report_data.get("ADAM", {})
    if _done(a):
        asym        = a.get("asym_index", 0)
        warn        = asym > 10.0
        vs          = s_val_wn if warn else s_val_ok
        depth_img   = b64_to_np(a.get("depth_b64",   ""))
        overlay_img = b64_to_np(a.get("overlay_b64", ""))
        raw_img     = b64_to_np(a.get("raw_b64",     ""))

        story.append(P("③ ADAM FORWARD BENDING TEST — Phân tích cột sống", s_h2))

        left = []
        if depth_img is not None:
            rl = np_to_rl(depth_img, half - 0.5*cm, 8*cm)
            if rl: left.append(rl)
        left += [Spacer(1, 3), P("Depth map MiDaS (MAGMA colormap)", s_center)]

        right = [
            P("CHỈ SỐ ĐỘ LỆCH CỘT SỐNG", s_label),
            P(f"Asym Index: <b>{asym:.1f}%</b>", vs),
            Spacer(1, 4),
            score_bar(asym, 30, 10, CW * 0.37),
            Spacer(1, 2),
            P("Ngưỡng: 10%  |  Vạch vàng = giới hạn cảnh báo", s_small),
            Spacer(1, 8),
            P("KẾT LUẬN", s_label),
            P("NGUY CƠ VẸO CỘT SỐNG" if warn else "Trong giới hạn bình thường", vs),
            Spacer(1, 10),
            P("PHƯƠNG PHÁP ĐO", s_label),
            P("Rib Hump Score: So sánh đỉnh độ sâu hai bên cột sống mỗi hàng<br/>"
              "Spine Line: Xác định từ điểm giữa hai vai (YOLO keypoint)", s_small),
        ]

        tbl = Table([[left, right]], colWidths=[half + 0.1*cm, half + 0.5*cm])
        tbl.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("BACKGROUND",    (0,0), (0,0),   C_CARD),
            ("BOX",           (0,0), (0,0),   0.5, C_BORDER),
        ]))
        story.append(tbl)

        # Overlay + raw
        story.append(Spacer(1, 6))
        sw = CW / 2 - 0.4 * cm
        ov_c, raw_c = [], []
        if overlay_img is not None:
            rl = np_to_rl(overlay_img, sw - 0.4*cm, 5.5*cm)
            if rl: ov_c.append(rl)
        ov_c.append(P("Overlay depth lên ảnh gốc", s_center))
        if raw_img is not None:
            rl = np_to_rl(raw_img, sw - 0.4*cm, 5.5*cm)
            if rl: raw_c.append(rl)
        raw_c.append(P("Ảnh gốc tại thời điểm scan", s_center))

        img_tbl = Table([[ov_c, raw_c]], colWidths=[sw + 0.3*cm, sw + 0.3*cm])
        img_tbl.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
            ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ]))
        story.append(img_tbl)

        story.append(Spacer(1, 6))
        story.append(P("HỆ THỐNG ĐO NHƯ THẾ NÀO?", s_label))
        story.append(P(
            "<b>Bước 1 — Chụp ảnh chiều sâu:</b> Camera phân tích ảnh người cúi về phía trước "
            "để tạo ra bản đồ độ lồi lõm của lưng. Vùng màu cam/vàng = nhô ra gần camera hơn, "
            "vùng màu xanh = phẳng hơn hoặc lõm vào.<br/>"
            "<b>Bước 2 — Xác định cột sống:</b> AI tìm đường chạy dọc giữa lưng — đó là vị trí cột sống. "
            "Đường này được dùng làm chuẩn để so sánh hai bên lưng trái và phải.<br/>"
            "<b>Bước 3 — Phát hiện lệch:</b> Hệ thống so sánh độ lồi của lưng bên trái và bên phải cột sống. "
            "Nếu một bên nhô cao hơn rõ rệt (như bướu sườn), đó là dấu hiệu cột sống bị vẹo.<br/>"
            "<b>Chỉ số Asym càng cao = hai bên lưng càng mất cân bằng = nguy cơ vẹo cột sống càng lớn.</b>",
            s_expl))

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))

    # ── TỔNG KẾT ──────────────────────────────────────────────────────────────
    story.append(P("TỔNG KẾT &amp; KHUYẾN NGHỊ", s_h2))

    f_done = _done(report_data.get("FRONT", {}))
    s_done = _done(report_data.get("SIDE",  {}))
    a_done = _done(report_data.get("ADAM",  {}))

    any_warn  = False
    warn_rows = []
    data_rows = []   # (label, value, conclusion, is_warn)

    if f_done:
        sh = report_data["FRONT"].get("sh_angle", 0)
        w  = sh > 4.0
        if w: any_warn = True
        data_rows.append(("Mặt trước", f"{sh:.1f}°",
                          "CẢNH BÁO LỆCH VAI" if w else "Bình thường", w))
    if s_done:
        kyph = report_data["SIDE"].get("kyphotic_angle", report_data["SIDE"].get("torso_tilt", 0))
        w    = kyph > 8.0
        if w: any_warn = True
        data_rows.append(("Ảnh nghiêng", f"{kyph:.1f}°",
                          "CẢNH BÁO GÙ LƯNG" if w else "Bình thường", w))
    if a_done:
        asym = report_data["ADAM"].get("asym_index", 0)
        w    = asym > 10.0
        if w: any_warn = True
        data_rows.append(("Adam Test", f"{asym:.1f}%",
                          "NGUY CƠ VẸO CỘT SỐNG" if w else "Bình thường", w))

    if data_rows:
        # Header row
        hdr = [P(t, s_tbl_hdr) for t in ["Hạng mục", "Chỉ số", "Kết luận"]]
        table_data = [hdr]
        for i, (lbl, val, conc, is_w) in enumerate(data_rows):
            cs = s_tag_warn if is_w else s_tag_ok
            table_data.append([
                P(lbl,  s_tbl_cell),
                P(val,  s_tbl_cell),
                P(conc, cs),
            ])
            if is_w:
                warn_rows.append(i + 1)

        sum_tbl = Table(table_data, colWidths=[4*cm, 3.5*cm, CW - 7.5*cm])
        ts = [
            ("BACKGROUND",    (0,0), (-1,0),  C_HEADER),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [C_ROW1, C_ROW2]),
            ("GRID",          (0,0), (-1,-1),  0.4, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1),  7),
            ("BOTTOMPADDING", (0,0), (-1,-1),  7),
            ("LEFTPADDING",   (0,0), (-1,-1),  10),
            ("VALIGN",        (0,0), (-1,-1),  "MIDDLE"),
        ]
        for i in warn_rows:
            ts.append(("BACKGROUND", (0,i), (-1,i), C_WARN_BG))
        sum_tbl.setStyle(TableStyle(ts))
        story.append(sum_tbl)
        story.append(Spacer(1, 8))

    if any_warn:
        story.append(P(
            "CẢNH BÁO: Phát hiện một hoặc nhiều dấu hiệu bất thường. "
            "Khuyến nghị thăm khám bác sĩ chuyên khoa chỉnh hình để kiểm tra chi tiết.",
            s_warn_box))
    else:
        story.append(P(
            "Tất cả chỉ số trong giới hạn bình thường. "
            "Tiếp tục theo dõi định kỳ theo khuyến nghị của bác sĩ.",
            s_ok_box))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4))
    story.append(P(
        f"Báo cáo được tạo tự động bởi Intelligent Medical Measurement — AI Scoliosis Screening | {now} | "
        "Không thay thế chẩn đoán y khoa chuyên nghiệp.",
        s_footer))

    doc.build(story)
    return buf.getvalue()