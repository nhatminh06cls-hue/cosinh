import matplotlib.pyplot as plt
import numpy as np

# Cài đặt font và kích thước chuẩn báo cáo khoa học
plt.rcParams.update({'font.size': 12, 'figure.figsize': (15, 5)})

fig, axs = plt.subplots(1, 3)
fig.suptitle('SƠ ĐỒ VECTOR ĐO LƯỜNG TƯ THẾ HỆ THỐNG AI', fontsize=16, fontweight='bold', y=1.05)

# ---------------------------------------------------------
# HÌNH 1: MẶT PHẲNG NGANG (SCOLIOMETER / TRỤC Z)
# ---------------------------------------------------------
ax1 = axs[0]
ax1.set_title('1. Góc xoay thân (Axial Plane / Scoliometer)', pad=15)
ax1.set_xlim(-1, 5)
ax1.set_ylim(-1, 4)
ax1.grid(True, linestyle='--', alpha=0.5)

# Vẽ trục chuẩn (Ox)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Mặt phẳng chuẩn (Ox)')

# Định nghĩa 2 điểm P_L (Lưng trái) và P_R (Lưng phải)
pl = np.array([0.5, 0.5])
pr = np.array([4, 2.5])

# Vẽ Vector nối 2 lưng
ax1.plot([pl[0], pr[0]], [pl[1], pr[1]], color='blue', linewidth=2.5, marker='o', markersize=8, label=r'Vector lưng ($\vec{v}$)')
ax1.text(pl[0]-0.3, pl[1]+0.2, '$P_L(x_1, z_1)$', fontsize=11)
ax1.text(pr[0], pr[1]+0.2, '$P_R(x_2, z_2)$', fontsize=11)

# Vẽ đường gióng thể hiện Delta Z và Delta X
ax1.plot([pr[0], pr[0]], [pl[1], pr[1]], color='red', linestyle=':', linewidth=2) # Delta Z
ax1.plot([pl[0], pr[0]], [pl[1], pl[1]], color='green', linestyle=':', linewidth=2) # Delta X
ax1.text(pr[0]+0.1, (pl[1]+pr[1])/2, r'$\Delta z$', color='red', fontsize=12)
ax1.text((pl[0]+pr[0])/2, pl[1]-0.4, r'$\Delta x$', color='green', fontsize=12)

# Ký hiệu góc theta
theta = np.linspace(0, np.arctan((pr[1]-pl[1])/(pr[0]-pl[0])), 50)
r = 0.8
ax1.plot(pl[0] + r*np.cos(theta), pl[1] + r*np.sin(theta), color='black')
ax1.text(pl[0] + r + 0.1, pl[1] + 0.2, r'$\theta$', fontsize=14, fontweight='bold')

ax1.set_aspect('equal')
ax1.axis('off')
ax1.legend(loc='upper left')

# ---------------------------------------------------------
# HÌNH 2: MẶT PHẲNG TRÁN (GONIOMETER / LỆCH VAI)
# ---------------------------------------------------------
ax2 = axs[1]
ax2.set_title('2. Góc lệch vai (Coronal Plane / Goniometer)', pad=15)
ax2.set_xlim(-1, 5)
ax2.set_ylim(-4, 1) # Vẽ y âm vì ảnh CV2 thường gốc tọa độ ở trên cùng
ax2.grid(True, linestyle='--', alpha=0.5)

# Vẽ trục chuẩn (Ox ngang, Oy dọc/dây dọi)
ax2.axhline(y=-3, color='gray', linestyle='--', linewidth=1.5, label='Phương ngang tuyệt đối (Ox)')
ax2.axvline(x=2, color='gray', linestyle='--', linewidth=1.5, label='Dây dọi (Oy)')

# Định nghĩa 2 vai
sl = np.array([0.5, -1])  # Vai trái (cao)
sr = np.array([3.5, -2.5]) # Vai phải (thấp)

# Vẽ Vector nối 2 vai
ax2.plot([sl[0], sr[0]], [sl[1], sr[1]], color='purple', linewidth=2.5, marker='s', markersize=8, label=r'Vector vai ($\vec{S}$)')
ax2.text(sl[0]-0.2, sl[1]+0.2, 'Vai Trái', fontsize=11)
ax2.text(sr[0]+0.1, sr[1]-0.3, 'Vai Phải', fontsize=11)

# Vẽ đường gióng phương ngang qua vai trái
ax2.plot([sl[0], 4], [sl[1], sl[1]], color='gray', linestyle=':') 

# Ký hiệu góc alpha
alpha_ang = np.arctan((sr[1]-sl[1])/(sr[0]-sl[0]))
theta2 = np.linspace(alpha_ang, 0, 50)
r2 = 1.0
ax2.plot(sl[0] + r2*np.cos(theta2), sl[1] + r2*np.sin(theta2), color='black')
ax2.text(sl[0] + r2 + 0.1, sl[1] - 0.4, r'$\alpha$', fontsize=14, fontweight='bold')

ax2.set_aspect('equal')
ax2.axis('off')
ax2.legend(loc='lower left')

# ---------------------------------------------------------
# HÌNH 3: MẶT PHẲNG ĐỨNG DỌC (PLUMB LINE / GÓC GÙ CỔ)
# ---------------------------------------------------------
ax3 = axs[2]
ax3.set_title('3. Góc đưa đầu trước (Sagittal Plane / Plumb line)', pad=15)
ax3.set_xlim(-2, 3)
ax3.set_ylim(-5, 1)
ax3.grid(True, linestyle='--', alpha=0.5)

# Vẽ Dây dọi (Vector trọng lực)
ax3.axvline(x=0, color='gray', linestyle='--', linewidth=2, label=r'Dây dọi - Vector $\vec{P}$')
ax3.plot([0, 0], [0, -4.5], color='black', linewidth=2)
ax3.plot(0, -4.5, marker='v', color='black', markersize=10) # Quả dọi

# Định nghĩa điểm Cổ/Vai và Tai
c_vai = np.array([0, -3]) # Điểm gốc
tai = np.array([1.5, -0.5]) # Đầu nhô ra trước

# Vẽ Vector thân trên
ax3.plot([c_vai[0], tai[0]], [c_vai[1], tai[1]], color='orange', linewidth=3, marker='o', markersize=8, label=r'Vector thân trên ($\vec{B}$)')
ax3.text(c_vai[0]-0.8, c_vai[1], 'Khớp vai', fontsize=11)
ax3.text(tai[0]+0.2, tai[1], 'Lỗ tai (Tragus)', fontsize=11)

# Ký hiệu góc beta
beta_ang = np.arctan((tai[0]-c_vai[0])/(tai[1]-c_vai[1]))
theta3 = np.linspace(np.pi/2 - beta_ang, np.pi/2, 50)
r3 = 1.2
ax3.plot(c_vai[0] + r3*np.cos(theta3), c_vai[1] + r3*np.sin(theta3), color='black')
ax3.text(c_vai[0] + 0.3, c_vai[1] + 1.2, r'$\beta$', fontsize=14, fontweight='bold')

ax3.set_aspect('equal')
ax3.axis('off')
ax3.legend(loc='lower left')

plt.tight_layout()
plt.show()