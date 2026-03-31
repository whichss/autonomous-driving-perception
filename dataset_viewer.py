import cv2
import numpy as np
from pypcd4 import PointCloud
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============ 경로 설정 ============
SAMPLE_DIR = os.path.expanduser("~/Downloads/sample")
RAW_DIR = os.path.join(SAMPLE_DIR, "01.원천데이터")
LABEL_DIR = os.path.join(SAMPLE_DIR, "02.라벨링데이터")

# ============ 데이터 로드 ============
print("=" * 50)
print("  자율주행 데이터셋 뷰어 + AI 분석")
print("=" * 50)

# 1. 4방향 카메라 이미지 로드
print("\n[1/5] 카메라 이미지 로드...")
images = {}
for direction in ['F', 'B', 'L', 'R']:
    path = os.path.join(RAW_DIR, f"가시광이미지/image_{direction}/CK_B13_R03_day_clear_19033645_{direction}.png")
    img = cv2.imread(path)
    if img is not None:
        images[direction] = img
        print(f"  {direction}: {img.shape}")

# 2. LiDAR 포인트클라우드 로드
print("\n[2/5] LiDAR 포인트클라우드 로드...")
pcd_path = os.path.join(RAW_DIR, "라이다/LK_B13_R03_day_clear_19033645.pcd")
pc = PointCloud.from_path(pcd_path)
points = pc.numpy()
print(f"  포인트 수: {len(points)}")
print(f"  필드: {pc.fields}")

# 3. 라벨링 데이터 로드
print("\n[3/5] 라벨링 데이터 로드...")
label_path = os.path.join(LABEL_DIR, "가시광이미지/image_F/CK_B13_R03_day_clear_19033645_F.png.json")
with open(label_path, 'r') as f:
    label_data = json.load(f)

# 카테고리 맵
cat_map = {}
for cat in label_data['category']:
    cat_map[cat['id']] = cat['name']

annotations = label_data['annotations']
print(f"  어노테이션 수: {len(annotations)}")
for ann in annotations:
    cat_name = cat_map.get(ann['category_id'], 'unknown')
    print(f"    - {cat_name}: bbox={ann['bbox']}")

# 4. AI 모델 로드
print("\n[4/5] AI 모델 로드...")
yolo = YOLO('yolo11s.pt')
depth_pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='mps')
print("  YOLO11s + Depth Anything v2 로드 완료")

# 5. 열화상 이미지
print("\n[5/5] 열화상 이미지 로드...")
thermal_path = os.path.join(RAW_DIR, "열화상이미지/thermal/TK_B13_R03_day_clear_19033645.png")
thermal = cv2.imread(thermal_path)
if thermal is not None:
    print(f"  열화상: {thermal.shape}")

# ============ AI 분석 ============
print("\n" + "=" * 50)
print("  AI 분석 시작")
print("=" * 50)

# 전방 이미지에 YOLO 실행
front_img = images['F']
print("\n[AI] YOLO11 객체 인식 중...")
yolo_results = yolo(front_img, verbose=False)

# YOLO 결과
yolo_display = front_img.copy()
yolo_objects = []
for r in yolo_results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo.names[cls]
        if conf < 0.3:
            continue
        yolo_objects.append({'label': label, 'conf': conf, 'bbox': [x1, y1, x2, y2]})
        cv2.rectangle(yolo_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(yolo_display, f"{label} {conf:.0%}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"  YOLO 감지: {len(yolo_objects)}개")
for obj in yolo_objects:
    print(f"    - {obj['label']} ({obj['conf']:.0%})")

# 깊이 추정
print("\n[AI] Depth Anything v2 깊이 추정 중...")
small = cv2.resize(front_img, (512, 384))
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)
depth_result = depth_pipe(pil_img)
depth_map = np.array(depth_result['depth'], dtype=np.float32)
depth_resized = cv2.resize(depth_map, (front_img.shape[1], front_img.shape[0]))
depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
print("  깊이 추정 완료")

# 라벨링 바운딩박스 그리기
label_display = front_img.copy()
LABEL_COLORS = {
    'Car': (255, 0, 0), 'TruckBus': (255, 128, 0), 'Two-wheel Vehicle': (0, 255, 255),
    'Personal Mobility': (255, 0, 255), 'Adult': (0, 0, 255), 'Kid student': (0, 0, 200),
    'Traffic Sign': (128, 255, 0), 'Traffic Light': (0, 255, 128),
    'Speed bump': (200, 200, 0), 'Parking space': (200, 0, 200), 'Crosswalk': (100, 100, 255),
}
for ann in annotations:
    cat_name = cat_map.get(ann['category_id'], 'unknown')
    if '-b' in cat_name:  # 바운딩박스 카테고리 스킵 (세그멘테이션용)
        continue
    x, y, w, h = ann['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)
    color = LABEL_COLORS.get(cat_name, (200, 200, 200))
    cv2.rectangle(label_display, (x, y), (x + w, y + h), color, 2)
    cv2.putText(label_display, cat_name, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ============ 시각화 ============
print("\n" + "=" * 50)
print("  시각화")
print("=" * 50)

# --- 화면 1: 4방향 카메라 ---
print("\n[뷰] 4방향 카메라 뷰 표시 중...")
h, w = 360, 640
top = np.hstack([cv2.resize(images.get('L', np.zeros((h, w, 3), np.uint8)), (w, h)),
                 cv2.resize(images.get('F', np.zeros((h, w, 3), np.uint8)), (w, h)),
                 cv2.resize(images.get('R', np.zeros((h, w, 3), np.uint8)), (w, h))])
cv2.putText(top, 'LEFT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(top, 'FRONT', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(top, 'RIGHT', (w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('4-Direction Camera View', top)

# --- 화면 2: GT 라벨 vs YOLO 비교 ---
print("[뷰] GT vs YOLO 비교 표시 중...")
compare_h, compare_w = 540, 960
label_r = cv2.resize(label_display, (compare_w, compare_h))
yolo_r = cv2.resize(yolo_display, (compare_w, compare_h))
cv2.putText(label_r, 'Ground Truth (Label)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(yolo_r, 'YOLO11 (AI)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
compare = np.hstack([label_r, yolo_r])
cv2.imshow('Ground Truth vs YOLO11', compare)

# --- 화면 3: 깊이 추정 ---
print("[뷰] 깊이 추정 표시 중...")
depth_display = np.hstack([
    cv2.resize(front_img, (compare_w, compare_h)),
    cv2.resize(depth_colored, (compare_w, compare_h))
])
cv2.putText(depth_display, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(depth_display, 'Depth Anything v2', (compare_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('AI Depth Estimation', depth_display)

# --- 화면 4: LiDAR 3D 포인트클라우드 (OpenCV로 표시) ---
print("[뷰] LiDAR 3D 포인트클라우드 표시 중...")
fig = plt.figure(figsize=(8, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

if len(points) > 10000:
    idx = np.random.choice(len(points), 10000, replace=False)
    pts = points[idx]
else:
    pts = points

x = pts[:, 0] if pts.shape[1] > 0 else np.zeros(len(pts))
y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts))

colors = plt.cm.viridis((z - z.min()) / (z.max() - z.min() + 1e-6))
ax.scatter(x, y, z, c=colors, s=0.5, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'LiDAR Point Cloud ({len(points)} points)')
plt.tight_layout()

# matplotlib → OpenCV 이미지로 변환
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
lidar_img = np.asarray(buf)[:, :, :3]
lidar_img = cv2.cvtColor(lidar_img, cv2.COLOR_RGB2BGR)
plt.close(fig)

cv2.imshow('LiDAR Point Cloud', lidar_img)

# --- 화면 5: 열화상 ---
if thermal is not None:
    print("[뷰] 열화상 이미지 표시 중...")
    cv2.imshow('Thermal Image', cv2.resize(thermal, (640, 480)))

print("\n" + "=" * 50)
print("  아무 키나 누르면 종료")
print("=" * 50)

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')

# ============ 분석 요약 ============
print("\n" + "=" * 50)
print("  분석 요약")
print("=" * 50)
print(f"\n  데이터셋: 자율주행 도로 데이터")
print(f"  카메라: 4방향 (F/B/L/R) 1920x1080")
print(f"  LiDAR: {len(points)} 포인트")
print(f"  GT 어노테이션: {len(annotations)}개 객체")
print(f"  YOLO11 감지: {len(yolo_objects)}개 객체")
print(f"  날씨: {label_data.get('weather', 'N/A')}")
print(f"  시간: {label_data.get('time', 'N/A')}")
print(f"  도로: {label_data.get('road', 'N/A')}")
print(f"  시나리오: {label_data.get('scenario', 'N/A')}")
print("=" * 50)
