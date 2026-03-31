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

# ============ 경로 설정 ============
SAMPLE_DIR = os.path.expanduser("~/Downloads/sample")
RAW_DIR = os.path.join(SAMPLE_DIR, "01.원천데이터")
LABEL_DIR = os.path.join(SAMPLE_DIR, "02.라벨링데이터")

# ============ 설정 ============
# 위험도 임계값 (깊이 정규화 기준)
DANGER_THRESHOLD = 0.7   # 이 이상이면 위험 (가까움)
WARNING_THRESHOLD = 0.4  # 이 이상이면 주의
# 위험 객체 (자율주행에서 주의해야 할 대상)
DANGER_CLASSES = ['person', 'bicycle', 'motorcycle', 'car', 'truck', 'bus']
# 차선 영역 (화면 비율 기준)
LANE_LEFT = 0.3
LANE_RIGHT = 0.7

# ============ 모델 로딩 ============
print("=" * 60)
print("  자율주행 인지 시스템 (Autonomous Driving Perception)")
print("=" * 60)

print("\n[모델] 로딩 중...")
yolo = YOLO('yolo11s.pt')
depth_pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='mps')
print("[모델] YOLO11s + Depth Anything v2 로드 완료")

# ============ 데이터 로드 ============
print("\n[데이터] 로드 중...")

# 전방 카메라
front_path = os.path.join(RAW_DIR, "가시광이미지/image_F/CK_B13_R03_day_clear_19033645_F.png")
front_img = cv2.imread(front_path)
print(f"  전방 카메라: {front_img.shape}")

# LiDAR
pcd_path = os.path.join(RAW_DIR, "라이다/LK_B13_R03_day_clear_19033645.pcd")
pc = PointCloud.from_path(pcd_path)
lidar_points = pc.numpy()
print(f"  LiDAR: {len(lidar_points)} points")

# 라벨 (GT)
label_path = os.path.join(LABEL_DIR, "가시광이미지/image_F/CK_B13_R03_day_clear_19033645_F.png.json")
with open(label_path, 'r') as f:
    label_data = json.load(f)

frame_h, frame_w = front_img.shape[:2]

# ============ 1. 객체 인식 (YOLO) ============
print("\n[인지] 객체 인식 중...")
results = yolo(front_img, verbose=False)

detected = []
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo.names[cls]
        if conf < 0.3:
            continue
        detected.append({
            'label': label, 'conf': conf,
            'bbox': [x1, y1, x2, y2],
            'cx': (x1 + x2) // 2,
            'cy': (y1 + y2) // 2,
        })

print(f"  감지된 객체: {len(detected)}개")

# ============ 2. 깊이 추정 (Depth Anything) ============
print("[인지] 깊이 추정 중...")
small = cv2.resize(front_img, (512, 384))
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
depth_result = depth_pipe(Image.fromarray(rgb))
depth_map = np.array(depth_result['depth'], dtype=np.float32)
depth_full = cv2.resize(depth_map, (frame_w, frame_h))
depth_norm = cv2.normalize(depth_full, None, 0, 1, cv2.NORM_MINMAX)

# 각 객체에 깊이 할당
for obj in detected:
    obj['depth'] = float(depth_norm[obj['cy'], obj['cx']])

# ============ 3. LiDAR 깊이 보정 ============
print("[인지] LiDAR 데이터 분석 중...")

# LiDAR 포인트 → 전방 투영 (간이)
lidar_x = lidar_points[:, 0]
lidar_y = lidar_points[:, 1]
lidar_z = lidar_points[:, 2]

# 전방 포인트만 필터 (x > 0)
forward_mask = lidar_x > 0
forward_points = lidar_points[forward_mask]

# 거리 계산
if len(forward_points) > 0:
    lidar_distances = np.sqrt(forward_points[:, 0]**2 + forward_points[:, 1]**2 + forward_points[:, 2]**2)
    min_lidar_dist = float(np.min(lidar_distances))
    avg_lidar_dist = float(np.mean(lidar_distances))
    print(f"  전방 LiDAR: 최소 {min_lidar_dist:.1f}m, 평균 {avg_lidar_dist:.1f}m")
else:
    min_lidar_dist = 999
    avg_lidar_dist = 999

# ============ 4. 위험도 판단 ============
print("\n[판단] 위험도 분석 중...")

risk_objects = []
for obj in detected:
    # 위험 클래스인지
    is_danger_class = obj['label'] in DANGER_CLASSES

    # 차선 내에 있는지 (전방 충돌 가능)
    cx_ratio = obj['cx'] / frame_w
    in_lane = LANE_LEFT <= cx_ratio <= LANE_RIGHT

    # 깊이 기반 위험도
    if obj['depth'] > DANGER_THRESHOLD:
        risk_level = "DANGER"
        risk_score = 3
    elif obj['depth'] > WARNING_THRESHOLD:
        risk_level = "WARNING"
        risk_score = 2
    else:
        risk_level = "SAFE"
        risk_score = 1

    # 차선 내 + 위험 클래스면 위험도 상승
    if in_lane and is_danger_class:
        risk_score = min(risk_score + 1, 3)
        if risk_score == 3:
            risk_level = "DANGER"
        elif risk_score == 2:
            risk_level = "WARNING"

    obj['risk_level'] = risk_level
    obj['risk_score'] = risk_score
    obj['in_lane'] = in_lane

    if risk_score >= 2:
        risk_objects.append(obj)

# 정렬 (위험한 순)
risk_objects.sort(key=lambda x: x['risk_score'], reverse=True)
detected.sort(key=lambda x: x.get('risk_score', 0), reverse=True)

# ============ 5. 주행 판단 ============
print("[판단] 주행 경로 분석 중...")

max_risk = max([obj.get('risk_score', 0) for obj in detected]) if detected else 0

if max_risk >= 3:
    driving_action = "EMERGENCY BRAKE"
    action_color = (0, 0, 255)
elif max_risk >= 2:
    driving_action = "SLOW DOWN"
    action_color = (0, 165, 255)
else:
    driving_action = "CLEAR - PROCEED"
    action_color = (0, 255, 0)

# 차선별 위험도
left_risk = sum(1 for obj in detected if obj['cx'] / frame_w < LANE_LEFT and obj.get('risk_score', 0) >= 2)
center_risk = sum(1 for obj in detected if LANE_LEFT <= obj['cx'] / frame_w <= LANE_RIGHT and obj.get('risk_score', 0) >= 2)
right_risk = sum(1 for obj in detected if obj['cx'] / frame_w > LANE_RIGHT and obj.get('risk_score', 0) >= 2)

if center_risk > 0 and left_risk == 0:
    lane_suggestion = "SUGGEST: Change to LEFT lane"
elif center_risk > 0 and right_risk == 0:
    lane_suggestion = "SUGGEST: Change to RIGHT lane"
elif center_risk > 0:
    lane_suggestion = "SUGGEST: BRAKE - No safe lane"
else:
    lane_suggestion = "SUGGEST: Maintain current lane"

# ============ 시각화 ============
print("\n[시각화] 렌더링 중...")

RISK_COLORS = {
    "DANGER": (0, 0, 255),
    "WARNING": (0, 165, 255),
    "SAFE": (0, 255, 0),
}

# --- 메인 뷰: 인지 결과 ---
main_display = front_img.copy()

# 차선 가이드라인
lane_l = int(frame_w * LANE_LEFT)
lane_r = int(frame_w * LANE_RIGHT)
cv2.line(main_display, (lane_l, 0), (lane_l, frame_h), (255, 255, 0), 2)
cv2.line(main_display, (lane_r, 0), (lane_r, frame_h), (255, 255, 0), 2)

# 객체 표시
for obj in detected:
    x1, y1, x2, y2 = obj['bbox']
    risk = obj.get('risk_level', 'SAFE')
    color = RISK_COLORS.get(risk, (200, 200, 200))

    # 바운딩 박스
    thickness = 3 if risk == "DANGER" else 2
    cv2.rectangle(main_display, (x1, y1), (x2, y2), color, thickness)

    # 라벨
    text = f"{obj['label']} [{risk}] {obj['depth']:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(main_display, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
    cv2.putText(main_display, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 위험 객체 깜빡임 효과
    if risk == "DANGER":
        overlay = main_display.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        main_display = cv2.addWeighted(overlay, 0.15, main_display, 0.85, 0)

# --- HUD (Head-Up Display) ---
# 상단 바
hud_h = 80
hud = np.zeros((hud_h, frame_w, 3), dtype=np.uint8) + 30

# 주행 판단
cv2.putText(hud, f"ACTION: {driving_action}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
cv2.putText(hud, lane_suggestion, (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# 차선별 위험도 바
bar_x = frame_w - 400
cv2.putText(hud, "L", (bar_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.rectangle(hud, (bar_x + 20, 15), (bar_x + 120, 45),
              (0, 0, 255) if left_risk > 0 else (0, 255, 0), -1)

cv2.putText(hud, "C", (bar_x + 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.rectangle(hud, (bar_x + 150, 15), (bar_x + 250, 45),
              (0, 0, 255) if center_risk > 0 else (0, 255, 0), -1)

cv2.putText(hud, "R", (bar_x + 260, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.rectangle(hud, (bar_x + 280, 15), (bar_x + 380, 45),
              (0, 0, 255) if right_risk > 0 else (0, 255, 0), -1)

# 객체 수
cv2.putText(hud, f"Objects: {len(detected)} | Risks: {len(risk_objects)}", (bar_x, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

# --- 깊이맵 ---
depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

# --- LiDAR Bird's Eye View ---
bev_size = 400
bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8) + 30
bev_cx, bev_cy = bev_size // 2, bev_size - 30
bev_scale = 8

# 그리드
for i in range(1, 8):
    cv2.circle(bev, (bev_cx, bev_cy), i * bev_scale * 5, (50, 50, 50), 1)
    cv2.putText(bev, f"{i*5}m", (bev_cx + i * bev_scale * 5 + 3, bev_cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

# LiDAR 포인트 (탑뷰: x=좌우, y=전후)
if len(forward_points) > 0:
    sample_idx = np.random.choice(len(forward_points), min(5000, len(forward_points)), replace=False)
    for idx in sample_idx:
        px = int(bev_cx + forward_points[idx, 1] * bev_scale)
        py = int(bev_cy - forward_points[idx, 0] * bev_scale)
        if 0 <= px < bev_size and 0 <= py < bev_size:
            h = forward_points[idx, 2]
            if h > 1:
                c = (0, 200, 255)  # 높은 물체 (차량 등)
            elif h > 0:
                c = (0, 150, 0)    # 중간 (사람 등)
            else:
                c = (100, 100, 100) # 지면
            bev[py, px] = c

# 자차 표시
pts = np.array([[bev_cx, bev_cy - 12], [bev_cx - 8, bev_cy + 6], [bev_cx + 8, bev_cy + 6]])
cv2.fillPoly(bev, [pts], (255, 255, 255))
cv2.putText(bev, "LiDAR BEV", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# --- 위험 객체 리스트 ---
list_h = bev_size
list_w = bev_size
obj_list = np.zeros((list_h, list_w, 3), dtype=np.uint8) + 30
cv2.putText(obj_list, "Risk Assessment", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
cv2.line(obj_list, (10, 35), (list_w - 10, 35), (100, 100, 100), 1)

y_pos = 55
for i, obj in enumerate(detected[:12]):  # 최대 12개
    risk = obj.get('risk_level', 'SAFE')
    color = RISK_COLORS.get(risk, (200, 200, 200))

    # 위험도 아이콘
    cv2.circle(obj_list, (20, y_pos - 4), 6, color, -1)

    # 텍스트
    lane_pos = "CENTER" if obj.get('in_lane', False) else "SIDE"
    text = f"{obj['label']} | {risk} | {lane_pos} | d={obj['depth']:.2f}"
    cv2.putText(obj_list, text, (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    y_pos += 25

# --- 최종 합성 ---
panel_w = 640
panel_h = 400

main_r = cv2.resize(main_display, (panel_w, panel_h))
depth_r = cv2.resize(depth_colored, (panel_w, panel_h))
hud_r = cv2.resize(hud, (panel_w * 2, 60))
bev_r = cv2.resize(bev, (panel_w, panel_h))
list_r = cv2.resize(obj_list, (panel_w, panel_h))

top = np.hstack([main_r, depth_r])
bottom = np.hstack([bev_r, list_r])
main_view = np.vstack([hud_r, top, bottom])

cv2.imshow('Autonomous Driving Perception System', main_view)

# ============ 콘솔 출력 ============
print("\n" + "=" * 60)
print("  자율주행 인지 시스템 분석 결과")
print("=" * 60)
print(f"\n  [주행 판단] {driving_action}")
print(f"  [차선 제안] {lane_suggestion}")
print(f"\n  [차선별 위험]")
print(f"    좌측: {'위험' if left_risk > 0 else '안전'} ({left_risk}개)")
print(f"    중앙: {'위험' if center_risk > 0 else '안전'} ({center_risk}개)")
print(f"    우측: {'위험' if right_risk > 0 else '안전'} ({right_risk}개)")
print(f"\n  [LiDAR 정보]")
print(f"    전방 최소 거리: {min_lidar_dist:.1f}m")
print(f"    전방 평균 거리: {avg_lidar_dist:.1f}m")
print(f"\n  [감지 객체] 총 {len(detected)}개")
for obj in detected:
    risk = obj.get('risk_level', 'SAFE')
    lane = "차선내" if obj.get('in_lane', False) else "차선외"
    print(f"    [{risk:7s}] {obj['label']:15s} | {lane} | 깊이={obj['depth']:.2f} | 신뢰도={obj['conf']:.0%}")

print(f"\n  [위험 객체] {len(risk_objects)}개")
for obj in risk_objects:
    print(f"    ⚠ {obj['label']} - {obj['risk_level']} (깊이={obj['depth']:.2f})")

print("\n" + "=" * 60)
print("  아무 키나 누르면 종료 (OpenCV 창 클릭 후)")
print("=" * 60)

cv2.waitKey(0)
cv2.destroyAllWindows()
