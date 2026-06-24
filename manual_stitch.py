"""
manual_stitch.py — ручная сшивка двух фрагментов
Запуск: python manual_stitch.py --data D:\TASK1\data\prostate_2
"""
import os, sys, argparse, gc
from pathlib import Path

import cv2
import numpy as np
import tifffile
import openslide

THUMB_SIZE = 1500   # px по длинной стороне для отображения

def get_thumb(path, max_px=THUMB_SIZE):
    with openslide.OpenSlide(str(path)) as sl:
        w0, h0 = sl.level_dimensions[0]
        scale = min(max_px/w0, max_px/h0, 1.0)
        tw, th = max(1,int(w0*scale)), max(1,int(h0*scale))
        img = np.array(sl.get_thumbnail((tw, th)))[:,:,:3]
    return img, scale

def find_images(folder):
    p = Path(folder)
    imgs = []
    for sub in [p, p/"raw_images"]:
        if sub.exists():
            for ext in ("*.tif","*.tiff","*.mrxs","*.svs"):
                imgs.extend(sub.glob(ext))
    seen, result = set(), []
    for x in sorted(imgs):
        if x not in seen:
            seen.add(x); result.append(x)
    return result

def save_stitched(img1_path, img2_path, H_l0, out_dir, scale1, scale2):
    """Сохраняет превью и полный TIFF (тайлами)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Превью ---
    t1, sc1 = get_thumb(img1_path, 2000)
    t2, sc2 = get_thumb(img2_path, 2000)

    S1 = np.diag([sc1, sc1, 1.0])
    S2_inv = np.diag([1/sc2, 1/sc2, 1.0])
    H_thumb = S2_inv @ H_l0 @ S1

    h1,w1 = t1.shape[:2]
    h2,w2 = t2.shape[:2]

    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    c1p = cv2.perspectiveTransform(corners1, H_thumb).reshape(-1,2)
    all_pts = np.vstack([c1p, [[0,0],[w2,0],[w2,h2],[0,h2]]])
    xmin,ymin = int(all_pts[:,0].min()), int(all_pts[:,1].min())
    xmax,ymax = int(all_pts[:,0].max()), int(all_pts[:,1].max())

    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
    cw, ch = xmax-xmin, ymax-ymin

    warped1 = cv2.warpPerspective(t1, T @ H_thumb, (cw, ch))
    canvas  = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:h2, -xmin:-xmin+w2] = t2  # если xmin<=0

    # простой холст: сначала img2, поверх img1 там где есть
    canvas2 = np.zeros((ch, cw, 3), dtype=np.uint8)
    # img2 со сдвигом
    ox = -xmin; oy = -ymin
    x0=max(0,ox); y0=max(0,oy)
    x1=min(cw,ox+w2); y1=min(ch,oy+h2)
    sx0=max(0,-ox); sy0=max(0,-oy)
    canvas2[y0:y1, x0:x1] = t2[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]

    mask1 = (warped1.sum(axis=2)>0)
    canvas2[mask1] = warped1[mask1]

    # блендинг в зоне перекрытия
    mask2 = (canvas2.sum(axis=2)>0) & ~mask1
    both  = mask1 & (canvas2.sum(axis=2)>0)
    # просто 50/50 где оба
    final = canvas2.copy()
    final[both] = (warped1[both].astype(np.float32)*0.5 +
                   canvas2[both].astype(np.float32)*0.5).astype(np.uint8)

    preview_path = out_dir / "manual_preview.jpg"
    cv2.imwrite(str(preview_path),
                cv2.cvtColor(final, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n✓ Превью сохранено: {preview_path}")
    print(f"  Размер превью: {cw}×{ch} px")
    return str(preview_path)


def click_points_cv2(img, title, n=4):
    """Сбор точек через окно OpenCV (работает везде без TkAgg)."""
    pts = []
    display = img.copy()
    COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),
              (255,0,255),(0,255,255),(128,0,255),(255,128,0)]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = len(pts)
            pts.append((x, y))
            col = COLORS[idx % len(COLORS)]
            cv2.circle(display, (x,y), 8, col, -1)
            cv2.putText(display, str(idx+1), (x+10, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.imshow(title, display)
            print(f"  Точка {idx+1}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            # перерисовать
            display[:] = img[:]
            for i,(px,py) in enumerate(pts):
                col = COLORS[i % len(COLORS)]
                cv2.circle(display,(px,py),8,col,-1)
                cv2.putText(display,str(i+1),(px+10,py-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
            cv2.imshow(title, display)
            print(f"  Точка удалена, осталось: {len(pts)}")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(1400, img.shape[1]), min(900, img.shape[0]))
    cv2.imshow(title, display)
    cv2.setMouseCallback(title, mouse_cb)

    print(f"\n  ЛКМ = добавить точку | ПКМ = удалить последнюю | Enter/Space = готово")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32):   # Enter или Space
            break
        if key == 27:         # Esc — отмена
            pts.clear()
            break

    cv2.destroyAllWindows()
    return pts


def run(data_folder):
    paths = find_images(data_folder)
    if len(paths) < 2:
        print(f"Ошибка: нужно минимум 2 файла, найдено {len(paths)}")
        sys.exit(1)

    print(f"\nНайдено файлов: {len(paths)}")
    for i,p in enumerate(paths):
        print(f"  {i+1}. {p.name}")

    # Берём первые два (или все пары последовательно)
    img1_path, img2_path = paths[0], paths[1]

    print(f"\nЗагрузка thumbnail...")
    t1, sc1 = get_thumb(img1_path)
    t2, sc2 = get_thumb(img2_path)

    # Конвертируем в BGR для OpenCV
    t1_bgr = cv2.cvtColor(t1, cv2.COLOR_RGB2BGR)
    t2_bgr = cv2.cvtColor(t2, cv2.COLOR_RGB2BGR)

    print("\n" + "="*60)
    print("ШАГ 1: Кликните 4+ точки на ПЕРВОМ фрагменте")
    print("  Выбирай характерные структуры: края ткани,")
    print("  крупные железы, границы фрагмента")
    print("="*60)
    pts1 = click_points_cv2(t1_bgr, f"ФРАГМЕНТ 1: {img1_path.name}")

    if len(pts1) < 4:
        print("Недостаточно точек на фрагменте 1 (нужно минимум 4)")
        sys.exit(1)

    print("\n" + "="*60)
    print("ШАГ 2: Кликните ТЕ ЖЕ точки на ВТОРОМ фрагменте")
    print(f"  Порядок тот же! Вы отметили {len(pts1)} точек.")
    print("="*60)
    pts2 = click_points_cv2(t2_bgr, f"ФРАГМЕНТ 2: {img2_path.name}")

    if len(pts2) < 4:
        print("Недостаточно точек на фрагменте 2")
        sys.exit(1)

    if len(pts1) != len(pts2):
        n = min(len(pts1), len(pts2))
        print(f"Количество точек не совпадает! Берём первые {n}")
        pts1, pts2 = pts1[:n], pts2[:n]

    print(f"\nВычисляем гомографию по {len(pts1)} точкам...")

    # Пересчёт из thumbnail в level-0
    p1_l0 = np.float32(pts1) / sc1
    p2_l0 = np.float32(pts2) / sc2

    H, mask = cv2.findHomography(
        p1_l0.reshape(-1,1,2),
        p2_l0.reshape(-1,1,2),
        cv2.RANSAC, 5.0
    )

    if H is None:
        print("Не удалось вычислить гомографию!")
        sys.exit(1)

    inliers = int(mask.sum()) if mask is not None else len(pts1)
    print(f"✓ Гомография найдена! Инлайеров: {inliers}/{len(pts1)}")

    # Показываем наложение для проверки
    print("\nПроверка наложения...")
    h1,w1 = t1.shape[:2]; h2,w2 = t2.shape[:2]
    S1 = np.diag([sc1,sc1,1.0]); S2_inv = np.diag([1/sc2,1/sc2,1.0])
    H_thumb = S2_inv @ H @ S1

    warped = cv2.warpPerspective(t1_bgr, H_thumb, (w2,h2))
    blend  = cv2.addWeighted(t2_bgr, 0.5, warped, 0.5, 0)
    cv2.namedWindow("Наложение (проверка) — нажми любую клавишу", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Наложение (проверка) — нажми любую клавишу", 1200, 800)
    cv2.imshow("Наложение (проверка) — нажми любую клавишу", blend)
    print("  Посмотри на наложение. Если хорошо — нажми Enter.")
    print("  Если плохо — нажми Esc для повтора.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == 27:
        print("Повторите расстановку точек заново.")
        sys.exit(0)

    out_dir = Path(data_folder) / "output"
    print(f"\nСохранение результата в {out_dir}...")
    save_stitched(img1_path, img2_path, H, out_dir, sc1, sc2)
    print("\n✓ Готово!")


def main():
    ap = argparse.ArgumentParser(description="Ручная сшивка двух фрагментов WSI")
    ap.add_argument("--data", required=True, help="Папка с фрагментами")
    args = ap.parse_args()
    run(args.data)

if __name__ == "__main__":
    main()
