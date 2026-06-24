"""
stitch_wsi.py  — memory-efficient WSI fragment stitcher
========================================================
Стратегия:
  1. Грубая регистрация на thumbnail (~2000px по длинной стороне)  →  быстро, мало памяти
  2. Уточнение гомографии на патчах зоны перекрытия уровня 4/5    →  точно, мало памяти
  3. Запись холста тайлами через tifffile                           →  не держим весь результат в RAM

Зависимости:
  pip install openslide-python opencv-python-headless tifffile numpy scipy scikit-image matplotlib
"""

import os, gc, sys, json, argparse, logging, time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import openslide
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Опционально, но желательно ──────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы — можно переопределить через config.json
# ─────────────────────────────────────────────────────────────────────────────
THUMB_MAX_PX    = 2000      # px по длинной стороне для грубой регистрации
REFINE_PATCH    = 2048      # размер патча для уточнения (px на уровне refine_level)
REFINE_N        = 6         # сколько патчей пробовать при уточнении
TARGET_MB_LOAD  = 200       # лимит на один загружаемый уровень (МБ)
SIFT_NFEATURES  = 3000      # количество точек SIFT
BLEND_BAND_PX   = 256       # ширина полосы линейного блендинга (px холста)
TILE_WRITE      = 1024      # размер тайла при записи OME-TIFF
OUTPUT_LEVEL    = None      # уровень для финальной сшивки (None = авто)


# ═════════════════════════════════════════════════════════════════════════════
# Часть 1 — утилиты OpenSlide
# ═════════════════════════════════════════════════════════════════════════════

def get_mpp(slide: openslide.OpenSlide) -> float | None:
    for k in ("openslide.mpp-x", "aperio.MPP", "hamamatsu.XOffsetFromSlideCentre"):
        if k in slide.properties:
            try:
                return float(slide.properties[k])
            except ValueError:
                pass
    return None


def best_level_for_mb(slide: openslide.OpenSlide, target_mb: int = TARGET_MB_LOAD) -> int:
    """Возвращает наивысший уровень, умещающийся в target_mb."""
    for lvl in range(slide.level_count):
        w, h = slide.level_dimensions[lvl]
        mb = w * h * 3 / 1024**2
        if mb <= target_mb:
            return lvl
    return slide.level_count - 1


def read_level(slide: openslide.OpenSlide, level: int) -> np.ndarray:
    w, h = slide.level_dimensions[level]
    img = np.array(slide.read_region((0, 0), level, (w, h)))
    return img[:, :, :3]   # RGBA → RGB


def thumb(slide: openslide.OpenSlide, max_px: int = THUMB_MAX_PX) -> tuple[np.ndarray, float]:
    """Возвращает thumbnail и коэффициент масштаба (thumb_px / level0_px)."""
    w0, h0 = slide.level_dimensions[0]
    scale = min(max_px / w0, max_px / h0, 1.0)
    tw, th = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    img = np.array(slide.get_thumbnail((tw, th)))[:, :, :3]
    return img, scale


def read_region_level0(slide: openslide.OpenSlide,
                       x: int, y: int, w: int, h: int,
                       level: int) -> np.ndarray:
    """Читает регион на уровне level; координаты (x,y,w,h) — в пикселях level-0."""
    ds = slide.level_downsamples[level]
    lx, ly = int(x), int(y)   # read_region принимает level-0 координаты
    lw = max(1, int(w / ds))
    lh = max(1, int(h / ds))
    # OpenSlide: read_region((x,y)_level0, level, (w,h)_at_level)
    img = np.array(slide.read_region((lx, ly), level, (lw, lh)))
    return img[:, :, :3]


# ═════════════════════════════════════════════════════════════════════════════
# Часть 2 — тканевая маска
# ═════════════════════════════════════════════════════════════════════════════

def tissue_mask(img: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Бинарная маска ткани (белый фон → отбрасываем)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((kernel, kernel), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    return m


def tissue_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box ткани: (x, y, w, h) или None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


# ═════════════════════════════════════════════════════════════════════════════
# Часть 3 — SIFT-регистрация
# ═════════════════════════════════════════════════════════════════════════════

def _sift_match(img1: np.ndarray, img2: np.ndarray,
                mask1: np.ndarray | None = None,
                mask2: np.ndarray | None = None,
                n_features: int = SIFT_NFEATURES,
                ratio: float = 0.75
                ) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + BF + ratio test. Возвращает (pts1, pts2) в пикселях img1/img2."""
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.03, edgeThreshold=10)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    kp1, d1 = sift.detectAndCompute(g1, mask1)
    kp2, d2 = sift.detectAndCompute(g2, mask2)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def _find_homography(pts1: np.ndarray, pts2: np.ndarray,
                     reproj_thresh: float = 4.0
                     ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(pts1) < 8:
        return None, None
    H, mask = cv2.findHomography(pts1.reshape(-1, 1, 2),
                                  pts2.reshape(-1, 1, 2),
                                  cv2.RANSAC, reproj_thresh)
    return H, mask


def validate_H(H: np.ndarray,
               max_scale: float = 3.0,
               max_shear_deg: float = 30.0) -> bool:
    if H is None:
        return False
    det = np.linalg.det(H[:2, :2])
    if det <= 0:
        return False
    sx = np.hypot(H[0, 0], H[1, 0])
    sy = np.hypot(H[0, 1], H[1, 1])
    if sx > max_scale or sy > max_scale or sx < 1 / max_scale or sy < 1 / max_scale:
        return False
    angle = abs(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    if angle > max_shear_deg:
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Часть 4 — многоуровневая регистрация (грубо → точно)
# ═════════════════════════════════════════════════════════════════════════════

class SlideInfo:
    """Кэшируем метаданные; сам слайд держим открытым минимально."""
    def __init__(self, path: str):
        self.path = Path(path)
        with openslide.OpenSlide(str(path)) as sl:
            self.dims0    = sl.level_dimensions[0]   # (w, h) level-0
            self.n_levels = sl.level_count
            self.downsamples = list(sl.level_downsamples)
            self.mpp      = get_mpp(sl)
        log.info(f"  {self.path.name}: {self.dims0[0]}×{self.dims0[1]} px, "
                 f"{self.n_levels} уровней, mpp={self.mpp}")

    @property
    def w0(self): return self.dims0[0]
    @property
    def h0(self): return self.dims0[1]


def coarse_registration(info1: SlideInfo, info2: SlideInfo,
                        thumb_max: int = THUMB_MAX_PX
                        ) -> tuple[np.ndarray | None, float, float]:
    """
    Грубая регистрация по thumbnail.
    Возвращает (H_level0, scale1, scale2) где H описывает трансформацию
    из img1_level0 → img2_level0.
    """
    log.info("  Грубая регистрация (thumbnail)...")
    with openslide.OpenSlide(str(info1.path)) as sl1:
        t1, sc1 = thumb(sl1, thumb_max)
    with openslide.OpenSlide(str(info2.path)) as sl2:
        t2, sc2 = thumb(sl2, thumb_max)

    m1 = tissue_mask(t1)
    m2 = tissue_mask(t2)

    pts1, pts2 = _sift_match(t1, t2, m1, m2)
    H_thumb, mask_r = _find_homography(pts1, pts2)

    if H_thumb is None or not validate_H(H_thumb):
        log.warning("  Грубая регистрация не удалась — пробуем без маски тканей")
        pts1, pts2 = _sift_match(t1, t2)
        H_thumb, _ = _find_homography(pts1, pts2)
        if H_thumb is None or not validate_H(H_thumb):
            log.error("  Грубая регистрация полностью провалилась")
            return None, sc1, sc2

    # Пересчёт H в систему level-0: pts_level0 = pts_thumb / scale
    # H_level0 = S2 @ H_thumb @ inv(S1)
    S1 = np.diag([sc1, sc1, 1.0])
    S2 = np.diag([sc2, sc2, 1.0])
    H_level0 = np.linalg.inv(S2) @ H_thumb @ S1

    inliers = int(mask_r.sum()) if mask_r is not None else 0
    log.info(f"  Грубая H найдена, инлайеров: {inliers}/{len(pts1)}")
    return H_level0, sc1, sc2


def refine_registration(info1: SlideInfo, info2: SlideInfo,
                        H_coarse: np.ndarray,
                        n_patches: int = REFINE_N,
                        patch_px: int = REFINE_PATCH,
                        target_mb: int = TARGET_MB_LOAD
                        ) -> np.ndarray:
    """
    Уточнение гомографии патчами на зоне перекрытия.
    Работает в координатах level-0, читает патчи на оптимальном уровне.
    """
    log.info("  Уточнение регистрации по патчам...")

    with openslide.OpenSlide(str(info1.path)) as sl1:
        lvl1 = best_level_for_mb(sl1, target_mb)
        ds1  = sl1.level_downsamples[lvl1]

    with openslide.OpenSlide(str(info2.path)) as sl2:
        lvl2 = best_level_for_mb(sl2, target_mb)
        ds2  = sl2.level_downsamples[lvl2]

    w0_1, h0_1 = info1.w0, info1.h0
    w0_2, h0_2 = info2.w0, info2.h0

    # Зона перекрытия: трансформируем углы img1 в систему img2
    corners1_l0 = np.float32([[0, 0], [0, h0_1], [w0_1, h0_1], [w0_1, 0]]).reshape(-1, 1, 2)
    corners1_in2 = cv2.perspectiveTransform(corners1_l0, H_coarse).reshape(-1, 2)

    ox_min = max(0, int(corners1_in2[:, 0].min()))
    ox_max = min(w0_2, int(corners1_in2[:, 0].max()))
    oy_min = max(0, int(corners1_in2[:, 1].min()))
    oy_max = min(h0_2, int(corners1_in2[:, 1].max()))

    ow = ox_max - ox_min
    oh = oy_max - oy_min

    if ow < patch_px * ds2 or oh < patch_px * ds2:
        log.warning("  Зона перекрытия мала — пропускаем уточнение")
        return H_coarse

    H_inv = np.linalg.inv(H_coarse)   # img2 → img1

    all_pts1, all_pts2 = [], []

    # Сетка центров патчей в img2
    step_x = max(1, ow // (n_patches + 1))
    step_y = max(1, oh // (n_patches + 1))

    with openslide.OpenSlide(str(info1.path)) as sl1, \
         openslide.OpenSlide(str(info2.path)) as sl2:

        for ci in range(1, n_patches + 1):
            cx2 = ox_min + ci * step_x        # центр в img2 level-0
            cy2 = oy_min + (ci % n_patches + 1) * step_y

            half = int(patch_px * ds2 // 2)
            px2_l0 = cx2 - half
            py2_l0 = cy2 - half
            pw2_l0 = 2 * half
            ph2_l0 = 2 * half

            # Соответствующий центр в img1
            c2_pt = np.float32([[[cx2, cy2]]])
            c1_pt = cv2.perspectiveTransform(c2_pt, H_inv).reshape(2)
            cx1, cy1 = float(c1_pt[0]), float(c1_pt[1])
            px1_l0 = cx1 - half
            py1_l0 = cy1 - half

            # Проверяем, что регион внутри обоих слайдов
            if (px2_l0 < 0 or py2_l0 < 0 or
                    px2_l0 + pw2_l0 > w0_2 or py2_l0 + ph2_l0 > h0_2 or
                    px1_l0 < 0 or py1_l0 < 0 or
                    px1_l0 + pw2_l0 > w0_1 or py1_l0 + ph2_l0 > h0_1):
                continue

            try:
                p1 = read_region_level0(sl1, px1_l0, py1_l0, pw2_l0, ph2_l0, lvl1)
                p2 = read_region_level0(sl2, px2_l0, py2_l0, pw2_l0, ph2_l0, lvl2)
            except Exception as e:
                log.debug(f"    Патч {ci}: {e}")
                continue

            m1 = tissue_mask(p1, kernel=5)
            m2 = tissue_mask(p2, kernel=5)
            if cv2.countNonZero(m1) < patch_px**2 * 0.1 or \
               cv2.countNonZero(m2) < patch_px**2 * 0.1:
                continue

            pts1_loc, pts2_loc = _sift_match(p1, p2, m1, m2, n_features=1500)
            if len(pts1_loc) < 8:
                continue

            # Пересчёт локальных координат в level-0
            pts1_l0 = pts1_loc * ds1 + np.array([px1_l0, py1_l0])
            pts2_l0 = pts2_loc * ds2 + np.array([px2_l0, py2_l0])

            all_pts1.append(pts1_l0)
            all_pts2.append(pts2_l0)
            log.info(f"    Патч {ci}: {len(pts1_loc)} совпадений")

            del p1, p2
            gc.collect()

    if len(all_pts1) < 2:
        log.warning("  Мало патчей для уточнения — используем грубую H")
        return H_coarse

    pts1_all = np.vstack(all_pts1)
    pts2_all = np.vstack(all_pts2)

    H_ref, mask_r = _find_homography(pts1_all, pts2_all, reproj_thresh=3.0)
    if H_ref is None or not validate_H(H_ref):
        log.warning("  Уточнённая H невалидна — используем грубую H")
        return H_coarse

    inliers = int(mask_r.sum()) if mask_r is not None else 0
    log.info(f"  Уточнённая H: {inliers}/{len(pts1_all)} инлайеров")
    return H_ref


def compute_homography(info1: SlideInfo, info2: SlideInfo,
                       config: dict) -> np.ndarray | None:
    """Полный pipeline регистрации двух слайдов (level-0 координаты)."""
    H_coarse, _, _ = coarse_registration(
        info1, info2,
        thumb_max=config.get("thumb_max_px", THUMB_MAX_PX)
    )
    if H_coarse is None:
        return None

    H_final = refine_registration(
        info1, info2, H_coarse,
        n_patches=config.get("refine_patches", REFINE_N),
        patch_px=config.get("refine_patch_px", REFINE_PATCH),
        target_mb=config.get("target_mb", TARGET_MB_LOAD),
    )
    return H_final


# ═════════════════════════════════════════════════════════════════════════════
# Часть 5 — вычисление холста и метрик
# ═════════════════════════════════════════════════════════════════════════════

def compute_canvas(infos: list[SlideInfo],
                   H_list: list[np.ndarray | None],
                   out_level: int,
                   ds_out: float
                   ) -> tuple[int, int, list[np.ndarray]]:
    """
    Вычисляет размер холста в координатах level-0 и матрицы H:
      slide_i_level0 -> canvas_level0  (со сдвигом чтобы все coords >= 0)
    H_list[i] уже в level-0: slide_i_level0 -> slide_0_level0
    """
    all_corners = []

    for i, info in enumerate(infos):
        H_l0 = H_list[i] if H_list[i] is not None else np.eye(3)
        w0, h0 = info.w0, info.h0
        corners = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2)
        corners_in_ref = cv2.perspectiveTransform(corners, H_l0).reshape(-1,2)
        all_corners.append(corners_in_ref)

    all_pts = np.vstack(all_corners)
    x_min = int(np.floor(all_pts[:,0].min()))
    y_min = int(np.floor(all_pts[:,1].min()))
    x_max = int(np.ceil(all_pts[:,0].max()))
    y_max = int(np.ceil(all_pts[:,1].max()))

    # Сдвиг чтобы все координаты были >= 0
    T_shift = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], dtype=np.float64)

    # H_canvas[i]: slide_i_level0 -> canvas_level0
    H_canvas_list = []
    for i in range(len(infos)):
        H_l0 = H_list[i] if H_list[i] is not None else np.eye(3)
        H_canvas_list.append(T_shift @ H_l0)

    canvas_w_l0 = x_max - x_min
    canvas_h_l0 = y_max - y_min

    # Возвращаем размер в level-0 coords — save_preview сам пересчитает
    log.info(f"  Холст level-0: {canvas_w_l0}x{canvas_h_l0} px")
    return canvas_w_l0, canvas_h_l0, H_canvas_list


def calc_metrics(info1: SlideInfo, info2: SlideInfo,
                 H: np.ndarray, mpp: float | None) -> dict:
    """
    Метрики качества регистрации.
    НЕ использует SIFT (не работает на однородной ткани простаты).
    Использует:
      1. SSIM на зоне перекрытия — основная метрика
      2. NCC (normalized cross-correlation) — дополнительная
      3. Оценка сдвига через фазовую корреляцию — для RMSE
    """
    with openslide.OpenSlide(str(info1.path)) as sl1:
        t1, sc1 = thumb(sl1, THUMB_MAX_PX)
    with openslide.OpenSlide(str(info2.path)) as sl2:
        t2, sc2 = thumb(sl2, THUMB_MAX_PX)

    h1, w1 = t1.shape[:2]
    h2, w2 = t2.shape[:2]

    # H: level0_img1 -> level0_img2
    # H_thumb: thumb1 -> thumb2
    S1 = np.diag([sc1, sc1, 1.0])
    S2_inv = np.diag([1/sc2, 1/sc2, 1.0])
    H_thumb = S2_inv @ H @ S1

    m = {}

    # Варпим t1 в систему t2
    warped1 = cv2.warpPerspective(t1, H_thumb, (w2, h2),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,0,0))

    # Маска перекрытия: исключаем чёрный фон (0) И белый фон стекла (>240)
    def tissue_px(img):
        r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
        black = (r<15)&(g<15)&(b<15)
        white = (r>240)&(g>240)&(b>240)
        return ~black & ~white
    mask_w  = tissue_px(warped1)
    mask_t2 = tissue_px(t2)
    overlap_mask = mask_w & mask_t2

    overlap_px = int(overlap_mask.sum())
    m["overlap_px"] = overlap_px
    m["overlap_pct"] = round(float(overlap_px) / max(mask_t2.sum(), 1) * 100, 1)

    if overlap_px < 100:
        # Фрагменты стыкуются встык (без перекрытия) — это нормально!
        # Используем метрику качества стыка по границе
        log.info(f"  Перекрытие мало ({overlap_px} px) — фрагменты стыкуются встык")
        m["verdict"] = "ВСТЫК (нет перекрытия) — визуально проверь превью"
        m["ssim"] = None
        m["ncc"] = None
        return m

    # Вырезаем зону перекрытия
    ys, xs = np.where(overlap_mask)
    xm, xM = xs.min(), xs.max()
    ym, yM = ys.min(), ys.max()

    ov1 = cv2.resize(warped1[ym:yM, xm:xM], (256, 256)).astype(np.float32)
    ov2 = cv2.resize(t2[ym:yM, xm:xM],      (256, 256)).astype(np.float32)

    # 1. SSIM
    if HAS_SKIMAGE:
        m["ssim"] = float(ssim(
            ov1.astype(np.uint8), ov2.astype(np.uint8),
            channel_axis=2, data_range=255
        ))
    else:
        # Простой аналог через корреляцию
        m["ssim"] = None

    # 2. NCC (normalized cross-correlation) — работает на однородной ткани
    ov1_g = cv2.cvtColor(ov1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ov2_g = cv2.cvtColor(ov2.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ov1_g -= ov1_g.mean(); ov2_g -= ov2_g.mean()
    denom = (np.linalg.norm(ov1_g) * np.linalg.norm(ov2_g))
    m["ncc"] = float(np.sum(ov1_g * ov2_g) / denom) if denom > 0 else 0.0

    # 3. Оценка остаточного сдвига через phase correlation
    #    Если регистрация идеальна — сдвиг будет (0,0)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(
            cv2.cvtColor(ov1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32),
            cv2.cvtColor(ov2.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        )
        # dx, dy — сдвиг в пикселях 256x256 патча, пересчитываем в level-0
        patch_scale_w = (xM - xm) / 256.0  # px_thumb / px_patch
        patch_scale_h = (yM - ym) / 256.0
        rmse_thumb = float(np.sqrt((dx * patch_scale_w)**2 + (dy * patch_scale_h)**2))
        m["residual_shift_thumb_px"] = round(rmse_thumb, 2)
        m["rmse_thumb_px"] = rmse_thumb
        m["rmse_level0_px"] = round(rmse_thumb / sc1, 1)
        if mpp is not None:
            m["rmse_um"] = round(m["rmse_level0_px"] * mpp, 2)
        else:
            m["rmse_um"] = None
    except Exception as e:
        log.debug(f"  Phase correlate failed: {e}")
        m.update({"rmse_thumb_px": None, "rmse_level0_px": None, "rmse_um": None})

    # Вердикт — теперь опираемся на SSIM + NCC, а не только RMSE
    ssim_v = m.get("ssim") or 0.0
    ncc_v  = m.get("ncc")  or 0.0
    rmse_um = m.get("rmse_um")

    if rmse_um is not None:
        if   rmse_um < 5.0  and ssim_v > 0.6: m["verdict"] = "ОТЛИЧНО"
        elif rmse_um < 15.0 and ssim_v > 0.4: m["verdict"] = "ПРИЕМЛЕМО"
        elif rmse_um < 50.0:                   m["verdict"] = "УДОВЛЕТВОРИТЕЛЬНО"
        else:                                  m["verdict"] = "НЕУДОВЛЕТВОРИТЕЛЬНО"
    elif ssim_v > 0.6 and ncc_v > 0.6:        m["verdict"] = "ОТЛИЧНО (нет mpp)"
    elif ssim_v > 0.4 or  ncc_v > 0.4:        m["verdict"] = "ПРИЕМЛЕМО (нет mpp)"
    elif overlap_px > 1000:                    m["verdict"] = "СЛАБОЕ ПЕРЕКРЫТИЕ"
    else:                                      m["verdict"] = "НЕТ ДАННЫХ"

    log.info(f"  SSIM={ssim_v:.3f}  NCC={ncc_v:.3f}  overlap={m['overlap_pct']}%  verdict={m['verdict']}")
    return m


def _warp_tile(slide: openslide.OpenSlide,
               H_slide_to_canvas: np.ndarray,
               out_level: int, ds_out: float,
               tx: int, ty: int,
               tile_w: int, tile_h: int,
               canvas_w: int, canvas_h: int,
               ) -> np.ndarray:
    """
    Рендерит один тайл холста из одного слайда.
    Возвращает RGB-тайл (tile_h, tile_w, 3) uint8.
    """
    # Вычислить, откуда в слайде брать пиксели для этого тайла
    # H maps slide_canvas_coords → output_canvas_coords
    # Нам нужно обратное: output_canvas → slide_canvas
    H_inv = np.linalg.inv(H_slide_to_canvas)

    # Угловые координаты тайла в output-canvas
    corners_out = np.float32([
        [tx, ty], [tx + tile_w, ty],
        [tx + tile_w, ty + tile_h], [tx, ty + tile_h]
    ]).reshape(-1, 1, 2)

    corners_slide = cv2.perspectiveTransform(corners_out, H_inv).reshape(-1, 2)
    sx_min = int(np.floor(corners_slide[:, 0].min()))
    sy_min = int(np.floor(corners_slide[:, 1].min()))
    sx_max = int(np.ceil(corners_slide[:, 0].max()))
    sy_max = int(np.ceil(corners_slide[:, 1].max()))

    # Clip to slide bounds (out_level coords)
    w_lvl, h_lvl = slide.level_dimensions[out_level]
    sx_min = max(0, sx_min); sy_min = max(0, sy_min)
    sx_max = min(w_lvl, sx_max); sy_max = min(h_lvl, sy_max)

    if sx_max <= sx_min or sy_max <= sy_min:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    # Читаем нужный регион слайда
    l0x = int(sx_min * ds_out)
    l0y = int(sy_min * ds_out)
    rw  = sx_max - sx_min
    rh  = sy_max - sy_min
    region = np.array(slide.read_region((l0x, l0y), out_level, (rw, rh)))[:, :, :3]

    # Варпим регион на тайл холста
    # Строим H: от coords_in_region → coords_in_tile
    T_slide_origin = np.array([[1,0,-sx_min],[0,1,-sy_min],[0,0,1]], dtype=np.float64)
    T_tile_origin  = np.array([[1,0,-tx],[0,1,-ty],[0,0,1]], dtype=np.float64)
    H_tile = T_tile_origin @ H_slide_to_canvas @ np.linalg.inv(T_slide_origin)

    warped = cv2.warpPerspective(region, H_tile, (tile_w, tile_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    return warped


def write_stitched_tiff(infos: list[SlideInfo],
                        H_canvas_list: list[np.ndarray],
                        canvas_w: int, canvas_h: int,
                        out_level: int, ds_out: float,
                        out_path: Path,
                        tile_size: int = TILE_WRITE,
                        blend_band: int = BLEND_BAND_PX) -> None:
    """
    Записывает итоговый OME-TIFF тайлами, не держа всё изображение в RAM.
    Для каждого тайла читаем регионы из всех слайдов и блендируем.
    """
    log.info(f"  Запись {canvas_w}×{canvas_h} px → {out_path}")

    n_tiles_x = (canvas_w + tile_size - 1) // tile_size
    n_tiles_y = (canvas_h + tile_size - 1) // tile_size
    total = n_tiles_x * n_tiles_y

    slides = [openslide.OpenSlide(str(info.path)) for info in infos]
    try:
        with tifffile.TiffWriter(str(out_path), bigtiff=True) as tif:
            opts = dict(
                tile=(tile_size, tile_size),
                compression="jpeg",
                photometric="rgb",
                subfiletype=0,
                metadata=None,
            )
            # Пишем построчно тайлов
            # tifffile принимает полные полосы или весь массив;
            # будем писать весь холст полосами по tile_size строк
            done = 0
            for row in range(n_tiles_y):
                ty = row * tile_size
                th = min(tile_size, canvas_h - ty)
                # Накапливаем полосу
                strip = np.zeros((th, canvas_w, 3), dtype=np.float32)
                weight = np.zeros((th, canvas_w), dtype=np.float32)

                for i, (info, H_c) in enumerate(zip(infos, H_canvas_list)):
                    # Быстро вычислим: попадает ли слайд i вообще в эту полосу
                    corners_slide = np.float32([
                        [0, 0], [info.w0/ds_out, 0],
                        [info.w0/ds_out, info.h0/ds_out], [0, info.h0/ds_out]
                    ]).reshape(-1, 1, 2)
                    corners_canvas = cv2.perspectiveTransform(corners_slide, H_c).reshape(-1, 2)
                    cy_min = corners_canvas[:, 1].min()
                    cy_max = corners_canvas[:, 1].max()
                    if cy_min > ty + th or cy_max < ty:
                        continue

                    # Для каждого столбца тайлов в полосе
                    for col in range(n_tiles_x):
                        tx = col * tile_size
                        tw = min(tile_size, canvas_w - tx)

                        cx_min = corners_canvas[:, 0].min()
                        cx_max = corners_canvas[:, 0].max()
                        if cx_min > tx + tw or cx_max < tx:
                            continue

                        piece = _warp_tile(slides[i], H_c, out_level, ds_out,
                                           tx, ty, tw, th, canvas_w, canvas_h)
                        # Маска ткани
                        m = (piece.sum(axis=2) > 0).astype(np.float32)

                        # Дистанционный вес (расстояние до ближайшей границы тайла)
                        if blend_band > 0:
                            dist = np.ones((th, tw), dtype=np.float32)
                            dist[:, :blend_band]  = np.minimum(
                                dist[:, :blend_band],
                                np.linspace(0, 1, min(blend_band, tw))[None, :tw])
                            dist[:, -blend_band:] = np.minimum(
                                dist[:, -blend_band:],
                                np.linspace(1, 0, min(blend_band, tw))[None, :tw])
                            w_tile = m * dist
                        else:
                            w_tile = m

                        strip[0:th, tx:tx+tw] += piece.astype(np.float32) * w_tile[:, :, None]
                        weight[0:th, tx:tx+tw] += w_tile

                # Нормализуем
                nz = weight > 0
                strip[nz] /= weight[nz, None]
                strip_u8 = np.clip(strip, 0, 255).astype(np.uint8)

                tif.write(strip_u8, **opts)

                done += n_tiles_x
                log.info(f"    {done}/{total} тайлов ({100*done//total}%)")

    finally:
        for sl in slides:
            sl.close()

    log.info(f"  Сохранено: {out_path}")


def manual_registration(info1: SlideInfo, info2: SlideInfo) -> np.ndarray | None:
    """
    Интерактивная расстановка точек на thumbnail.
    Возвращает H в координатах level-0.
    """
    import matplotlib
    matplotlib.use("TkAgg" if "TkAgg" in matplotlib.rcsetup.all_backends else "Qt5Agg")
    import matplotlib.pyplot as plt

    with openslide.OpenSlide(str(info1.path)) as sl1:
        t1, sc1 = thumb(sl1, 1500)
    with openslide.OpenSlide(str(info2.path)) as sl2:
        t2, sc2 = thumb(sl2, 1500)

    pts1, pts2 = [], []

    def collect(img, pts, title):
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(img); ax.set_title(title)

        def onclick(e):
            if e.inaxes and e.button == 1:
                pts.append((e.xdata, e.ydata))
                ax.plot(e.xdata, e.ydata, "ro", ms=8)
                ax.text(e.xdata+5, e.ydata-5, str(len(pts)), color="w", fontsize=11)
                fig.canvas.draw()
        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.tight_layout(); plt.show()

    print("\n РУЧНАЯ РЕГИСТРАЦИЯ")
    print("1) Кликните 4+ характерных точки на ПЕРВОМ слайде, затем закройте окно")
    collect(t1, pts1, f"Слайд 1 ({info1.path.name}) — кликните 4+ точки, закройте окно")

    print("2) Кликните ТЕ ЖЕ точки на ВТОРОМ слайде, затем закройте окно")
    collect(t2, pts2, f"Слайд 2 ({info2.path.name}) — кликните те же точки, закройте окно")

    if len(pts1) < 4 or len(pts1) != len(pts2):
        print("Недостаточно точек!")
        return None

    p1 = np.float32(pts1).reshape(-1, 1, 2)
    p2 = np.float32(pts2).reshape(-1, 1, 2)
    H_thumb, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if H_thumb is None:
        return None

    S1 = np.diag([sc1, sc1, 1.0])
    S2_inv = np.diag([1 / sc2, 1 / sc2, 1.0])
    return S2_inv @ H_thumb @ S1


# ═════════════════════════════════════════════════════════════════════════════
# Часть 8 — визуальный отчёт
# ═════════════════════════════════════════════════════════════════════════════

def save_report(infos: list[SlideInfo],
                H_list: list[np.ndarray | None],
                metrics_list: list[dict],
                mpp: float | None,
                canvas_w: int, canvas_h: int,
                out_dir: Path) -> None:
    """Сохраняет PNG-отчёт и JSON с метриками."""
    n = len(infos)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (info, m) in enumerate(zip(infos, metrics_list)):
        with openslide.OpenSlide(str(info.path)) as sl:
            t, _ = thumb(sl, 600)
        axes[i, 0].imshow(t)
        axes[i, 0].set_title(f"Слайд {i+1}: {info.path.name}", fontsize=9)
        axes[i, 0].axis("off")

        # SSIM (если есть)
        txt = "\n".join(f"{k}: {v}" for k, v in m.items())
        axes[i, 1].text(0.05, 0.95, txt, transform=axes[i, 1].transAxes,
                        va="top", fontsize=8, family="monospace")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Метрики", fontsize=9)

        if H_list[i] is not None and i > 0:
            ref_info = infos[0]
            with openslide.OpenSlide(str(ref_info.path)) as sl0, \
                 openslide.OpenSlide(str(info.path)) as sli:
                t0, sc0 = thumb(sl0, 600)
                ti, sci = thumb(sli, 600)
            S_i = np.diag([sci, sci, 1.0])
            S0_inv = np.diag([1 / sc0, 1 / sc0, 1.0])
            H_t = S0_inv @ H_list[i] @ S_i
            warped = cv2.warpPerspective(ti, H_t, (t0.shape[1], t0.shape[0]))
            blend = cv2.addWeighted(t0, 0.5, warped, 0.5, 0)
            axes[i, 2].imshow(blend)
            axes[i, 2].set_title("Наложение (50/50)", fontsize=9)
        else:
            axes[i, 2].axis("off")

    plt.tight_layout()
    report_img = out_dir / "registration_report.png"
    plt.savefig(report_img, dpi=120, bbox_inches="tight")
    plt.close()

    report_json = out_dir / "metrics.json"
    with open(report_json, "w", encoding="utf-8") as f:
        # Сериализуем None и float
        def serialize(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v
        data = [
            {"slide": info.path.name, "metrics": {k: serialize(v) for k, v in m.items()}}
            for info, m in zip(infos, metrics_list)
        ]
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"  Отчёт: {report_img}")
    log.info(f"  Метрики: {report_json}")


# ═════════════════════════════════════════════════════════════════════════════
# Часть 9 — превью (быстро, без записи тяжёлого TIFF)
# ═════════════════════════════════════════════════════════════════════════════

def save_preview(infos: list[SlideInfo],
                 H_canvas_list: list[np.ndarray],
                 canvas_w: int, canvas_h: int,
                 out_path: Path,
                 max_px: int = 3000) -> None:
    """
    Рендерит превью.
    H_canvas_list[i]: slide_i_level0_coords -> canvas_level0_coords
    Алгоритм:
      1. Считаем scale превью = max_px / max(canvas_w, canvas_h)
      2. Для каждого слайда читаем thumbnail
      3. Строим H_final: thumbnail_coords -> preview_coords
         H_final = S_canvas @ H_canvas @ S_slide_inv
         где S_canvas = diag(scale_prev), S_slide_inv = diag(1/sc_slide)
      4. Варпим и блендируем
    """
    scale_prev = min(max_px / max(canvas_w, 1), max_px / max(canvas_h, 1), 1.0)
    pw = max(1, int(canvas_w * scale_prev))
    ph = max(1, int(canvas_h * scale_prev))
    log.info(f"  Превью: {pw}x{ph} px (холст {canvas_w}x{canvas_h} level-0, scale={scale_prev:.5f})")

    canvas = np.zeros((ph, pw, 3), dtype=np.float32)
    weight = np.zeros((ph, pw),    dtype=np.float32)

    for idx, (info, H_c) in enumerate(zip(infos, H_canvas_list)):
        # Читаем thumbnail слайда
        with openslide.OpenSlide(str(info.path)) as sl:
            w0, h0 = sl.level_dimensions[0]
            sc = min(1500 / w0, 1500 / h0, 1.0)
            tw = max(1, int(w0 * sc))
            th = max(1, int(h0 * sc))
            t = np.array(sl.get_thumbnail((tw, th)))[:, :, :3]

        # H_c: level0_slide -> level0_canvas (большие числа)
        # Нам нужно: thumb_slide -> preview_canvas
        #
        # thumb_slide -> level0_slide : умножить координаты на (1/sc)
        #   [x_l0, y_l0] = [x_th/sc, y_th/sc]  =>  S_inv = [[1/sc,0,0],[0,1/sc,0],[0,0,1]]
        #
        # level0_canvas -> preview_canvas : умножить на scale_prev
        #   S_prev = [[scale_prev,0,0],[0,scale_prev,0],[0,0,1]]
        #
        # H_final = S_prev @ H_c @ S_inv
        S_inv  = np.array([[1/sc, 0,    0],
                            [0,    1/sc, 0],
                            [0,    0,    1]], dtype=np.float64)
        S_prev = np.array([[scale_prev, 0,          0],
                            [0,          scale_prev, 0],
                            [0,          0,          1]], dtype=np.float64)
        H_final = S_prev @ H_c @ S_inv

        log.info(f"  Слайд {idx+1}: thumb={tw}x{th} sc={sc:.5f} "
                 f"tx={H_final[0,2]:.1f} ty={H_final[1,2]:.1f}")

        warped = cv2.warpPerspective(
            t.astype(np.float32), H_final, (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Маска непустых пикселей (ткань светлее чёрного фона после варпа)
        # Маска ткани: исключаем чёрный фон (после варпа = 0) И белый фон стекла (>240)
        w_r = warped[:,:,0]; w_g = warped[:,:,1]; w_b = warped[:,:,2]
        is_black = (w_r < 15) & (w_g < 15) & (w_b < 15)   # чёрный фон после варпа
        is_white = (w_r > 240) & (w_g > 240) & (w_b > 240) # белый фон стекла
        m = (~is_black & ~is_white).astype(np.float32)
        log.info(f"  Слайд {idx+1}: непустых пикселей = {int(m.sum())}")

        canvas += warped * m[:, :, None]
        weight += m

    nz = weight > 0
    log.info(f"  Итого непустых пикселей: {int(nz.sum())}")
    canvas[nz] /= weight[nz, None]
    result = np.clip(canvas, 0, 255).astype(np.uint8)

    if result.sum() == 0:
        log.warning("  Превью пустое! Гомография выводит слайды за пределы холста.")

    cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"  Превью сохранено: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Часть 10 — главный pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _manual_cv2(info1: SlideInfo, info2: SlideInfo,
                thumb_px: int = 1500) -> np.ndarray | None:
    """
    Ручная регистрация через окна OpenCV.
    ЛКМ = добавить точку, ПКМ = удалить последнюю, Enter/Пробел = готово, Esc = отмена.
    Возвращает H в координатах level-0.
    """
    def _get_thumb(info, max_px):
        with openslide.OpenSlide(str(info.path)) as sl:
            w0, h0 = sl.level_dimensions[0]
            sc = min(max_px / w0, max_px / h0, 1.0)
            tw, th = max(1, int(w0 * sc)), max(1, int(h0 * sc))
            img = np.array(sl.get_thumbnail((tw, th)))[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), sc

    COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),
              (255,0,255),(255,255,0),(128,0,255),(0,128,255)]

    def _collect_points(bgr_img, win_title):
        pts = []
        display = bgr_img.copy()

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = len(pts)
                pts.append((x, y))
                col = COLORS[idx % len(COLORS)]
                cv2.circle(display, (x, y), 9, col, -1)
                cv2.putText(display, str(idx + 1), (x + 12, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                cv2.imshow(win_title, display)
                print(f"    Точка {idx+1}: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN and pts:
                pts.pop()
                display[:] = bgr_img[:]
                for j, (px, py) in enumerate(pts):
                    col = COLORS[j % len(COLORS)]
                    cv2.circle(display, (px, py), 9, col, -1)
                    cv2.putText(display, str(j+1), (px+12, py-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                cv2.imshow(win_title, display)
                print(f"    Точка удалена. Осталось: {len(pts)}")

        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, min(1400, bgr_img.shape[1]),
                                    min(900,  bgr_img.shape[0]))
        cv2.imshow(win_title, display)
        cv2.setMouseCallback(win_title, mouse_cb)

        print(f"  ЛКМ=добавить  ПКМ=удалить последнюю  Enter/Пробел=готово  Esc=отмена")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 32):   # Enter или Пробел
                break
            if key == 27:         # Esc
                pts.clear()
                break
        cv2.destroyWindow(win_title)
        return pts


    print(f"  ШАГ 1: Расставьте точки на ПЕРВОМ фрагменте")
    print(f"  Файл: {info1.path.name}")
    print(f"  Кликайте на характерные структуры: края ткани, крупные железы")
    print(f"  Минимум 4 точки!")


    t1_bgr, sc1 = _get_thumb(info1, thumb_px)
    pts1 = _collect_points(t1_bgr, f"1: {info1.path.name} — кликайте точки")

    if len(pts1) < 4:
        print("  Недостаточно точек на фрагменте 1 (нужно ≥ 4)")
        return None

    print(f"  ШАГ 2: Расставьте ТЕ ЖЕ точки на ВТОРОМ фрагменте")
    print(f"  Файл: {info2.path.name}")
    print(f"  Порядок ВАЖЕН — те же точки в том же порядке!")
    print(f"  Вы отметили {len(pts1)} точек — отметьте столько же.")

    t2_bgr, sc2 = _get_thumb(info2, thumb_px)
    pts2 = _collect_points(t2_bgr, f"2: {info2.path.name} — те же точки, тот же порядок")

    if len(pts2) < 4:
        print("  Недостаточно точек на фрагменте 2 (нужно ≥ 4)")
        return None

    n = min(len(pts1), len(pts2))
    if len(pts1) != len(pts2):
        print(f"  Количество точек не совпадает! Берём первые {n}")
    pts1, pts2 = pts1[:n], pts2[:n]

    # Пересчёт thumbnail → level-0
    p1_l0 = np.float32(pts1) / sc1
    p2_l0 = np.float32(pts2) / sc2

    # Сначала пробуем частичное аффинное (сдвиг+поворот+масштаб) — стабильнее при малом n
    H_aff, mask_aff = cv2.estimateAffinePartial2D(
        p1_l0.reshape(-1, 1, 2),
        p2_l0.reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=50.0
    )

    if H_aff is not None:
        det = np.linalg.det(H_aff[:, :2])
        print(f"  Аффинная матрица det={det:.3f}")
        if det > 0.1:  # нормальная, не вырожденная
            # Дополняем до 3x3
            H = np.vstack([H_aff, [0, 0, 1]]).astype(np.float64)
            inliers = int(mask_aff.sum()) if mask_aff is not None else n
            print(f"\n  ✓ Аффинная трансформация найдена! Инлайеров: {inliers}/{n}")
        else:
            print("  Аффинная вырожденная, пробуем полную гомографию...")
            H_aff = None

    if H_aff is None:
        H, mask = cv2.findHomography(
            p1_l0.reshape(-1, 1, 2),
            p2_l0.reshape(-1, 1, 2),
            cv2.RANSAC, 50.0
        )
        if H is None:
            print("  Не удалось вычислить трансформацию!")
            return None
        det = np.linalg.det(H[:2,:2])
        if det < 0:
            print(f"  ВНИМАНИЕ: det={det:.3f} < 0 — возможно точки расставлены в разном порядке!")
            print("  Попробуй расставить точки в одном направлении на обоих фрагментах")
        inliers = int(mask.sum()) if mask is not None else n
        print(f"\n  ✓ Гомография найдена! Инлайеров: {inliers}/{n}")

    # Показываем наложение для проверки
    S1 = np.diag([sc1, sc1, 1.0])
    S2_inv = np.diag([1 / sc2, 1 / sc2, 1.0])
    H_thumb = S2_inv @ H @ S1
    warped = cv2.warpPerspective(t1_bgr, H_thumb, (t2_bgr.shape[1], t2_bgr.shape[0]))
    blend = cv2.addWeighted(t2_bgr, 0.5, warped, 0.5, 0)

    win_check = "ПРОВЕРКА наложения — Enter=принять  Esc=повторить"
    cv2.namedWindow(win_check, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_check, min(1400, blend.shape[1]), min(900, blend.shape[0]))
    cv2.imshow(win_check, blend)
    print(f"\n  Смотрите на наложение в окне!")
    print(f"  Enter = всё хорошо, продолжить")
    print(f"  Esc   = плохо, нужно повторить (запустите скрипт заново)")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == 27:
        print("  Регистрация отклонена. Запустите скрипт заново и расставь точки точнее.")
        return None

    return H


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning(f"config не найден ({path}), используем defaults")
        return {}


def load_images(folder: str) -> list[Path]:
    p = Path(folder)
    imgs = []
    for ext in ("*.mrxs", "*.tif", "*.tiff", "*.svs", "*.ndpi", "*.scn"):
        imgs.extend(p.glob(ext))
        imgs.extend((p / "raw_images").glob(ext) if (p / "raw_images").exists() else [])
    seen = set()
    result = []
    for x in sorted(imgs):
        if x not in seen:
            seen.add(x); result.append(x)
    return result


def run(data_folder: str, config_path: str,
        output_dir: str | None = None,
        manual: bool = False) -> str | None:
    t0 = time.time()
    config = load_config(config_path)

    paths = load_images(data_folder)
    if len(paths) < 2:
        raise ValueError(f"Нужно ≥ 2 файла в {data_folder}, найдено: {len(paths)}")
    log.info(f"Найдено слайдов: {len(paths)}")

    infos = [SlideInfo(str(p)) for p in paths]

    # Определяем выходной уровень
    out_level_cfg = config.get("output_level", OUTPUT_LEVEL)
    target_mb = config.get("target_mb", TARGET_MB_LOAD)

    with openslide.OpenSlide(str(infos[0].path)) as sl:
        if out_level_cfg is None:
            out_level = best_level_for_mb(sl, target_mb)
        else:
            out_level = int(out_level_cfg)
        ds_out = sl.level_downsamples[min(out_level, sl.level_count - 1)]

    log.info(f"Выходной уровень: {out_level}, downsample: {ds_out:.1f}x")

    # === Регистрация ===
    # H_list[i]: level-0 coords img_i → level-0 coords img_0
    H_list: list[np.ndarray | None] = [np.eye(3)]   # 0-й слайд — референс

    for i in range(1, len(infos)):
        log.info(f"\n── Регистрация слайда {i+1}/{len(infos)}: {infos[i].path.name}")

        H_i = None

        if manual:
            # Принудительно ручной режим
            H_i = _manual_cv2(infos[i - 1], infos[i])
        else:
            # Сначала пробуем автоматику
            H_i = compute_homography(infos[i - 1], infos[i], config)

            if H_i is None:
                # Автоматика провалилась — предлагаем ручную
                log.warning(f"  Автоматическая регистрация не удалась!")
                log.warning(f"  Открываю окно ручной расстановки точек...")
                print("\n" + "="*60)
                print(f"  АВТОМАТИКА НЕ СПРАВИЛАСЬ со слайдом {i+1}")
                print(f"  Сейчас откроются два окна — расставьте точки вручную")
                print("="*60)
                H_i = _manual_cv2(infos[i - 1], infos[i])

        if H_i is None:
            log.error(f"  Регистрация слайда {i+1} не удалась даже вручную — пропускаем")
            H_list.append(None)
            continue

        # Цепочка: img_i в систему img_0
        H_chain = H_list[i - 1] @ H_i if H_list[i - 1] is not None else H_i
        H_list.append(H_chain)

    # Метрики
    mpp = infos[0].mpp
    metrics_list: list[dict] = [{}]
    for i in range(1, len(infos)):
        if H_list[i] is not None:
            m = calc_metrics(infos[0], infos[i], H_list[i], mpp)
            log.info(f"  Метрики {infos[i].path.name}: {m.get('verdict','?')} "
                     f"| RMSE={m.get('rmse_um','?')} мкм "
                     f"| confidence={m.get('confidence',0):.1%}")
        else:
            m = {"verdict": "НЕ ЗАРЕГИСТРИРОВАН"}
        metrics_list.append(m)

    # Выходная папка
    case_name = Path(data_folder).name
    if output_dir is None:
        out_dir = Path(data_folder) / "output"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Холст
    valid_H = [(info, H) for info, H in zip(infos, H_list) if H is not None]
    if not valid_H:
        log.error("Ни один слайд не зарегистрирован!")
        return None

    canvas_w, canvas_h, H_canvas_list_valid = compute_canvas(
        [x[0] for x in valid_H],
        [x[1] for x in valid_H],
        out_level, ds_out
    )
    # canvas_w/h — в level-0 координатах (для save_preview)
    # для записи TIFF нужен размер в out_level coords
    canvas_w_tiff = max(1, int(canvas_w / ds_out))
    canvas_h_tiff = max(1, int(canvas_h / ds_out))
    log.info(f"\nХолст level-0: {canvas_w}×{canvas_h} px")
    log.info(f"Холст level-{out_level}: {canvas_w_tiff}×{canvas_h_tiff} px")

    # Отчёт
    save_report(infos, H_list, metrics_list, mpp, canvas_w, canvas_h, out_dir)

    # Превью (быстро)
    preview_path = out_dir / f"{case_name}_preview.jpg"
    save_preview([x[0] for x in valid_H], H_canvas_list_valid,
                 canvas_w, canvas_h, preview_path)

    # Запись полного TIFF
    if config.get("write_full_tiff", True):
        out_tiff = out_dir / f"{case_name}_stitched.tiff"
        write_stitched_tiff(
            [x[0] for x in valid_H], H_canvas_list_valid,
            canvas_w_tiff, canvas_h_tiff,
            out_level, ds_out,
            out_tiff,
            tile_size=config.get("tile_size", TILE_WRITE),
            blend_band=config.get("blend_band_px", BLEND_BAND_PX),
        )
    else:
        out_tiff = None
        log.info("write_full_tiff=false — полный TIFF не записывается")

    elapsed = time.time() - t0
    log.info(f"\nГотово за {elapsed:.1f} сек")
    return str(out_tiff) if out_tiff else str(preview_path)


def main():
    ap = argparse.ArgumentParser(
        description="Memory-efficient WSI fragment stitcher (MRXS/TIF/TIFF/SVS)")
    ap.add_argument("--data",   required=True, help="Папка со слайдами (или raw_images/)")
    ap.add_argument("--config", default="config.json", help="JSON-конфиг (опционально)")
    ap.add_argument("--output", default=None, help="Выходная папка (по умолчанию data/output)")
    ap.add_argument("--manual", action="store_true",
                    help="Ручная расстановка контрольных точек (GUI)")
    ap.add_argument("--no-tiff", action="store_true",
                    help="Не записывать полный TIFF (только превью и отчёт)")
    ap.add_argument("--level",  type=int, default=None,
                    help="Принудительный выходной уровень (0=макс разрешение)")
    args = ap.parse_args()

    cfg_path = args.config
    # Патчим конфиг из CLI-флагов
    try:
        cfg = load_config(cfg_path)
    except Exception:
        cfg = {}
    if args.no_tiff:
        cfg["write_full_tiff"] = False
    if args.level is not None:
        cfg["output_level"] = args.level
    # Временно сохраним патченый конфиг
    tmp_cfg = Path("_stitch_cfg.json")
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f)

    try:
        run(args.data, str(tmp_cfg), args.output, args.manual)
    except Exception as e:
        log.error(f"Ошибка: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
