"""
===============================================================================
Lunar Base Construction Cost Model
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import pandas as pd
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

plt.rcParams.update({

    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],


    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,


    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'grid.linewidth': 0.3,
    'lines.linewidth': 0.8,


    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
})


ALPHA     = 0.2
G_GRAVITY = 9.80665
I_SP=450

def compute_G_total(DV, I_SP):

    G_total = 1.0
    for dv in DV:
        G_total *= np.exp(dv / (G_GRAVITY * I_SP))
    return G_total

def compute_unit_cost(s, c_launch, V):

    V_pressurized = s['V_pressurized']
    m_structure = s['m_structure']
    m = V * m_structure / V_pressurized * 1e3

    C_dev = s['C_development']
    C_prod = s['P_production'] * m / 1e6
    C_con = s['P_construction'] * m / 1e6

    DV = s['DV']
    G_total = compute_G_total(DV, I_SP)
    m_trans_1 = G_total * (s['m_prefabrication'] + s['m_material'])
    m_trans_2 = G_total * ALPHA * s['m_equipment']
    C_trans = (m_trans_1 * c_launch / m_structure * m + m_trans_2 * c_launch) / 1e6

    return (C_dev + C_prod + C_con + C_trans) / V

def compute_earth_dependence(s, V):

    fixed   = ALPHA * s['m_equipment'] / V
    scaling = (s['m_prefabrication'] + s['m_material']) * 1e3 / s['V_pressurized']
    return fixed + scaling


def earth_dependence_floor(s):

    return (s['m_prefabrication'] + s['m_material']) * 1e3 / s['V_pressurized']

SCHEMES = {

    'Xuanwu': {
        'V_pressurized'      : 170,
        'm_structure'        : 706,
        'C_development'      : 84.48252,
        'P_production'       : 0,
        'm_prefabrication'   : 0,
        'm_material'         : 0,
        'm_equipment'        : 4100,
        'P_construction'     : 9.75628,
        'DV'                 : [7384, 3150, 1000, 2100]
    },

    'Lunar Ampullae': {
        'V_pressurized'      : 43.55361,
        'm_structure'        : 45.90737,
        'C_development'      : 40.44657,
        'P_production'       : 1.55362,
        'm_prefabrication'   : 0,
        'm_material'         : 22.95369,
        'm_equipment'        : 1150,
        'P_construction'     : 21.01730,
        'DV'                 : [7384, 3150, 1000, 2100]
    },

    'Lunar Safe Haven': {
        'V_pressurized'      : 1325.35940,
        'm_structure'        : 1683.23479,
        'C_development'      : 74.00834,
        'P_production'       : 4.45571,
        'm_prefabrication'   : 15,
        'm_material'         : 0,
        'm_equipment'        : 3531,
        'P_construction'     : 0.30263,
        'DV'                 : [7384, 3150, 1000, 2100]
    },

    'Starship HLS': {
        'V_pressurized'      : 2370,
        'm_structure'        : 8687.15884,
        'C_development'      : 74.56304,
        'P_production'       : 0.86334,
        'm_prefabrication'   : 15,
        'm_material'         : 0,
        'm_equipment'        : 2400,
        'P_construction'     : 2.58494,
        'DV'                 : [7384, 3150, 1000, 2100]
    },

    'Moon Village': {
        'V_pressurized'      : 698.29000,
        'm_structure'        : 30.23900,
        'C_development'      : 1197.94314,
        'P_production'       : 3961.58318,
        'm_prefabrication'   : 30.23900,
        'm_material'         : 0,
        'm_equipment'        : 10000,
        'P_construction'     : 0.99580,
        'DV'                 : [7384, 3150, 1000, 2100]
    }
}

SCHEME_NAMES  = list(SCHEMES.keys())
SCHEME_COLORS = ['#EDDC42', '#94de84', '#52b5a5', '#638cad', '#734a7b']


# =============================================================================
# SANITY CHECK
# =============================================================================

print("\n========== Sanity Check at c_launch = $2700/kg ==========")
print(f"{'Scheme':<20} {'V_pressurized':>8} {'C_total ($M/m³)':>18}")
for name, s in SCHEMES.items():
    if s['V_pressurized'] > 1.0:
        c = compute_unit_cost(s, 2700, s['V_pressurized'])
        print(f"{name:<20} {s['V_pressurized']:>8.1f} {c:>18.4f}")

print("\n===================== Draw Figures ======================")

# =============================================================================
# FIGURE 2 2D heatmap
# =============================================================================



c_grid_h = np.logspace(0, 5, 200)
v_grid_h = np.logspace(1, 5, 200)
C_mesh, V_mesh = np.meshgrid(c_grid_h, v_grid_h)

cost_panels = {name: compute_unit_cost(SCHEMES[name], C_mesh, V_mesh)
               for name in SCHEME_NAMES}

vmin, vmax = 0.01, 1000

fig, axes = plt.subplots(1, 5, figsize=(7, 1.5), sharey=True)

for i, (ax, name) in enumerate(zip(axes, SCHEME_NAMES)):
    cost = np.clip(cost_panels[name], vmin, vmax)

    pcm = ax.pcolormesh(
        C_mesh, V_mesh, cost,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap='viridis',
        shading='auto',
        rasterized=True,
    )

    cs = ax.contour(C_mesh, V_mesh, cost,
                    levels=[0.1, 1, 10, 100],
                    colors='white',alpha=0.3, linewidths=0.5)
    manual = []
    for segs in cs.allsegs:
        segs = [s for s in segs if len(s) > 0]
        if not segs:
            continue
        seg = max(segs, key=len)
        manual.append(tuple(seg[len(seg) // 2]))
    labels = ax.clabel(cs, inline=False, fmt='%g', fontsize=6, manual=manual)
    for txt in labels:
        level = float(txt.get_text().split()[0])
        bg_rgba = pcm.cmap(pcm.norm(level))
        txt.set_bbox(dict(facecolor=bg_rgba, edgecolor='none',
                          alpha=1.0, pad=0.6))
        txt.set_color('white')
        txt.set_rotation(0)
        txt.set_alpha(0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$C_{\mathrm{Launch}}$ (\$/kg)')

    ax.set_xticks([10.0**e for e in range(0, 6)])
    ax.set_xlim(1e0, 1e5)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: rf'$10^{{{int(round(np.log10(v)))}}}$'
        if int(round(np.log10(v))) % 2 == 0 else ''))

    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))

    ax.set_title(name, pad=4, color='black',fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_zorder(10)

axes[0].set_ylabel(r'Pressurized volume $V$ (m³)')

fig.subplots_adjust(left=0.06, right=0.93, bottom=0.22, top=0.90, wspace=0.15)
cbar_ax = fig.add_axes([0.94, 0.22, 0.012, 0.68])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label(r'Unit cost (\$M/m³)')

plt.savefig('./outputs_1006/fig2_2D_heatmaps.tif', dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
# plt.show()
print("\nFigure 2 Done")
#
# =============================================================================
# FIGURE 3 cost curves vs V & cost composition heatmap
# =============================================================================

def gradient_fill_log(ax, x, y, color, alpha_top=0.30, n_strips=80,
                      depth_decades=1.0, zorder=2):
    rgb = mcolors.to_rgb(color)
    x = np.asarray(x); y = np.asarray(y)
    if np.any(np.diff(x) < 0):
        sort_idx = np.argsort(x); x = x[sort_idx]; y = y[sort_idx]
    ymin, ymax = ax.get_ylim()
    y_base = np.maximum(ymin, y / (10.0 ** depth_decades))
    log_y = np.log10(y); log_base = np.log10(y_base)
    for i in range(n_strips):
        frac_low = i / n_strips; frac_high = (i + 1) / n_strips
        y_low  = 10.0 ** (log_base + frac_low  * (log_y - log_base))
        y_high = 10.0 ** (log_base + frac_high * (log_y - log_base))
        a = (((i + 0.5) / n_strips) ** 2) * alpha_top
        if a <= 0: continue
        mask = y_high > y_low
        if not np.any(mask): continue
        ax.fill_between(x, y_low, y_high, where=mask, color=rgb, alpha=a,
                        linewidth=0, zorder=zorder, interpolate=True)

c_launch_values = [5000, 500, 50]
target_volume   = 300
v_grid          = np.linspace(1, 3000, 400)
v_grid_log      = np.logspace(-1, 4, 400)
cost_components = ['Development', 'Production', 'Transportation', 'Construction']

zorder_config = {
    2: {'Xuanwu': 30, 'Lunar Ampullae': 20, 'Lunar Safe Haven': 40,
        'Starship HLS': 50, 'Moon Village': 10},
    1: {'Xuanwu': 50, 'Lunar Ampullae': 10, 'Lunar Safe Haven': 30,
        'Starship HLS': 40, 'Moon Village': 20},
    0: {'Xuanwu': 50, 'Lunar Ampullae': 10, 'Lunar Safe Haven': 30,
        'Starship HLS': 40, 'Moon Village': 20},
}

fig = plt.figure(figsize=(7, 6.1))

gs_top = fig.add_gridspec(2, 7,
                          width_ratios=[1, 0.15, 1, 0.15, 1, 0.05, 0.04],
                          height_ratios=[1.35, 0.62],
                          hspace=0.70, wspace=0,
                          left=0.12, right=0.94, top=0.955, bottom=0.455)
axes_top    = [fig.add_subplot(gs_top[0, j]) for j in (0, 2, 4)]
axes_bottom = [fig.add_subplot(gs_top[1, j]) for j in (0, 2, 4)]
cbar_ax     = fig.add_subplot(gs_top[1, 6])

gs_low = fig.add_gridspec(1, 1, left=0.12, right=0.918, top=0.335, bottom=0.085)
ax_mid = fig.add_subplot(gs_low[0, 0])

# ------------------------------- A -------------------------------
for i, (ax, c_launch) in enumerate(zip(axes_top, c_launch_values)):
    ax.set_yscale('log'); ax.set_xlim(0, 3000); ax.set_ylim(3e-2, 3e2)
    ax.axvline(target_volume, color='#555555', linestyle=':',
               alpha=0.7, zorder=2, linewidth=0.5)

    curve_data = [(idx, name, SCHEMES[name],
                   compute_unit_cost(SCHEMES[name], c_launch, v_grid))
                  for idx, name in enumerate(SCHEME_NAMES)]
    scheme_z_map = zorder_config[i]

    for idx, name, s, costs in curve_data:
        gradient_fill_log(ax, v_grid, costs, color=SCHEME_COLORS[idx],
                          alpha_top=0.15, n_strips=20, depth_decades=1,
                          zorder=scheme_z_map[name])
    for idx, name, s, costs in curve_data:
        ax.plot(v_grid, costs, color=SCHEME_COLORS[idx], linewidth=0.8,
                label=name, zorder=scheme_z_map[name] + 1)
    for idx, name, s, costs in curve_data:
        V_single = s['V_pressurized']
        ax.plot(V_single, compute_unit_cost(s, c_launch, V_single), 'o',
                color=SCHEME_COLORS[idx], markersize=2.5,
                markeredgecolor=SCHEME_COLORS[idx], markeredgewidth=0.7,
                zorder=scheme_z_map[name] + 50)

    if i == 0:
        ax.text(target_volume + 60, 1.5e2, f'V = {target_volume} m³',
                color='#555555', fontsize=7, ha='left', va='top',
                style='italic', fontweight='bold')

    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_xlabel(r'Pressurized volume $V$ (m³)')
    ax.set_title(rf'$\boldsymbol{{C}}_{{{{\mathbf{{Launch}}}}}} \mathbf{{=}} \mathbf{{\${c_launch}/}}\mathbf{{kg}}$',
                 pad=4)
    ax.grid(True, which='major', axis='y', alpha=0.25, zorder=5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('black'); spine.set_linewidth(0.5); spine.set_zorder(100)
    if i > 0:
        ax.set_yticklabels([])

axes_top[0].set_ylabel(r'Unit cost (\$M/m³)')
axes_top[2].legend(loc='upper right', framealpha=0.9, edgecolor=(1, 1, 1, 0.85),
                   labelspacing=0.3, fontsize=6, fancybox=False)

# --------------------- C : Earth dependence ---------------------
ax_mid.set_xscale('log'); ax_mid.set_yscale('log')
ax_mid.set_xlim(1e-1, 1e4); ax_mid.set_ylim(5e-2, 5e3)

for idx, name in enumerate(SCHEME_NAMES):
    s = SCHEMES[name]
    D = compute_earth_dependence(s, v_grid_log)
    ax_mid.plot(v_grid_log, D, color=SCHEME_COLORS[idx], linewidth=1.0,
                zorder=10 + idx, label=name)

    floor = earth_dependence_floor(s)
    if floor > 0:
        # ax_mid.axhline(floor, color=SCHEME_COLORS[idx], linestyle='--',
        #                linewidth=0.5, alpha=0.55, zorder=3)
        ax_mid.text(1.2e4, floor, f'{floor:.0f}', color=SCHEME_COLORS[idx],
                    fontsize=6, ha='left', va='center', fontweight='bold')

    V_single = s['V_pressurized']
    ax_mid.plot(V_single, compute_earth_dependence(s, V_single), 'o',
                color=SCHEME_COLORS[idx], markersize=2.5,
                markeredgecolor=SCHEME_COLORS[idx], markeredgewidth=0.7, zorder=30)


_x_lo, _x_hi = ax_mid.get_xlim()
_y_lo        = ax_mid.get_ylim()[0]
_tick_tf     = ax_mid.get_xaxis_transform()
for idx, name in enumerate(SCHEME_NAMES):
    s = SCHEMES[name]
    floor = earth_dependence_floor(s)
    if floor <= 0:
        continue
    V_knee = ALPHA * s['m_equipment'] / floor
    if not (_x_lo < V_knee < _x_hi):
        continue
    D_knee = compute_earth_dependence(s, V_knee)
    ax_mid.plot([V_knee, V_knee], [_y_lo, D_knee], color=SCHEME_COLORS[idx],
                linestyle='--', linewidth=0.5, alpha=0.8, zorder=4)
    ax_mid.plot(V_knee, D_knee, marker='D', markersize=3, markerfacecolor='white',
                markeredgecolor=SCHEME_COLORS[idx], markeredgewidth=0.7, linestyle='none', zorder=25)
    # ax_mid.plot([V_knee, V_knee], [0, 0.05], transform=_tick_tf,
    #             color=SCHEME_COLORS[idx], linewidth=1.0, solid_capstyle='butt',
    #             clip_on=False, zorder=101)


# ax_mid.axvline(target_volume, color='#555555', linestyle=':',
#                alpha=0.7, linewidth=0.5, zorder=2)

_x_lab = 1.4e4
_y_lab = ALPHA * SCHEMES['Xuanwu']['m_equipment'] / _x_lab
ax_mid.text(_x_lab, _y_lab, 'No \n  floor',
            color=SCHEME_COLORS[0], fontsize=6,
            rotation_mode='anchor', ha='center', va='bottom', fontweight='bold')

ax_mid.text(1.015, 1.005, r'$\boldsymbol{D}_{{\mathbf{Earth},\mathbf{\infty}}}$', transform=ax_mid.transAxes,
            color='#111111', fontsize=8, ha='left', va='bottom')

# ax_mid.text(0.012, 0.955, r'Independent of $c_{\mathrm{Launch}}$',
#             transform=ax_mid.transAxes, fontsize=6.5, color='#333333',
#             style='italic', ha='left', va='top')
ax_mid.set_xlabel(r'Pressurized volume $V$ (m³)')
ax_mid.set_ylabel(r'Earth dependence $D_\mathrm{Earth}$ (kg/m³)')
ax_mid.grid(True, which='major', alpha=0.25, zorder=0)
ax_mid.set_axisbelow(True)
for spine in ax_mid.spines.values():
    spine.set_color('black'); spine.set_linewidth(0.5); spine.set_zorder(100)
# ax_mid.legend(loc='lower left', framealpha=0.9, edgecolor=(1, 1, 1, 0.85),
#               labelspacing=0.3, fontsize=6, fancybox=False, ncol=1)

# --------------------- B : cost composition ---------------------
im_last = None
for i, (ax, c_launch) in enumerate(zip(axes_bottom, c_launch_values)):
    composition_matrix = []
    for scheme_name in SCHEME_NAMES:
        s = SCHEMES[scheme_name]
        scaled_mass = target_volume * s['m_structure'] / s['V_pressurized'] * 1e3
        cost_dev  = s['C_development']
        cost_prod = s['P_production']   * scaled_mass / 1e6
        cost_con  = s['P_construction'] * scaled_mass / 1e6
        G_total   = compute_G_total(s['DV'], I_SP)
        m_t1 = G_total * (s['m_prefabrication'] + s['m_material'])
        m_t2 = G_total * ALPHA * s['m_equipment']
        cost_trans = (m_t1 * c_launch / s['m_structure'] * scaled_mass
                      + m_t2 * c_launch) / 1e6
        total = cost_dev + cost_prod + cost_trans + cost_con
        composition_matrix.append([100 * cost_dev / total, 100 * cost_prod / total,
                                   100 * cost_trans / total, 100 * cost_con / total])
    composition_matrix = np.array(composition_matrix)

    im = ax.imshow(composition_matrix.T, cmap='Blues', aspect='auto',
                   vmin=0, vmax=100, interpolation='nearest')
    im_last = im

    ax.set_xticks(np.arange(len(SCHEME_NAMES)))
    ax.set_xticklabels(SCHEME_NAMES, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(cost_components)))
    ax.set_yticklabels(cost_components if i == 0 else [])

    for row in range(composition_matrix.shape[1]):
        for col in range(composition_matrix.shape[0]):
            val = composition_matrix[col, row]
            ax.text(col, row, f'{val:.0f}', ha='center', va='center',
                    color='white' if val > 50 else 'black')
    for spine in ax.spines.values():
        spine.set_color('black'); spine.set_linewidth(0.5)

cbar = fig.colorbar(im_last, cax=cbar_ax)
cbar.set_label('Cost contribution (%)')

# 面板字母
for ax, letter in ((axes_top[0], 'A'), (axes_bottom[0], 'B'), (ax_mid, 'C')):
    bb = ax.get_position()
    fig.text(0.001, bb.y1 + 0.045, letter, fontsize=10, fontweight='bold',
             ha='left', va='top')

plt.savefig('./outputs_1006/fig3_combined.tif', dpi=1000,
            pil_kwargs={"compression": "tiff_lzw"})
print("\nFigure 3 (A cost / B Earth dependence / C composition) Done")


# =============================================================================
# FIGURE 4 - Zoning map
# =============================================================================

c_launch_grid = np.logspace(0, 5, 2000)
scale_grid    = np.logspace(1, 5, 1000)

optimal_idx  = np.zeros((len(scale_grid), len(c_launch_grid)), dtype=int)
optimal_cost = np.zeros_like(optimal_idx, dtype=float)

for i, V in enumerate(scale_grid):
    for j, c in enumerate(c_launch_grid):
        costs = [compute_unit_cost(SCHEMES[n], c, V) for n in SCHEME_NAMES]
        optimal_idx[i, j]  = int(np.argmin(costs))
        optimal_cost[i, j] = float(np.min(costs))

fig, ax = plt.subplots(figsize=(7, 2.8))

cmap = ListedColormap(SCHEME_COLORS)
norm = BoundaryNorm(np.arange(-0.5, len(SCHEME_NAMES)), len(SCHEME_NAMES))
C_mesh, S_mesh = np.meshgrid(c_launch_grid, scale_grid)

log_cost = np.log10(np.clip(optimal_cost, 1e-6, None))
from matplotlib.colors import to_rgb

levels = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]
log_levels = np.log10(levels)
n_bands = len(levels) + 1

band_idx = np.digitize(log_cost, log_levels)


facecolor = np.zeros((*optimal_idx.shape, 4))
for k in range(len(SCHEME_NAMES)):
    base = to_rgb(SCHEME_COLORS[k])
    mask = (optimal_idx == k)
    facecolor[mask, 0] = base[0]
    facecolor[mask, 1] = base[1]
    facecolor[mask, 2] = base[2]

t = band_idx / (n_bands - 1)
alpha_map = 0.4 + 0.6 * t
facecolor[..., 3] = alpha_map


qm = ax.pcolormesh(C_mesh, S_mesh, np.zeros_like(optimal_idx),
                   shading='auto')
qm.set_facecolor(facecolor.reshape(-1, 4))
qm.set_array(None)


CS = ax.contour(C_mesh, S_mesh, log_cost,
                levels=np.log10([0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]),
                colors='black', linewidths=0.1, alpha=0.1, linestyles='-')


manual_positions = [
    (12.86, 75006),
    (16, 3564),
    (39, 2036),
    (42.6, 855),
    (180, 260),
    (218, 160),
    (278, 73),
    (772, 27),
    (775, 16),
    (15936, 30)
]

labels=ax.clabel(CS, manual=manual_positions, inline=False, fmt=lambda v: f'{10**v:.2f}',
          fontsize=6,
          colors='black')

for txt in labels:
    txt.set_bbox(None)
    txt.set_rotation(0)
    txt.set_alpha(0.15)

fig.canvas.draw()

lw_points = 2.0
dpi = fig.dpi
lw_pixels = lw_points * dpi / 72.0
half_px = lw_pixels / 2.0


bbox = ax.get_window_extent()
ax_px_width = bbox.width
x0, x1 = ax.get_xlim()
log_span = np.log10(x1) - np.log10(x0)


log_per_px = log_span / ax_px_width
delta_log = half_px * log_per_px

lo, hi = 1800, 45000
lo_edge = lo * 10**(-delta_log)
hi_edge = hi * 10**(+delta_log)

ax.axvspan(lo_edge, hi_edge, facecolor='#66CCFF', alpha=0.20, zorder=10, edgecolor='none')
ax.text(np.sqrt(1800*45000), 12, 'Current launch capability',
        fontsize=7, style='italic', ha='center', va='bottom', color='#333333', zorder=11, fontweight='bold')



texts = []
linelist = [
    (3100,  '◐ Falcon 9',      '-', -300),
    (1800,  '◐ Falcon Heavy',  '-', -150),
    (45000, '○ SLS (1/1B/2)',  '-', -4000),
    (1200,  '● Starship V1',   '-', -90),
    (500,   '● Starship V2',   '-', -40),
    (120,   '● Starship V3',   '-', -10),
    (20,    '● Starship V4',   '-', -1.5),
    (9300,  '○ Long March 5',  '-', -800),
    (4500,  '● Long March 10', '-', -500),
    (5400,  '○ Ariane 6',      '-', 200)
]

for c_ref, label, ls, offset in linelist:
    ax.axvline(c_ref, color='black', linestyle=ls, linewidth=2, alpha=0.06, zorder=15)
    ax.text(c_ref + offset, 1.01,
            label, color='black',
            fontsize=6,
            va='bottom', ha='left', rotation=30,
            transform=ax.get_xaxis_transform(),
            # fontweight='bold'
            )

ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='x')
ax.tick_params(axis='y')
ax.set_xlabel(r'Launch cost $C_{\mathrm{Launch}}$ (\$/kg)')
ax.set_ylabel(r'Pressurized volume $V$ (m³)')

legend_patches = [mpatches.Patch(color=SCHEME_COLORS[k], label=SCHEME_NAMES[k])
                  for k in range(len(SCHEME_NAMES))]


leg = ax.legend(handles=legend_patches, loc='upper left',
                fancybox=False, framealpha=0.65, labelspacing=0.2, fontsize=6)
leg.get_frame().set_edgecolor('none')
leg.get_frame().set_linewidth(0)

plt.tight_layout()
plt.savefig('./outputs_1006/fig4_zoning_map.tif', dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
# plt.show()
print("\nFigure 4 Done")
#
# =============================================================================
# FIGURE 5 — Sensitivity heatmap
# =============================================================================

FILE_PATH = './Sensitivity.xlsx'
df = pd.read_excel(FILE_PATH, header=0)

stage_labels = df.iloc[:, 0].astype(str)
param_labels = df.iloc[:, 1].astype(str)
# row_labels = stage_labels + ' | ' + param_labels
row_labels = param_labels
col_labels = df.columns[2:]
data = df.iloc[:, 2:].to_numpy(dtype=float)
data_vis = np.cbrt(data)

n_rows, n_cols = data.shape
fig, ax = plt.subplots(figsize=(3.42, 3.42))

vmin, vmax = -1, 1
if vmin > 0 and vmax > 0 and vmax / vmin > 10:
    norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    norm = LogNorm(vmin=vmin, vmax=vmax) if (vmin > 0 and vmax > 0) else plt.Normalize(vmin=vmin, vmax=vmax)

# cmap = plt.cm.viridis
cmap = plt.cm.bwr

x_edges = np.arange(n_cols + 1)
y_edges = np.arange(n_rows + 1)
pcm = ax.pcolormesh(x_edges, y_edges, data_vis,
                    norm=norm, cmap=cmap,
                    shading='flat',
                    rasterized=True)

ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_xticklabels(col_labels, rotation=20, ha='right')
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels(row_labels)

for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_zorder(10)

SIG_THRESHOLD = 0.0049

for row in range(n_rows):
    for col in range(n_cols):
        raw = data[row, col]
        vis = data_vis[row, col]
        if np.isnan(raw) or abs(raw) < SIG_THRESHOLD:
            continue


        text_color = 'white' if abs(vis) > 0.6 else 'black'
        ax.text(col + 0.5, row + 0.5, f'{raw:.2f}',
                ha='center', va='center',
                color=text_color, fontsize=7, zorder=6)

fig.subplots_adjust(left=0.41, right=0.81, bottom=0.15, top=0.98)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.83])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label('Sensitivity value (cube‑root scale)')

plt.savefig('./outputs_1006/fig5_sensitivity_heatmap.tif', dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
# plt.show()
print("\nFigure 5 Done")

# =============================================================================
# FIGURE 6 — Cost-reliability Pareto structure at three construction scales
# =============================================================================

RELIABILITY = {
    'Xuanwu'           : 0.9500,
    'Lunar Ampullae'   : 0.9317,
    'Lunar Safe Haven' : 0.9093,
    'Starship HLS'     : 0.8989,
    'Moon Village'     : 0.9387,
}

target_volumes_fig6  = [30, 300, 3000]
launch_price_fig6    = [5000, 500, 50]

regime_markers = ['o', 's', '^']
regime_alpha   = [1.0, 0.65, 0.3]


def pareto_front(points):
    """points: list of (cost, rel, idx). Returns Pareto-optimal subset."""
    pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_rel = -np.inf
    for cost, rel, idx in pts:
        if rel > best_rel:
            frontier.append((cost, rel, idx))
            best_rel = rel
    return frontier

label_positions_middle = {
    'Xuanwu'           : (0.015, 0.95),
    'Lunar Ampullae'   : (0.015, 0.9317),
    'Lunar Safe Haven' : (0.015, 0.9093),
    'Starship HLS'     : (0.015, 0.8989),
    'Moon Village'     : (0.015, 0.9387),
}


fig = plt.figure(figsize=(7, 2.5))
gs = GridSpec(1, 3, figure=fig,
              wspace=0.15,
              left=0.08, right=0.98, top=0.86, bottom=0.15)

axes_fig6 = [fig.add_subplot(gs[0, j]) for j in range(3)]

for panel_i, (ax, V_target) in enumerate(zip(axes_fig6, target_volumes_fig6)):

    for idx, name in enumerate(SCHEME_NAMES):
        rel = RELIABILITY[name]
        costs = [compute_unit_cost(SCHEMES[name], c, V_target)
                 for c in launch_price_fig6]
        ax.plot(costs, [rel] * len(costs),
                color=SCHEME_COLORS[idx],
                linewidth=4, alpha=0.35,
                solid_capstyle='round',
                zorder=2)

    for r_i, c_launch in enumerate(launch_price_fig6):
        points = []
        for idx, name in enumerate(SCHEME_NAMES):
            rel = RELIABILITY[name]
            if rel is None:
                continue
            cost = compute_unit_cost(SCHEMES[name], c_launch, V_target)
            points.append((cost, rel, idx))

        if not points:
            continue

        for cost, rel, idx in points:
            ax.scatter(cost, rel,
                       marker=regime_markers[r_i],
                       s=20,
                       facecolor=SCHEME_COLORS[idx],
                       edgecolor='black',
                       linewidth=0.5,
                       alpha=regime_alpha[r_i],
                       zorder=10)

        front = pareto_front(points)
        if len(front) >= 2:
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(fx, fy, linestyle='--', color='black',
                    linewidth=0.5,
                    alpha=regime_alpha[r_i], zorder=5)
        elif len(front) == 1:
            cost, rel, idx = front[0]
            ax.scatter(cost, rel, s=100, facecolor='none',
                       edgecolor='black',
                       linewidth=0.5,
                       linestyle='--', alpha=regime_alpha[r_i], zorder=4)

    if panel_i == 0:
        for idx, name in enumerate(SCHEME_NAMES):
            x, y = label_positions_middle[name]
            ax.text(x, y, name,
                    color=SCHEME_COLORS[idx],
                    fontsize=7, fontweight='bold',
                    ha='left', va='center', zorder=11)


    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.89, 0.96)

    ax.set_xlabel(r'Unit cost (\$M/m³)')
    if panel_i == 0:
        ax.set_ylabel('Relative reliability')
    else:
        ax.set_yticklabels([])

    ax.set_title(rf'$\boldsymbol{{V}} \mathbf{{=}}  \mathbf{{{V_target}}}\,\mathbf{{m^3}}$', pad=4, fontsize=7)

    ax.grid(True, which='major', alpha=0.25, linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)


ax_right = axes_fig6[2]
ax_right.annotate('',
                  xy=(0.82, 0.88), xytext=(0.92, 0.78),
                  xycoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color='black',
                                  lw=1, alpha=0.7))
ax_right.text(0.92, 0.92, 'Ideal',
              transform=ax_right.transAxes,
              fontsize=7, ha='center', va='top',
              color='black', alpha=0.8, style='italic', fontweight='bold')


regime_handles = [
    Line2D([0], [0], marker=regime_markers[i], color='black',
           markerfacecolor='gray', markeredgecolor='black',
           linestyle='None', linewidth=0.5, markersize=4,
           alpha=regime_alpha[i],
           markeredgewidth=0.3,
           label=rf'$C_{{\mathrm{{Launch}}}} = \${launch_price_fig6[i]}/\mathrm{{kg}}$')
    for i in range(len(launch_price_fig6))
]
ax_right.legend(handles=regime_handles, loc='lower right', bbox_to_anchor=(1, 0),borderaxespad=0.,
                framealpha=0, fancybox=False, fontsize=6,handlelength=0.3,
                edgecolor=(1, 1, 1, 0.85))


for panel_i, ax in enumerate(axes_fig6):
    letter = chr(ord('A') + panel_i)
    bbox = ax.get_position()
    fig.text(bbox.x0-0.01, bbox.y1 + 0.11,
             letter, fontsize=10, fontweight='bold',
             ha='left', va='top')


plt.savefig('./outputs_1006/fig6_pareto_frontier.tif',
            dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
# plt.show()
print("\nFigure 6 Done")