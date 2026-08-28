import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 31.6,
    "legend.fontsize": 28.6,
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "pdf.compression": 9,
})

n = 10
a = (3 / 4) ** n
folder_path = rf"xxxx"  # User defined
out_dir = Path(folder_path)
out_dir.mkdir(parents=True, exist_ok=True)

R_values = np.logspace(-0.2, 8.2, 501)

x_fixed = 1e5
eps_inv_vals = np.logspace(-0.12, 4.15, 501)

EI, Rl = np.meshgrid(eps_inv_vals, R_values, indexing="xy")

Al = a + (1 - a) * EI**2 / Rl
Gl = (EI**2 * (1 + x_fixed)) / (Al * (Rl + x_fixed))

coeff_l = np.sqrt((1.0 / a - 1.0) * x_fixed)

eps_fixed = 1e-4
xv = np.logspace(-0.12, 5.15, 501)

Xg, Rr = np.meshgrid(xv, R_values, indexing="xy")

Ar = a + (1 - a) / (Rr * eps_fixed**2)
Gr = (eps_fixed**(-2) * (1 + Xg)) / (Ar * (Rr + Xg))

coeff_r = np.sqrt((1.0 / a - 1.0)) / eps_fixed

tiny = np.finfo(float).tiny

Gl_plot = np.log10(np.clip(Gl, tiny, None))
Gr_plot = np.log10(np.clip(Gr, tiny, None))

G_all_plot = np.concatenate([Gl_plot.ravel(), Gr_plot.ravel()])
G_all_plot = G_all_plot[np.isfinite(G_all_plot)]

p_low = 0.5
p_high = 99.5
gamma = 0.8

vmin_log = np.percentile(G_all_plot, p_low)
vmax_log = np.percentile(G_all_plot, p_high)

norm = PowerNorm(
    gamma=gamma,
    vmin=vmin_log,
    vmax=vmax_log,
    clip=False
)

print(f"Gl range: {Gl.min():.4e} to {Gl.max():.4e}")
print(f"Gr range: {Gr.min():.4e} to {Gr.max():.4e}")
print(f"Displayed color range: {10**vmin_log:.4e} to {10**vmax_log:.4e}")
print(f"gamma = {gamma}")

l = 10.5 * 9 / 9.5

fig, (axL, axR) = plt.subplots(
    1,
    2,
    figsize=(2 * l, l * 0.75),
    sharey=True
)

fig.subplots_adjust(
    left=0.06,
    right=0.965,
    top=0.975,
    bottom=0.12,
    wspace=0.035,
)

axL.set_box_aspect(0.8)
axR.set_box_aspect(0.8)

ml = axL.pcolormesh(
    eps_inv_vals,
    R_values,
    Gl_plot,
    shading="auto",
    norm=norm,
    cmap="viridis",
    rasterized=True,
)

axL.set_xscale("log")
axL.set_yscale("log")

axL.set_xlabel(r"$\epsilon^{-1}$", labelpad=1)
axL.set_ylabel(r"$R$", labelpad=1)

axL.tick_params(
    axis="both",
    which="major",
    labelsize=31.6
)

levels = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
vl = [lv for lv in levels if Gl.min() < lv < Gl.max()]

x_label_left = 5 * 10**3

if vl:
    cl = axL.contour(
        eps_inv_vals,
        R_values,
        Gl,
        levels=vl,
        colors="black",
        linewidths=1.2
    )

    manual_positions = []
    ix = np.argmin(np.abs(eps_inv_vals - x_label_left))

    for lv in vl:
        g_col = Gl[:, ix]

        crossings = []
        for j in range(len(R_values) - 1):
            if (g_col[j] - lv) * (g_col[j + 1] - lv) <= 0:
                if g_col[j] == g_col[j + 1]:
                    continue

                R_cross = (
                    R_values[j]
                    + (lv - g_col[j])
                    * (R_values[j + 1] - R_values[j])
                    / (g_col[j + 1] - g_col[j])
                )

                crossings.append(R_cross)

        if crossings:
            R_lv = min(crossings)

            if R_values[0] <= R_lv <= R_values[-1]:
                manual_positions.append((x_label_left, R_lv))

    if manual_positions:
        axL.clabel(
            cl,
            manual=manual_positions,
            inline=True,
            inline_spacing=5,
            fontsize=28.6,
            fmt={
                lv: r"$10^{" + str(int(np.log10(lv))) + r"}$"
                for lv in vl
            },
        )

        for label in cl.labelTexts:
            label.set_rotation(0)

Rop_l = coeff_l * eps_inv_vals
mk_l = (Rop_l >= R_values[0]) & (Rop_l <= R_values[-1])

axL.plot(
    eps_inv_vals[mk_l],
    Rop_l[mk_l],
    linestyle="--",
    linewidth=2.0,
    color="red"
)

label_x_left_opt = 1e2

axL.text(
    label_x_left_opt,
    8e3,
    r"$R_{\rm opt}=\epsilon^{-1}\sqrt{[2^{M_2(\sigma)}-1]x}$",
    rotation=0,
    ha="center",
    va="top",
    fontsize=28.6,
    color="red"
)

axL.set_xticks([10**i for i in range(0, 5)])

mr = axR.pcolormesh(
    xv,
    R_values,
    Gr_plot,
    shading="auto",
    norm=norm,
    cmap="viridis",
    rasterized=True,
)

axR.set_xscale("log")
axR.set_yscale("log")

axR.set_xlabel(r"$x$", labelpad=1)

axR.tick_params(
    axis="both",
    which="major",
    labelsize=31.6
)

vr = [lv for lv in levels if Gr.min() < lv < Gr.max()]
x_label_right = 5 * 10**4

if vr:
    cr = axR.contour(
        xv,
        R_values,
        Gr,
        levels=vr,
        colors="black",
        linewidths=1.2
    )

    manual_positions_r = []
    ix_r = np.argmin(np.abs(xv - x_label_right))

    for lv in vr:
        g_col = Gr[:, ix_r]

        crossings = []
        for j in range(len(R_values) - 1):
            if (g_col[j] - lv) * (g_col[j + 1] - lv) <= 0:
                if g_col[j] == g_col[j + 1]:
                    continue

                R_cross = (
                    R_values[j]
                    + (lv - g_col[j])
                    * (R_values[j + 1] - R_values[j])
                    / (g_col[j + 1] - g_col[j])
                )

                crossings.append(R_cross)

        if crossings:
            R_lv = min(crossings)

            if R_values[0] <= R_lv <= R_values[-1]:
                manual_positions_r.append((x_label_right, R_lv))

    if manual_positions_r:
        axR.clabel(
            cr,
            manual=manual_positions_r,
            inline=True,
            inline_spacing=5,
            fontsize=28.6,
            fmt={
                lv: r"$10^{" + str(int(np.log10(lv))) + r"}$"
                for lv in vr
            },
        )

        for label in cr.labelTexts:
            label.set_rotation(0)

Rop_r = coeff_r * np.sqrt(xv)
mk_r = (Rop_r >= R_values[0]) & (Rop_r <= R_values[-1])

axR.plot(
    xv[mk_r],
    Rop_r[mk_r],
    linestyle="--",
    linewidth=2.0,
    color="red"
)

axR.text(
    7e2,
    2e5,
    r"$R_{\rm opt}=\epsilon^{-1}\sqrt{[2^{M_2(\sigma)}-1]x}$",
    rotation=0,
    ha="center",
    va="top",
    fontsize=28.6,
    color="red"
)

axR.set_xticks([10**i for i in range(0, 6, 1)])

cbar = fig.colorbar(
    mr,
    ax=[axL, axR],
    label=r"$\Lambda(R,x,\epsilon)$",
    pad=0.01,
    fraction=0.06,
    aspect=20,
)

cbar.ax.yaxis.labelpad = -0.01

fig.canvas.draw()

subplot_bbox = axL.get_position()
cbar_bbox = cbar.ax.get_position()

cbar.ax.set_position([
    cbar_bbox.x0,
    subplot_bbox.y0,
    cbar_bbox.width,
    subplot_bbox.height,
])

tick_min = -1
tick_max = 5
tick_exponents = np.arange(tick_min, tick_max + 1, dtype=int)

if len(tick_exponents) > 7:
    tick_exponents = tick_exponents[::2]

cbar.set_ticks(tick_exponents)
cbar.set_ticklabels([
    r"$10^{" + str(k) + r"}$"
    for k in tick_exponents
])

axL.set_yticks([10**i for i in range(0, 9, 1)])

fig.savefig(
    out_dir / "gain_combined_n{}.pdf".format(n),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    out_dir / f"gain_combined_n{n}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
