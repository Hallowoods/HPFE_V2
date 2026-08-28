import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

folder_path = rf"xxxx"  # User defined
out_dir = Path(folder_path)
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 36.0,
    "axes.labelsize": 38.0,
    "xtick.labelsize": 34.0,
    "ytick.labelsize": 34.0,
    "legend.fontsize": 32.0,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.7,
})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Tinos"]

n = 10
x_values = [1e5, 1e8]
epsilon_inv_values = [1e1, 1e2, 1e3, 1e5]
m2_values = np.linspace(0.0, float(n), 1201)

def lambda_opt(m2, x, epsilon):
    m2 = np.asarray(m2, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    two_minus_m2 = np.exp2(-m2)
    denominator = (
        np.sqrt(np.clip(1.0 - two_minus_m2, 0.0, None))
        + epsilon * np.sqrt(x * two_minus_m2)
    ) ** 2
    return (1.0 + x) / denominator

l = 10.5 * 9 / 9.5
fig, axes = plt.subplots(1, 2, figsize=(2 * l, l * 0.75), sharey=True)

cmap = plt.get_cmap("tab10")
plot_formats = ["o-", "s--", "^-.", "D:"]

for ax, x in zip(axes, x_values):
    for i, (epsilon_inv, fmt) in enumerate(zip(epsilon_inv_values, plot_formats)):
        epsilon = 1.0 / epsilon_inv
        exponent = int(np.log10(epsilon_inv))
        ax.plot(
            m2_values,
            lambda_opt(m2_values, x=x, epsilon=epsilon),
            fmt,
            alpha=0.75,
            markersize=12,
            markeredgewidth=1.4,
            color=cmap(i),
            linewidth=3.2,
            markevery=100,
            label=fr"$\epsilon^{{-1}}=10^{{{exponent}}}$",
        )

    ax.set_yscale("log")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(5e1, 2e10)
    ax.set_xlabel(r"$M_2(\sigma)$")

    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_xticks(np.arange(0, 11, 1), minor=True)

    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=False,
        bottom=True,
        length=7,
        width=1.4,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="in",
        top=False,
        bottom=True,
        length=3.5,
        width=1.0,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        right=False,
        left=True,
        length=7,
        width=1.4,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        right=False,
        left=True,
        length=0,
    )

axes[0].set_ylabel(r"$\Lambda_{\mathrm{opt}}$")

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles[::-1],
    labels[::-1],
    loc="upper right",
    frameon=True,
    ncol=1,
    labelspacing=0.30,
    handlelength=2.0,
    handletextpad=0.7,
    borderpad=0.5,
)

plt.tight_layout(pad=0.005)
plt.subplots_adjust(wspace=0.035)

plt.savefig(
    out_dir / "lambda_vs_M2_two_panels_short_ticks_legend_reversed.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    out_dir / "lambda_vs_M2_two_panels_short_ticks_legend_reversed.pdf",
    bbox_inches="tight",
)
plt.show()
