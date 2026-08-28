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


N_QUBITS = 10
K_MAGIC = 10
DIMENSION = 2**N_QUBITS

X_FIXED = 1e5
EPS_INV_MIN = 1.0
EPS_INV_MAX = 10**4.15
N_EPSILON = 251

EPSILON_FIXED = 1e-4
X_VALUES = np.logspace(-0.12, 5.15, 501)

R_VALUES = np.logspace(-0.2, 8.2, 501)

# Monotonic branch used to invert infidelity numerically.
THETA_LOOKUP_MAX = 1.2
N_THETA_LOOKUP = 20001

folder_path = rf"xxxx"  # User defined
OUT_DIR = Path(folder_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = OUT_DIR / "coherent_s77_data"
SAVE_NUMERICAL_DATA = True


PAULIS = np.asarray([
    [[1.0, 0.0], [0.0, 1.0]],          # I
    [[0.0, 1.0], [1.0, 0.0]],          # X
    [[0.0, -1.0j], [1.0j, 0.0]],       # Y
    [[1.0, 0.0], [0.0, -1.0]],         # Z
], dtype=np.complex128)

KET_ZERO = np.asarray([1.0, 0.0], dtype=np.complex128)
KET_T = np.asarray(
    [1.0, np.exp(1.0j * np.pi / 4.0)],
    dtype=np.complex128,
) / np.sqrt(2.0)


def build_target_product_state(n_qubits, k_magic):
    """Return the factors of |0>^(n-k) tensor |T>^k."""
    if not (0 <= k_magic <= n_qubits):
        raise ValueError("K_MAGIC must satisfy 0 <= K_MAGIC <= N_QUBITS.")

    states = []
    for q in range(n_qubits):
        if q < n_qubits - k_magic:
            states.append(KET_ZERO.copy())
        else:
            states.append(KET_T.copy())
    return states


TARGET_QUBITS = build_target_product_state(N_QUBITS, K_MAGIC)


def coherent_rotation(theta):
    """Single-qubit coherent rotation from Appendix J.1(b)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray([
        [np.exp(-1.0j * theta) * c, -s],
        [s, np.exp(1.0j * theta) * c],
    ], dtype=np.complex128)


def build_pauli_digits(n_qubits):
    """Encode every n-qubit Pauli string as a base-4 row."""
    n_paulis = 4**n_qubits
    indices = np.arange(n_paulis, dtype=np.int64)
    digits = np.empty((n_paulis, n_qubits), dtype=np.int8)

    for q in range(n_qubits):
        digits[:, q] = (indices // (4**q)) % 4

    return digits


PAULI_DIGITS = build_pauli_digits(N_QUBITS)
N_PAULIS = PAULI_DIGITS.shape[0]


def local_pauli_table(bra_state, ket_state):
    """Return <bra_state|P|ket_state> for P in {I,X,Y,Z}."""
    return np.asarray(
        [np.vdot(bra_state, pauli @ ket_state) for pauli in PAULIS],
        dtype=np.complex128,
    )


def product_over_pauli_strings(local_tables):
    """Evaluate all tensor-product Pauli strings from local tables."""
    values = np.ones(N_PAULIS, dtype=np.complex128)
    for q, table in enumerate(local_tables):
        values *= table[PAULI_DIGITS[:, q]]
    return values


SIGMA_LOCAL_TABLES = [
    local_pauli_table(ket, ket)
    for ket in TARGET_QUBITS
]
SIGMA_PAULI = product_over_pauli_strings(SIGMA_LOCAL_TABLES)
SIGMA_PAULI = np.real_if_close(SIGMA_PAULI, tol=1000).real

# O = sigma - I/d; the identity coefficient is zero.
OBSERVABLE_PAULI = SIGMA_PAULI.copy()
OBSERVABLE_PAULI[0] = 0.0


def infidelity_from_theta(theta):
    unitary = coherent_rotation(theta)
    overlap = 1.0 + 0.0j

    for target_ket in TARGET_QUBITS:
        noisy_ket = unitary @ target_ket
        overlap *= np.vdot(target_ket, noisy_ket)

    fidelity = np.abs(overlap) ** 2
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def build_theta_lookup():
    theta_lookup = np.linspace(
        0.0,
        THETA_LOOKUP_MAX,
        N_THETA_LOOKUP,
    )
    epsilon_lookup = np.asarray(
        [infidelity_from_theta(theta) for theta in theta_lookup],
        dtype=float,
    )

    # Remove tiny floating-point violations of monotonicity.
    epsilon_lookup = np.maximum.accumulate(epsilon_lookup)

    if np.any(np.diff(epsilon_lookup) < -1e-12):
        raise RuntimeError(
            "The selected theta interval is not monotonic in infidelity."
        )

    return theta_lookup, epsilon_lookup


THETA_LOOKUP, EPSILON_LOOKUP = build_theta_lookup()


def theta_from_infidelity(epsilon):
    """Invert the monotonic map epsilon(theta)."""
    epsilon = np.asarray(epsilon, dtype=float)
    eps_min = EPSILON_LOOKUP[0]
    eps_max = EPSILON_LOOKUP[-1]

    if np.any(epsilon < eps_min - 1e-14):
        raise ValueError("Requested infidelity is below the lookup range.")

    if np.any(epsilon > eps_max):
        print(
            "Warning: requested epsilon exceeds the theta lookup range; "
            f"clipping to epsilon_max={eps_max:.8e}."
        )

    epsilon_clipped = np.clip(epsilon, eps_min, eps_max)
    return np.interp(
        epsilon_clipped,
        EPSILON_LOOKUP,
        THETA_LOOKUP,
    )


EPS_INV_REQUESTED = np.logspace(
    np.log10(EPS_INV_MIN),
    np.log10(EPS_INV_MAX),
    N_EPSILON,
)
EPSILON_REQUESTED = 1.0 / EPS_INV_REQUESTED
THETA_VALUES = theta_from_infidelity(EPSILON_REQUESTED)


def clifford_vstar(
    tau_pauli,
    twisted_tau_o,
    trace_tau_o,
):
    """Evaluate V_*(O, tau) from Eqs. (21) and (22)."""
    xi_tau_o = np.real(tau_pauli * OBSERVABLE_PAULI)
    twisted_tau_o = np.real(twisted_tau_o)

    xi_norm_sq = float(np.dot(xi_tau_o, xi_tau_o))
    twisted_dot = float(np.dot(twisted_tau_o, xi_tau_o))

    v_circle = (
        (DIMENSION + 1.0)
        / (DIMENSION * (DIMENSION + 2.0))
        * (xi_norm_sq + twisted_dot)
    )
    v_star = v_circle - trace_tau_o**2 / (DIMENSION + 2.0)

    if v_star < -1e-10:
        raise RuntimeError(
            f"V_* is unexpectedly negative: {v_star:.8e}"
        )

    return max(float(v_star), 0.0), xi_norm_sq, twisted_dot


def standard_shadow_variance(fidelity):
    """Equation (13) for O = sigma - I/d and pure sigma."""
    trace_o2 = 1.0 - 1.0 / DIMENSION
    trace_rho_o2 = (
        (1.0 - 2.0 / DIMENSION) * fidelity
        + 1.0 / DIMENSION**2
    )
    trace_rho_o = fidelity - 1.0 / DIMENSION

    variance = (
        (DIMENSION + 1.0)
        / (DIMENSION + 2.0)
        * (trace_o2 + 2.0 * trace_rho_o2)
        - trace_rho_o**2
    )
    return float(variance)


def variance_components_at_theta(theta):
    """Compute the theta-dependent terms used in Eq. (6)."""
    unitary = coherent_rotation(theta)

    rho_local_tables = []
    cross_local_tables = []

    for target_ket in TARGET_QUBITS:
        noisy_ket = unitary @ target_ket
        rho_local_tables.append(
            local_pauli_table(noisy_ket, noisy_ket)
        )
        cross_local_tables.append(
            local_pauli_table(noisy_ket, target_ket)
        )

    rho_pauli = product_over_pauli_strings(rho_local_tables)
    rho_pauli = np.real_if_close(rho_pauli, tol=1000).real

    # cross(P) = <phi|P|psi>, where rho=|phi><phi| and
    # sigma=|psi><psi|.
    cross_pauli = product_over_pauli_strings(cross_local_tables)
    cross_abs_sq = np.abs(cross_pauli) ** 2

    fidelity = float(np.clip(cross_abs_sq[0], 0.0, 1.0))
    epsilon = float(1.0 - fidelity)

    delta_pauli = rho_pauli - SIGMA_PAULI

    # tildeXi_{rho,O}(P) = Tr[rho P O P]
    #                       = |<phi|P|psi>|^2 - 1/d.
    twisted_rho_o = cross_abs_sq - 1.0 / DIMENSION

    # tildeXi_{Delta,O}(P)
    #   = |<phi|P|psi>|^2 - |<psi|P|psi>|^2.
    twisted_delta_o = cross_abs_sq - SIGMA_PAULI**2

    trace_delta_o = -epsilon
    trace_rho_o = fidelity - 1.0 / DIMENSION

    vstar_delta, xi_delta_norm_sq, twisted_delta_dot = (
        clifford_vstar(
            delta_pauli,
            twisted_delta_o,
            trace_delta_o,
        )
    )
    vstar_rho, xi_rho_norm_sq, twisted_rho_dot = (
        clifford_vstar(
            rho_pauli,
            twisted_rho_o,
            trace_rho_o,
        )
    )

    v_standard = standard_shadow_variance(fidelity)
    shot_coefficient = v_standard - vstar_rho

    if shot_coefficient < -1e-10:
        raise RuntimeError(
            "V(O,rho)-V_*(O,rho) is unexpectedly negative: "
            f"{shot_coefficient:.8e}"
        )
    shot_coefficient = max(float(shot_coefficient), 0.0)

    return {
        "theta": float(theta),
        "epsilon": epsilon,
        "epsilon_inverse": 1.0 / epsilon,
        "fidelity": fidelity,
        "V_standard": v_standard,
        "Vstar_delta": vstar_delta,
        "Vstar_rho": vstar_rho,
        "shot_coefficient": shot_coefficient,
        "Xi_delta_norm_sq": xi_delta_norm_sq,
        "twisted_delta_dot": twisted_delta_dot,
        "Xi_rho_norm_sq": xi_rho_norm_sq,
        "twisted_rho_dot": twisted_rho_dot,
    }


def scan_variance_components(theta_values):
    results = []
    total = len(theta_values)

    for index, theta in enumerate(theta_values):
        results.append(variance_components_at_theta(theta))

        if (
            index == 0
            or (index + 1) % 25 == 0
            or index + 1 == total
        ):
            print(
                "Variance scan: "
                f"{index + 1:4d}/{total}, "
                f"theta={theta:.6e}, "
                f"epsilon={results[-1]['epsilon']:.6e}"
            )

    return results


VARIANCE_RESULTS = scan_variance_components(THETA_VALUES)


def result_column(name):
    return np.asarray(
        [result[name] for result in VARIANCE_RESULTS],
        dtype=float,
    )


THETA_VALUES = result_column("theta")
EPSILON_VALUES = result_column("epsilon")
EPS_INV_VALUES = result_column("epsilon_inverse")
V_STANDARD_VALUES = result_column("V_standard")
VSTAR_DELTA_VALUES = result_column("Vstar_delta")
VSTAR_RHO_VALUES = result_column("Vstar_rho")
SHOT_COEFFICIENT_VALUES = result_column("shot_coefficient")


sort_order = np.argsort(EPS_INV_VALUES)
THETA_VALUES = THETA_VALUES[sort_order]
EPSILON_VALUES = EPSILON_VALUES[sort_order]
EPS_INV_VALUES = EPS_INV_VALUES[sort_order]
V_STANDARD_VALUES = V_STANDARD_VALUES[sort_order]
VSTAR_DELTA_VALUES = VSTAR_DELTA_VALUES[sort_order]
VSTAR_RHO_VALUES = VSTAR_RHO_VALUES[sort_order]
SHOT_COEFFICIENT_VALUES = SHOT_COEFFICIENT_VALUES[sort_order]


def crm_variance(vstar_delta, shot_coefficient, reuse):
    """Equation (6): V_R = V_*(O,Delta) + [V-V_*(O,rho)]/R."""
    return vstar_delta + shot_coefficient / reuse


def lambda_total(
    v_standard,
    vstar_delta,
    shot_coefficient,
    reuse,
    switching_ratio,
):
    """Evaluate Lambda from Eqs. (41) and (42)."""
    v_r = crm_variance(vstar_delta, shot_coefficient, reuse)
    return (
        (1.0 + switching_ratio)
        / (reuse + switching_ratio)
        * v_standard
        / v_r
    )


def numerical_optimal_reuse(
    vstar_delta,
    shot_coefficient,
    switching_ratio,
):
    """Return the continuous optimum of (R+x)V_R."""
    tiny = np.finfo(float).tiny
    return np.sqrt(
        np.maximum(shot_coefficient, 0.0)
        * switching_ratio
        / np.maximum(vstar_delta, tiny)
    )


def lambda_grid_at_fixed_epsilon(
    epsilon,
    reuse_values=R_VALUES,
    switching_values=X_VALUES,
):
    """Return Lambda(R, x) at fixed infidelity."""
    theta = float(theta_from_infidelity(epsilon))
    result = variance_components_at_theta(theta)

    reuse_mesh = reuse_values[:, None]
    switching_mesh = switching_values[None, :]

    lambda_grid = lambda_total(
        result["V_standard"],
        result["Vstar_delta"],
        result["shot_coefficient"],
        reuse_mesh,
        switching_mesh,
    )
    return theta, result, lambda_grid


R_LEFT = R_VALUES[:, None]
LAMBDA_LEFT = lambda_total(
    V_STANDARD_VALUES[None, :],
    VSTAR_DELTA_VALUES[None, :],
    SHOT_COEFFICIENT_VALUES[None, :],
    R_LEFT,
    X_FIXED,
)

R_OPT_LEFT = numerical_optimal_reuse(
    VSTAR_DELTA_VALUES,
    SHOT_COEFFICIENT_VALUES,
    X_FIXED,
)

(
    THETA_FIXED,
    FIXED_RESULT,
    LAMBDA_RIGHT,
) = lambda_grid_at_fixed_epsilon(EPSILON_FIXED)

R_OPT_RIGHT = numerical_optimal_reuse(
    FIXED_RESULT["Vstar_delta"],
    FIXED_RESULT["shot_coefficient"],
    X_VALUES,
)


if SAVE_NUMERICAL_DATA:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    variance_table = np.column_stack([
        THETA_VALUES,
        EPSILON_VALUES,
        EPS_INV_VALUES,
        V_STANDARD_VALUES,
        VSTAR_DELTA_VALUES,
        VSTAR_RHO_VALUES,
        SHOT_COEFFICIENT_VALUES,
        R_OPT_LEFT,
    ])

    np.savetxt(
        DATA_DIR / "coherent_s77_variance_vs_infidelity.csv",
        variance_table,
        delimiter=",",
        header=(
            "theta,epsilon,epsilon_inverse,"
            "V_standard,Vstar_delta,Vstar_rho,"
            "shot_coefficient,R_opt_at_x_fixed"
        ),
        comments="",
    )

    np.savez_compressed(
        DATA_DIR / "coherent_s77_lambda_grids.npz",
        n_qubits=N_QUBITS,
        k_magic=K_MAGIC,
        dimension=DIMENSION,
        theta_values=THETA_VALUES,
        epsilon_values=EPSILON_VALUES,
        epsilon_inverse_values=EPS_INV_VALUES,
        R_values=R_VALUES,
        x_values=X_VALUES,
        x_fixed=X_FIXED,
        epsilon_fixed=EPSILON_FIXED,
        theta_fixed=THETA_FIXED,
        V_standard_values=V_STANDARD_VALUES,
        Vstar_delta_values=VSTAR_DELTA_VALUES,
        Vstar_rho_values=VSTAR_RHO_VALUES,
        shot_coefficient_values=SHOT_COEFFICIENT_VALUES,
        Lambda_left=LAMBDA_LEFT,
        Lambda_right=LAMBDA_RIGHT,
        R_opt_left=R_OPT_LEFT,
        R_opt_right=R_OPT_RIGHT,
    )


tiny = np.finfo(float).tiny

LAMBDA_LEFT_PLOT = np.log10(
    np.clip(LAMBDA_LEFT, tiny, None)
)
LAMBDA_RIGHT_PLOT = np.log10(
    np.clip(LAMBDA_RIGHT, tiny, None)
)

all_log_values = np.concatenate([
    LAMBDA_LEFT_PLOT.ravel(),
    LAMBDA_RIGHT_PLOT.ravel(),
])
all_log_values = all_log_values[np.isfinite(all_log_values)]

p_low = 0.5
p_high = 99.5
gamma = 1.0

vmin_log = np.percentile(all_log_values, p_low)
vmax_log = np.percentile(all_log_values, p_high)

norm = PowerNorm(
    gamma=gamma,
    vmin=vmin_log,
    vmax=vmax_log,
    clip=False,
)

print()
print(f"n = {N_QUBITS}, k = {K_MAGIC}, d = {DIMENSION}")
print(f"Number of Pauli strings = {N_PAULIS}")
print(
    "epsilon range = "
    f"{EPSILON_VALUES.min():.6e} to {EPSILON_VALUES.max():.6e}"
)
print(
    "epsilon^(-1) range = "
    f"{EPS_INV_VALUES.min():.6e} to {EPS_INV_VALUES.max():.6e}"
)
print(
    "Lambda left range = "
    f"{np.nanmin(LAMBDA_LEFT):.6e} to "
    f"{np.nanmax(LAMBDA_LEFT):.6e}"
)
print(
    "Lambda right range = "
    f"{np.nanmin(LAMBDA_RIGHT):.6e} to "
    f"{np.nanmax(LAMBDA_RIGHT):.6e}"
)
print(
    f"fixed epsilon = {FIXED_RESULT['epsilon']:.8e}, "
    f"theta = {THETA_FIXED:.8e}"
)


def add_horizontal_contour_labels(
    axis,
    x_values,
    y_values,
    z_values,
    contour_set,
    contour_levels,
    x_label_position,
):
    manual_positions = []
    ix = np.argmin(np.abs(x_values - x_label_position))

    for level in contour_levels:
        column = z_values[:, ix]
        crossings = []

        for j in range(len(y_values) - 1):
            z0 = column[j]
            z1 = column[j + 1]

            if not (np.isfinite(z0) and np.isfinite(z1)):
                continue

            if (z0 - level) * (z1 - level) <= 0.0:
                if z0 == z1:
                    continue

                y_cross = (
                    y_values[j]
                    + (level - z0)
                    * (y_values[j + 1] - y_values[j])
                    / (z1 - z0)
                )
                crossings.append(y_cross)

        if crossings:
            y_level = min(crossings)
            if y_values[0] <= y_level <= y_values[-1]:
                manual_positions.append(
                    (x_label_position, y_level)
                )

    if manual_positions:
        axis.clabel(
            contour_set,
            manual=manual_positions,
            inline=True,
            inline_spacing=5,
            fontsize=28.6,
            fmt={
                level: (
                    r"$10^{"
                    + str(int(np.log10(level)))
                    + r"}$"
                )
                for level in contour_levels
            },
        )

        for label in contour_set.labelTexts:
            label.set_rotation(0)


l = 10.5 * 9.0 / 9.5

fig, (axL, axR) = plt.subplots(
    1,
    2,
    figsize=(2.0 * l, l * 0.75),
    sharey=True,
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


mesh_left = axL.pcolormesh(
    EPS_INV_VALUES,
    R_VALUES,
    LAMBDA_LEFT_PLOT,
    shading="auto",
    norm=norm,
    cmap="viridis",
    rasterized=True,
)

axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlim(1.0, EPS_INV_MAX)

axL.set_xlabel(r"$\epsilon^{-1}$", labelpad=1)
axL.set_ylabel(r"$R$", labelpad=1)

axL.tick_params(
    axis="both",
    which="major",
    labelsize=31.6,
)

levels = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
left_min = np.nanmin(LAMBDA_LEFT)
left_max = np.nanmax(LAMBDA_LEFT)
left_levels = [
    level
    for level in levels
    if left_min < level < left_max
]

if left_levels:
    contour_left = axL.contour(
        EPS_INV_VALUES,
        R_VALUES,
        LAMBDA_LEFT,
        levels=left_levels,
        colors="black",
        linewidths=1.2,
    )
    add_horizontal_contour_labels(
        axL,
        EPS_INV_VALUES,
        R_VALUES,
        LAMBDA_LEFT,
        contour_left,
        left_levels,
        x_label_position=5e3,
    )

mask_left_opt = (
    np.isfinite(R_OPT_LEFT)
    & (R_OPT_LEFT >= R_VALUES[0])
    & (R_OPT_LEFT <= R_VALUES[-1])
)

axL.plot(
    EPS_INV_VALUES[mask_left_opt],
    R_OPT_LEFT[mask_left_opt],
    linestyle="--",
    linewidth=2.0,
    color="red",
)

axL.text(
    1e2,
7e4,
    r"$R_{\rm opt}$",
    rotation=0,
    ha="center",
    va="top",
    fontsize=28.6,
    color="red",
)

axL.set_xticks([10**i for i in range(0, 5)])


mesh_right = axR.pcolormesh(
    X_VALUES,
    R_VALUES,
    LAMBDA_RIGHT_PLOT,
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
    labelsize=31.6,
)

right_min = np.nanmin(LAMBDA_RIGHT)
right_max = np.nanmax(LAMBDA_RIGHT)
right_levels = [
    level
    for level in levels
    if right_min < level < right_max
]

if right_levels:
    contour_right = axR.contour(
        X_VALUES,
        R_VALUES,
        LAMBDA_RIGHT,
        levels=right_levels,
        colors="black",
        linewidths=1.2,
    )
    add_horizontal_contour_labels(
        axR,
        X_VALUES,
        R_VALUES,
        LAMBDA_RIGHT,
        contour_right,
        right_levels,
        x_label_position=5e4,
    )

mask_right_opt = (
    np.isfinite(R_OPT_RIGHT)
    & (R_OPT_RIGHT >= R_VALUES[0])
    & (R_OPT_RIGHT <= R_VALUES[-1])
)

axR.plot(
    X_VALUES[mask_right_opt],
    R_OPT_RIGHT[mask_right_opt],
    linestyle="--",
    linewidth=2.0,
    color="red",
)

axR.text(
    7e2,
7e4,
    r"$R_{\rm opt}$",
    rotation=0,
    ha="center",
    va="top",
    fontsize=28.6,
    color="red",
)

axR.set_xticks([10**i for i in range(0, 6)])


cbar = fig.colorbar(
    mesh_right,
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

tick_min = -2
tick_max = 4
tick_exponents = np.arange(
    tick_min,
    tick_max + 1,
    dtype=int,
)

if len(tick_exponents) > 7:
    tick_exponents = tick_exponents[::2]

cbar.set_ticks(tick_exponents)
cbar.set_ticklabels([
    r"$10^{" + str(k) + r"}$"
    for k in tick_exponents
])

axL.set_yticks([10**i for i in range(0, 9)])


fig.savefig(
    OUT_DIR / "gain_combined_coherent_s77_numerical.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    OUT_DIR / "gain_combined_coherent_s77_numerical.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.show()
