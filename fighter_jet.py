"""Fighter jet 3D trajectory animation (matplotlib).

Trajectory: high-speed pass -> vertical loop -> aileron roll -> climbing turn

Usage:
    python fighter_jet.py                 # renders fighter_jet.mp4
    python fighter_jet.py --gif           # renders fighter_jet.gif
    python fighter_jet.py --frame 300     # saves a single PNG preview
    python fighter_jet.py --show          # interactive window
"""
import argparse

import matplotlib

import numpy as np

# ----------------------------------------------------------------- parameters
V = 80.0            # forward speed [m/s] (scaled so the whole show fits the axes)
R_LOOP = 75.0       # loop radius [m]  -> ~8.7 g
FPS = 30
DT = 1.0 / FPS
START = np.array([-380.0, 0.0, 25.0])
MODEL_SCALE = 2.8

# (name, duration [s]) -- rates are generated in `commands()`
PHASES = [
    ("HIGH-SPEED PASS", 2.6),
    ("VERTICAL LOOP", 2 * np.pi * R_LOOP / V),
    ("AILERON ROLL", 1.8),
    ("CLIMBING TURN", 6.5),
]
BOUNDS = np.cumsum([d for _, d in PHASES])
T_END = BOUNDS[-1]


# ------------------------------------------------------------------ dynamics
def smooth(x):
    """0->1 smoothstep."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def phase_at(t):
    i = int(np.searchsorted(BOUNDS, t, side="right"))
    i = min(i, len(PHASES) - 1)
    t0 = 0.0 if i == 0 else BOUNDS[i - 1]
    return i, t - t0


def commands(t):
    """Return body rates (p roll, q pitch) [rad/s] and world turn rate [rad/s]."""
    i, tau = phase_at(t)
    p = q = turn = 0.0
    if i == 1:                      # vertical loop: constant pull
        q = V / R_LOOP
    elif i == 2:                    # aileron roll: one full 360 deg roll
        dur = PHASES[2][1]
        p = 2 * np.pi / dur * (6 * smooth(tau / dur) * (1 - smooth(tau / dur)) * 1.0 + 0.0)
        # normalise so the integral is exactly 2*pi (see `ROLL_NORM`)
        p *= ROLL_NORM
    elif i == 3:                    # climbing turn: bank 38 deg, pitch up ~50 deg
        if tau < 1.0:
            p = -np.radians(22) / 1.0 * 6 * smooth(tau) * (1 - smooth(tau)) * ROLL_NORM2
        if 0.6 < tau < 2.6:
            q = np.radians(68) / 2.0 * 6 * smooth((tau - 0.6) / 2.0) * (1 - smooth((tau - 0.6) / 2.0)) * PITCH_NORM
        if tau > 0.6:
            turn = np.radians(20) * smooth((tau - 0.6) / 1.5)
    return p, q, turn


def _normaliser(fn, dur, n=4000):
    xs = np.linspace(0, 1, n)
    return 1.0 / np.trapezoid(fn(xs), xs)


_bump = lambda x: 6 * smooth(x) * (1 - smooth(x))
ROLL_NORM = _normaliser(_bump, 1.0)
ROLL_NORM2 = _normaliser(_bump, 1.0)
PITCH_NORM = _normaliser(_bump, 1.0)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def simulate(substeps=4):
    """Integrate the flight. Body axes: x forward, y left, z up."""
    n = int(T_END * FPS) + 1
    pos = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    names = []
    p_, R = START.copy(), np.eye(3)
    h = DT / substeps
    t = 0.0
    for k in range(n):
        pos[k], rot[k] = p_, R
        names.append(PHASES[phase_at(t)[0]][0])
        for _ in range(substeps):
            p, q, turn = commands(t)
            # pitch up = rotation about body y by -q (nose toward body z)
            R = rot_z(turn * h) @ R @ rot_x(p * h) @ rot_y(-q * h)
            p_ = p_ + R[:, 0] * V * h
            t += h
    return pos, rot, names


def attitude(R):
    """Pitch / bank in degrees from a body->world rotation matrix."""
    fwd, left, up = R[:, 0], R[:, 1], R[:, 2]
    pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    bank = np.degrees(np.arctan2(-left[2], up[2]))
    return pitch, bank


# ----------------------------------------------------------------- jet model
def _mirror(poly):
    return [(x, -y, z) for x, y, z in poly]


def build_model():
    """Return list of (polygon[body frame], facecolor)."""
    body, wing, fin, dark = "#3a4250", "#5b6574", "#2b323d", "#1c2129"
    nose = (18, 0, 0)
    ring = [(2, 0, 2.6), (2, 2.2, 0), (2, 0, -1.8), (2, -2.2, 0)]
    tail = [(-14, 0, 1.6), (-14, 1.6, 0), (-14, 0, -1.2), (-14, -1.6, 0)]
    parts = []
    for i in range(4):                       # forward fuselage
        a, b = ring[i], ring[(i + 1) % 4]
        parts.append(([nose, a, b], body))
    for i in range(4):                       # aft fuselage
        a, b = ring[i], ring[(i + 1) % 4]
        c, d = tail[i], tail[(i + 1) % 4]
        parts.append(([a, b, d, c], body))
    parts.append((tail, dark))               # nozzle plate
    wing_r = [(5, 2, 0), (-11, 2, 0), (-9, 15, 0), (-2, 15, 0)]
    parts += [(wing_r, wing), (_mirror(wing_r), wing)]
    stab_r = [(-9, 1.5, 0), (-15, 1.5, 0), (-15, 7.5, 0), (-12, 7.5, 0)]
    parts += [(stab_r, wing), (_mirror(stab_r), wing)]
    fin_r = [(-6, 1.2, 1.5), (-15, 1.2, 1.5), (-15, 3.2, 9.5), (-11, 3.2, 9.5)]
    parts += [(fin_r, fin), (_mirror(fin_r), fin)]
    canopy = [(11, 0, 1.4), (4, 0.9, 2.2), (4, -0.9, 2.2)]
    parts.append((canopy, "#7fb2d9"))
    return parts


def flame_model(L):
    """Afterburner cone in the body frame; L = flame length."""
    tip = (-14 - L, 0, 0)
    ring = [(-14, 0, 1.1), (-14, 1.1, 0), (-14, 0, -1.1), (-14, -1.1, 0)]
    return [([tip, ring[i], ring[(i + 1) % 4]], "#ff9a2e") for i in range(4)]


def transform(poly, R, p):
    pts = np.asarray(poly, float) * MODEL_SCALE
    return (pts @ R.T + p).tolist()


# ------------------------------------------------------------------ plotting
def make_figure(show=False):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

    fig = plt.figure(figsize=(6.2, 8.4), dpi=110, facecolor="white")
    ax = fig.add_axes([0.0, 0.17, 1.0, 0.64], projection="3d")
    ax.set_xlim(-400, 400)
    ax.set_ylim(-250, 300)
    ax.set_zlim(0, 500)
    ax.set_box_aspect((800, 550, 500))
    ax.set_xlabel("X (m)", labelpad=-4)
    ax.set_ylabel("Y (m)", labelpad=-4)
    ax.set_zlabel("Z (m)", labelpad=-4)
    ax.tick_params(labelsize=8, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d0d0d0")
        axis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    fig.text(0.5, 0.955, "FIGHTER JET", ha="center", va="center",
             fontsize=26, fontweight="bold", color="#10131c")
    fig.text(0.5, 0.915, "Supersonic, highly manoeuvrable", ha="center",
             fontsize=11, fontweight="bold", color="#2a2f3a")
    fig.text(0.5, 0.88, "Trajectory: high-speed pass  ->  vertical loop  ->  "
             "aileron roll  ->  climb", ha="center", fontsize=8.5, color="#444")
    return plt, fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--frame", type=int, default=None, help="save one PNG frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pos, rot, names = simulate()
    n = len(pos)
    model = build_model()

    plt, fig, ax = make_figure(args.show)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    trail, = ax.plot([], [], [], color="#2b3040", lw=0.9)
    shadow, = ax.plot([], [], [], color="#9aa0aa", lw=0.6, ls=":")
    state = {"jet": None, "flame": None}
    t_phase = fig.text(0.5, 0.135, "", ha="center", fontsize=17,
                       fontweight="bold", color="#3b4a63")
    t_hud = fig.text(0.5, 0.100, "", ha="center", fontsize=10,
                     family="monospace", color="#222")
    fig.text(0.5, 0.062, "top speed above Mach 2", ha="center", fontsize=10, color="#444")
    fig.text(0.5, 0.032, "turns at up to 9 g", ha="center", fontsize=10, color="#444")
    fig.text(0.5, 0.008, "afterburning turbofan engines", ha="center", fontsize=10, color="#444")

    def draw(k):
        k = min(k, n - 1)
        for key in ("jet", "flame"):
            if state[key] is not None:
                state[key].remove()
        p, R = pos[k], rot[k]
        polys = [transform(poly, R, p) for poly, _ in model]
        state["jet"] = Poly3DCollection(
            polys, facecolors=[c for _, c in model], edgecolors="#111", linewidths=0.3)
        ax.add_collection3d(state["jet"])
        flicker = 8 + 5 * np.sin(k * 1.7) ** 2
        fl = flame_model(flicker)
        state["flame"] = Poly3DCollection(
            [transform(poly, R, p) for poly, _ in fl],
            facecolors=[c for _, c in fl], edgecolors="none", alpha=0.85)
        ax.add_collection3d(state["flame"])

        tr = pos[: k + 1]
        trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
        shadow.set_data_3d(tr[:, 0], tr[:, 1], np.zeros(len(tr)))
        pitch, bank = attitude(R)
        t_phase.set_text(names[k])
        t_hud.set_text(f"ALT {p[2]:5.0f} m    PITCH {pitch:+4.0f}°    BANK {bank:+4.0f}°")
        # slow camera drift
        ax.view_init(elev=20, azim=-60 + 12 * k / n)

    if args.frame is not None:
        draw(args.frame)
        out = args.out or "frame.png"
        fig.savefig(out)
        print("saved", out)
        return

    if args.show:
        from matplotlib.animation import FuncAnimation
        _anim = FuncAnimation(fig, draw, frames=n, interval=1000 / FPS, repeat=True)
        plt.show()
        return

    from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
    out = args.out or ("fighter_jet.gif" if args.gif else "fighter_jet.mp4")
    writer = PillowWriter(fps=FPS // 2) if args.gif else FFMpegWriter(fps=FPS, bitrate=3500)
    anim = FuncAnimation(fig, draw, frames=n)
    anim.save(out, writer=writer)
    print("saved", out, f"({n} frames)")


if __name__ == "__main__":
    main()
