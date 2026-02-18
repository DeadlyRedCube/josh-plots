import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from colors import DATA_COLORS, deemphasize_color, darken_color
from matplotlib.lines import Line2D
from scipy import stats

class Bench:
    def __init__(self, path):
        self.data = {}
        self.label = ""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if not line.startswith("Length"):
                    assert(not self.label)
                    self.label = line
                    continue

                parts = [p.strip() for p in line.split(",")]
                length = int(parts[0].split()[-1])
                samples = np.array([int(x) for x in parts[1:]])
                self.data[length] = samples

parser = argparse.ArgumentParser(prog="bench", description="Benchmark results comparison")
parser.add_argument("control", type=str, help="Path of the control .csv file")
parser.add_argument("treatment", nargs="+", help="Path(s) to the treatment file(s)")
parser.add_argument("-o", "--output-dir", default=None, type=str, help="Directory to output images to")
parser.add_argument("-p", "--output-prefix", default="bench", type=str, help="Prefix for output image filenames")
args = parser.parse_args()

if args.output_dir:
    if args.output_dir.endswith("/") or args.output_dir.endswith("\\"):
        args.output_dir = args.output_dir[:-1]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(args.output_dir):
        if not os.path.isdir(args.output_dir):
          print(f"Error: '{args.output_dir}' is not a directory.")
          sys.exit(1)

control = Bench(args.control)
treatments = [Bench(x) for x in args.treatment]

if len(treatments) > len(DATA_COLORS) - 1:
    print("Too many treatments! We're going to need more colors")
    sys.exit(1)

maxLabelLength = max([len(x.label) for x in [control, *treatments]])

lengths = sorted(control.data.keys())
alpha = 0.05
ncols = 3
nrows = (len(lengths) + ncols - 1) // ncols


def vs_label():
    return f"'{control.label}' vs {f"'{treatments[0].label}'" if len(treatments) == 1 else "multiple"}"

def make_output_path(suffixWithExt):
    if args.output_dir:
        return f"{args.output_dir}/{args.output_prefix}_{suffixWithExt}"
    else:
        return f"{args.output_prefix}_{suffixWithExt}"

def save_fig(plt, name, dpi=300):
    path = make_output_path(name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved {path}")

# ============================================================
# 1. Summary table
# ============================================================
header = f"{'Length':>10} { 'Treatment':>{maxLabelLength}} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'Mean':>7} {'Std':>6} {'IQR':>6}"
print(header)
print("-" * len(header))

for L in lengths:
    for label, samples in [
        (control.label, control.data[L]), 
        *((t.label, t.data[L]) for t in treatments)
    ]:
        p25, p50, p75, p90 = np.percentile(samples, [25, 50, 75, 90])
        print(f"{L:>10} {label:>{maxLabelLength}} {p25:>6.1f} {p50:>6.1f} {p75:>6.1f} {p90:>6.1f} {np.mean(samples):>7.1f} {np.std(samples, ddof=1):>6.1f} {p75-p25:>6.1f}")
    print()

# ============================================================
# 2. Simple overview (absolute times + difference)
# ============================================================
def make_simple(bench):
    means = []
    ci_lo = []
    ci_hi = []
    for L in lengths:
        data = bench.data[L]
        m = np.mean(data)
        ci = stats.t.interval(1 - alpha, len(data) - 1, loc=m, scale=stats.sem(data))
        means.append(m)
        ci_lo.append(ci[0])
        ci_hi.append(ci[1])
    return (np.array(means), np.array(ci_lo), np.array(ci_hi))

means_ctrl, ci_lo_ctrl, ci_hi_ctrl = make_simple(control)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
x = np.arange(len(lengths))
xlabels = [str(L) for L in lengths]

ax1.errorbar(x - 0.15, means_ctrl, yerr=[means_ctrl - ci_lo_ctrl, ci_hi_ctrl - means_ctrl],
             fmt='o-', capsize=4, label=control.label, color=DATA_COLORS[0])

for (treatment, color) in ((treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
  means_tmnt, ci_lo_tmnt, ci_hi_tmnt = make_simple(treatment)
  ax1.errorbar(x + 0.15, means_tmnt, yerr=[means_tmnt - ci_lo_tmnt, ci_hi_tmnt - means_tmnt],
              fmt='s-', capsize=4, label=treatment.label, color=color)
ax1.set_ylabel("Time (units from benchmark)")
ax1.set_title(f"Benchmark: {vs_label()}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, rotation=45, ha='right')

for (treatment, color) in ((treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
    diffs, diff_ci_lo, diff_ci_hi = [], [], []
    for L in lengths:
        a, b = control.data[L], treatment.data[L]
        d = np.mean(b) - np.mean(a)
        se = np.sqrt(stats.sem(a)**2 + stats.sem(b)**2)
        df_w = (stats.sem(a)**2 + stats.sem(b)**2)**2 / (
            stats.sem(a)**4 / (len(a)-1) + stats.sem(b)**4 / (len(b)-1))
        t_crit = stats.t.ppf(1 - alpha/2, df_w)
        diffs.append(d)
        diff_ci_lo.append(d - t_crit * se)
        diff_ci_hi.append(d + t_crit * se)

    diffs = np.array(diffs)
    diff_ci_lo = np.array(diff_ci_lo)
    diff_ci_hi = np.array(diff_ci_hi)

    ax2.errorbar(x, diffs, yerr=[diffs - diff_ci_lo, diff_ci_hi - diffs],
                 fmt='D-', capsize=4, color=color)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel(f"Difference ({"treatments" if len(treatments) > 1 else f"'{treatments[0].label}'"} minus '{control.label}'")
ax2.set_xlabel("Array Length")
ax2.set_title(f"Mean Difference with 95% CI (positive = slower than control)")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, rotation=45, ha='right')

plt.tight_layout()
save_fig(plt, "simple.png")

# ============================================================
# 3. A/B forest plot
# ============================================================

ab_labels, pct_diffs, pct_ci_lo, pct_ci_hi, p_values, base_colors = [], [], [], [], [], []
for L in lengths:
    for (treatmentIndex, treatment, base_color) in ((i, treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
        ctrl, treat = control.data[L], treatment.data[L]
        mc, mt = np.mean(ctrl), np.mean(treat)
        se = np.sqrt(stats.sem(ctrl)**2 + stats.sem(treat)**2)
        df_w = (stats.sem(ctrl)**2 + stats.sem(treat)**2)**2 / (
            stats.sem(ctrl)**4 / (len(ctrl)-1) + stats.sem(treat)**4 / (len(treat)-1))
        t_crit = stats.t.ppf(1 - alpha/2, df_w)
        diff_lo = (mt - mc) - t_crit * se
        diff_hi = (mt - mc) + t_crit * se

        pct_diffs.append((mt - mc) / mc * 100)
        pct_ci_lo.append(diff_lo / mc * 100)
        pct_ci_hi.append(diff_hi / mc * 100)
        _, p = stats.ttest_ind(ctrl, treat, equal_var=False)
        p_values.append(p)
        base_colors.append(base_color)
        ab_labels.append(f"Length {L}: '{treatment.label}'" if treatmentIndex == 0 else f"'{treatment.label}'")

pct_diffs = np.array(pct_diffs)
pct_ci_lo = np.array(pct_ci_lo)
pct_ci_hi = np.array(pct_ci_hi)
p_values = np.array(p_values)
significant = p_values < alpha

fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(ab_labels))[::-1]

for i in range(len(ab_labels)):
    color, marker = (darken_color(base_colors[i]), 'v') if significant[i] and pct_diffs[i] > 0 else \
            (base_colors[i], '^') if significant[i] and pct_diffs[i] < 0 else \
            ('#7f8c8d', 'o')
    ax.plot([pct_ci_lo[i], pct_ci_hi[i]], [y[i], y[i]], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(pct_diffs[i], y[i], marker=marker, color=color, markersize=8, zorder=5)
    sig_str = " *" if significant[i] else ""
    ax.text(max(pct_ci_hi[i], 0) + 0.8, y[i],
            f"{pct_diffs[i]:+.1f}%{sig_str}",
            va='center', fontsize=9, color=color, fontweight='bold')

ax.axvline(0, color='black', linewidth=1, linestyle='-', zorder=0)
ax.axvspan(-0.5, 0.5, color='#ecf0f1', alpha=0.5, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(ab_labels, fontsize=10)
ax.set_ylabel("Name (Length)")
ax.set_xlabel(f"% Change from Control ({control.label})", fontsize=11)
ax.set_title(f"A/B Comparison\n"
             "Lower is faster",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([0], [0], color=darken_color(DATA_COLORS[1]), marker='v', linewidth=2.5, label='Significantly slower (p<0.05)'),
    Line2D([0], [0], color=DATA_COLORS[1], marker='^', linewidth=2.5, label='Significantly faster (p<0.05)'),
    Line2D([0], [0], color=DATA_COLORS[0], marker='o', linewidth=2.5, label='No significant difference'),
], loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
save_fig(plt, "comparision.png")

print(f"\n{'Metric':<16} {'Control':>8} {'Treatment':>10} {'% Diff':>8} {'95% CI':>20} {'p':>8} {'Sig':>4}")
print("-" * 80)
for i, L in enumerate(lengths):
    mc = np.mean(control.data[L])
    mt = np.mean(treatments[0].data[L])
    print(f"Length {L:<8,} {mc:>8.1f} {mt:>10.1f} {pct_diffs[i]:>+7.1f}% [{pct_ci_lo[i]:>+6.1f}%, {pct_ci_hi[i]:>+5.1f}%] {p_values[i]:>8.4f} {'*' if significant[i] else ''}")

# ============================================================
# 4. Time series scatter
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.4), sharex=True)
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    n_ctrl= len(control.data[L])

    ax.scatter(np.arange(n_ctrl), control.data[L], s=12, alpha=0.5, color=DATA_COLORS[0], label=control.label, zorder=2)

    for (treatment, color) in ((treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
        nt = len(treatment.data[L])
        ax.scatter(np.arange(nt), treatment.data[L], s=12, alpha=0.5, color=color, label=treatment.label, zorder=2)

    w = 10
    if n_ctrl >= w:
        rmc = np.convolve(control.data[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n_ctrl), rmc, color=deemphasize_color(DATA_COLORS[0]), linewidth=1.5, zorder=3)

    for (treatment, color) in ((treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
        nt = len(treatment.data[L])
        if nt >= w:
            rmt = np.convolve(treatment.data[L], np.ones(w)/w, mode='valid')
            ax.plot(np.arange(w-1, nt), rmt, color=deemphasize_color(color), linewidth=1.5, zorder=3)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

for ax in axes[-ncols:]:
    ax.set_xlabel("Sample #", fontsize=9)
for ax in axes[::ncols]:
    ax.set_ylabel("Time", fontsize=9)
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Sample-by-sample time series\n"
             "(dots = raw samples, line = rolling mean window=10)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "timeseries.png")

# ============================================================
# 5. ECDF
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [
        (control.label, control.data[L], DATA_COLORS[0]), 
        *((treatments[i].label, treatments[i].data[L], DATA_COLORS[i + 1]) for i in range(len(treatments)))
    ]:
        xs = np.sort(samples)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where='post', color=color, linewidth=1.5, label=label)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    if i == 0:
        ax.legend(fontsize=7, loc='lower right')

for ax in axes[-ncols:]:
    ax.set_xlabel("Time", fontsize=9)
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)


fig.suptitle(f"Empirical CDF: {vs_label()}",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "ecdf.png")

# ============================================================
# 6. KDE density
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [
        (control.label, control.data[L], DATA_COLORS[0]), 
        *((treatments[i].label, treatments[i].data[L], DATA_COLORS[i + 1]) for i in range(len(treatments)))
    ]:
        kde = stats.gaussian_kde(samples)
        lo, hi = samples.min() - 5, samples.max() + 5
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, kde(xs), color=color, linewidth=1.5, label=label)
        ax.fill_between(xs, kde(xs), alpha=0.15, color=color)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.set_ylabel("Density", fontsize=8)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

for ax in axes[-ncols:]:
    ax.set_xlabel("Time", fontsize=9)
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"KDE Density: {vs_label()}",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "kde.png")
