import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from colors import CONTROL, CONTROL_DARK, TREATMENT

def parse_bench(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("Length"):
                continue
            parts = [p.strip() for p in line.split(",")]
            length = int(parts[0].split()[-1])
            samples = np.array([int(x) for x in parts[1:]])
            data[length] = samples
    return data

d128 = parse_bench("128bit.csv")  # control
d256 = parse_bench("256bit.csv")  # treatment

lengths = sorted(d128.keys())
alpha = 0.05
ncols = 3
nrows = (len(lengths) + ncols - 1) // ncols

# ============================================================
# 1. Summary table
# ============================================================
header = f"{'Length':>10} {'Treatment':>10} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'Mean':>7} {'Std':>6} {'IQR':>6}"
print(header)
print("-" * len(header))

for L in lengths:
    for label, samples in [("128-bit", d128[L]), ("256-bit", d256[L])]:
        p25, p50, p75, p90 = np.percentile(samples, [25, 50, 75, 90])
        print(f"{L:>10} {label:>10} {p25:>6.1f} {p50:>6.1f} {p75:>6.1f} {p90:>6.1f} {np.mean(samples):>7.1f} {np.std(samples, ddof=1):>6.1f} {p75-p25:>6.1f}")
    print()

# ============================================================
# 2. Simple overview (absolute times + difference)
# ============================================================
means_128, means_256 = [], []
ci_lo_128, ci_hi_128 = [], []
ci_lo_256, ci_hi_256 = [], []

for L in lengths:
    a, b = d128[L], d256[L]
    m_a, m_b = np.mean(a), np.mean(b)
    ci_a = stats.t.interval(1 - alpha, len(a) - 1, loc=m_a, scale=stats.sem(a))
    ci_b = stats.t.interval(1 - alpha, len(b) - 1, loc=m_b, scale=stats.sem(b))
    means_128.append(m_a)
    means_256.append(m_b)
    ci_lo_128.append(ci_a[0])
    ci_hi_128.append(ci_a[1])
    ci_lo_256.append(ci_b[0])
    ci_hi_256.append(ci_b[1])

means_128 = np.array(means_128)
means_256 = np.array(means_256)
ci_lo_128 = np.array(ci_lo_128)
ci_hi_128 = np.array(ci_hi_128)
ci_lo_256 = np.array(ci_lo_256)
ci_hi_256 = np.array(ci_hi_256)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
x = np.arange(len(lengths))
xlabels = [str(L) for L in lengths]

ax1.errorbar(x - 0.15, means_128, yerr=[means_128 - ci_lo_128, ci_hi_128 - means_128],
             fmt='o-', capsize=4, label='128-bit loads', color=CONTROL)
ax1.errorbar(x + 0.15, means_256, yerr=[means_256 - ci_lo_256, ci_hi_256 - means_256],
             fmt='s-', capsize=4, label='256-bit loads', color=TREATMENT)
ax1.set_ylabel("Time (units from benchmark)")
ax1.set_title("AVX/FMA f32 Benchmark: 128-bit vs 256-bit loads")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, rotation=45, ha='right')

diffs, diff_ci_lo, diff_ci_hi = [], [], []
for L in lengths:
    a, b = d128[L], d256[L]
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
             fmt='D-', capsize=4, color='#2ca02c')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel("Difference (256-bit minus 128-bit)")
ax2.set_xlabel("Array Length")
ax2.set_title("Mean Difference with 95% CI (positive = 256-bit slower)")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("bench_simple.png", dpi=150)
print("Saved bench_simple.png")

# ============================================================
# 3. A/B forest plot
# ============================================================
ab_labels = []
pct_diffs, pct_ci_lo, pct_ci_hi, p_values = [], [], [], []

for L in lengths:
    ctrl, treat = d128[L], d256[L]
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
    ab_labels.append(f"Length {L:,}")

pct_diffs = np.array(pct_diffs)
pct_ci_lo = np.array(pct_ci_lo)
pct_ci_hi = np.array(pct_ci_hi)
p_values = np.array(p_values)
significant = p_values < alpha

fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(lengths))[::-1]

for i in range(len(lengths)):
    color = '#e74c3c' if significant[i] and pct_diffs[i] > 0 else \
            '#27ae60' if significant[i] and pct_diffs[i] < 0 else \
            '#7f8c8d'
    ax.plot([pct_ci_lo[i], pct_ci_hi[i]], [y[i], y[i]], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(pct_diffs[i], y[i], 'o', color=color, markersize=8, zorder=5)
    sig_str = " *" if significant[i] else ""
    ax.text(max(pct_ci_hi[i], 0) + 0.8, y[i],
            f"{pct_diffs[i]:+.1f}%{sig_str}",
            va='center', fontsize=9, color=color, fontweight='bold')

ax.axvline(0, color='black', linewidth=1, linestyle='-', zorder=0)
ax.axvspan(-0.5, 0.5, color='#ecf0f1', alpha=0.5, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(ab_labels, fontsize=10)
ax.set_xlabel("% Change from Control (128-bit loads)", fontsize=11)
ax.set_title("A/B Comparison: 256-bit vs 128-bit loads (AVX/FMA f32)\n"
             "Control = 128-bit  |  Treatment = 256-bit  |  Lower is faster",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([0], [0], color='#e74c3c', marker='o', linewidth=2.5, label='Significantly slower (p<0.05)'),
    Line2D([0], [0], color='#27ae60', marker='o', linewidth=2.5, label='Significantly faster (p<0.05)'),
    Line2D([0], [0], color='#7f8c8d', marker='o', linewidth=2.5, label='No significant difference'),
], loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("bench_comparison.png", dpi=150)
print("Saved bench_comparison.png")

print(f"\n{'Metric':<16} {'Control':>8} {'Treatment':>10} {'% Diff':>8} {'95% CI':>20} {'p':>8} {'Sig':>4}")
print("-" * 80)
for i, L in enumerate(lengths):
    mc = np.mean(d128[L])
    mt = np.mean(d256[L])
    print(f"Length {L:<8,} {mc:>8.1f} {mt:>10.1f} {pct_diffs[i]:>+7.1f}% [{pct_ci_lo[i]:>+6.1f}%, {pct_ci_hi[i]:>+5.1f}%] {p_values[i]:>8.4f} {'*' if significant[i] else ''}")

# ============================================================
# 4. Time series scatter
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.4), sharex=True)
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    n128 = len(d128[L])
    n256 = len(d256[L])

    ax.scatter(np.arange(n128), d128[L], s=12, alpha=0.5, color=CONTROL, label='128-bit', zorder=2)
    ax.scatter(np.arange(n256), d256[L], s=12, alpha=0.5, color=TREATMENT, label='256-bit', zorder=2)

    w = 10
    if n128 >= w:
        rm128 = np.convolve(d128[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n128), rm128, color=CONTROL_DARK, linewidth=1.5, zorder=3)
    if n256 >= w:
        rm256 = np.convolve(d256[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n256), rm256, color=TREATMENT, linewidth=1.5, zorder=3)

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

fig.suptitle("Sample-by-sample time series: 128-bit vs 256-bit loads\n"
             "(dots = raw samples, line = rolling mean window=10)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_timeseries.png", dpi=150, bbox_inches='tight')
print("\nSaved bench_timeseries.png")

# ============================================================
# 5. ECDF
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [("128-bit", d128[L], CONTROL), ("256-bit", d256[L], TREATMENT)]:
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

fig.suptitle("Empirical CDF: 128-bit vs 256-bit loads",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_ecdf.png", dpi=150, bbox_inches='tight')
print("Saved bench_ecdf.png")

# ============================================================
# 6. KDE density
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [("128-bit", d128[L], CONTROL), ("256-bit", d256[L], TREATMENT)]:
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

fig.suptitle("KDE Density: 128-bit vs 256-bit loads",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_kde.png", dpi=150, bbox_inches='tight')
print("Saved bench_kde.png")
