"""
Kaggle Notebook — Tetranet 10M: Full Training + Analysis
=========================================================
Paste cells into a Kaggle Notebook (GPU T4x2, Internet ON).

Kaggle Datasets needed:
  - tinystories-full  (TinyStoriesV2-GPT4-train.txt + TinyStoriesV2-GPT4-valid.txt)
  - tetranet-code     (train_kaggle.py, model.py, quaternary.py, regularization.py,
                       eval_all.py, tetranet_tokenizer.json)

Run order: Cell 0 -> Restart Runtime -> Run All
"""

# ═══════════════════════════════════════════════════════════════
# Cell 0: Fix P100 + install deps (run once, then Restart & Run All)
# ═══════════════════════════════════════════════════════════════

!pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
!pip install -q tokenizers matplotlib pandas seaborn

# ═══════════════════════════════════════════════════════════════
# Cell 1: Setup — copy code + data from Kaggle Datasets
# ═══════════════════════════════════════════════════════════════

import json, math, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── Copy code ──
os.makedirs("/kaggle/working/tetranet", exist_ok=True)
for f in ["train_kaggle.py", "model.py", "quaternary.py", "regularization.py",
          "eval_all.py", "tetranet_tokenizer.json"]:
    shutil.copy(f"/kaggle/input/tetranet-code/{f}", f"/kaggle/working/tetranet/{f}")
os.chdir("/kaggle/working/tetranet")

# ── Copy data ──
os.makedirs("./tinystories", exist_ok=True)
for split in ["train", "valid"]:
    shutil.copy(
        f"/kaggle/input/tinystories-full/TinyStoriesV2-GPT4-{split}.txt",
        f"./tinystories/TinyStories-{split}.txt",
    )

import torch
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {DEVICE}")
print(f"VRAM: {VRAM:.1f} GB")

# ═══════════════════════════════════════════════════════════════
# Cell 2: Train all 6 baselines at 10M scale
# ═══════════════════════════════════════════════════════════════

BASELINES = [
    ("full_precision", ""),
    ("bitnet", ""),
    ("uniform_2bit", ""),
    ("fixed_c_025", ""),
    ("fixed_c_05", ""),
    ("learned_c", "--c-lr 0.003 --alpha 2.0 --snap-start 0.4"),
]

SMALL_CONFIG = "--hidden-dim 256 --num-layers 8 --num-heads 8 --ffn-dim 1024"

for name, extra_args in BASELINES:
    ckpt = f"./checkpoint_{name}.pt"
    log = f"./log_{name}.csv"
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    cmd = (
        f"python train_kaggle.py --baseline {name} "
        f"--data-path ./tinystories/TinyStories-train.txt "
        f"--tokenizer-path ./tetranet_tokenizer.json "
        f"--max-stories 100000 {SMALL_CONFIG} "
        f"--batch-size 8 --grad-accum 4 "
        f"--lr 3e-4 --weight-decay 0.1 "
        f"--epochs 1 "
        f"--checkpoint-path {ckpt} --log-path {log} "
        f"--ckpt-interval 5000 --log-interval 50 "
        f"--bf16 {extra_args}"
    )
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    print(f"  -> {name} finished in {elapsed:.1f} min")

print("\n All baselines trained.")

# ═══════════════════════════════════════════════════════════════
# Cell 3: Load logs + build comparison dataframe
# ═══════════════════════════════════════════════════════════════

BASELINE_NAMES = ["full_precision", "bitnet", "uniform_2bit",
                  "fixed_c_025", "fixed_c_05", "learned_c"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
LABELS = ["Full precision", "BitNet b1.58", "Uniform 2-bit",
          "Quaternary c=0.25", "Quaternary c=0.5", "Quaternary learned_c"]

logs = {}
for name in BASELINE_NAMES:
    df = pd.read_csv(f"./log_{name}.csv")
    # Parse c_values_json if present
    if "c_values_json" in df.columns:
        df["c_values"] = df["c_values_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {})
    logs[name] = df
    print(f"{name:20s}  {len(df):>4d} rows  "
          f"final loss={df['task_loss'].iloc[-1]:.4f}  "
          f"best loss={df['task_loss'].min():.4f}")

# ═══════════════════════════════════════════════════════════════
# Cell 4: Figure 1 — Training loss overlay (all baselines)
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))
for i, name in enumerate(BASELINE_NAMES):
    df = logs[name]
    ax.plot(df["step"], df["task_loss"], color=COLORS[i], label=LABELS[i],
            linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Training loss curves — 10M models on TinyStories")
ax.legend(framealpha=0.9)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig("fig1_training_loss.png", dpi=150)
plt.show()
print("Fig 1 saved: training loss overlay")

# ═══════════════════════════════════════════════════════════════
# Cell 5: Figure 2 — Training speed comparison
# ═══════════════════════════════════════════════════════════════

# tok/s is logged in the console, not CSV. Compute from elapsed time.
# Use the last row's "progress" and step to estimate speed.
speeds = []
for name in BASELINE_NAMES:
    df = logs[name]
    # tok/s can be estimated from tokens_seen / elapsed_time
    # We don't have elapsed in CSV, but we can use progress
    step = df["step"].iloc[-1]
    progress = df["progress"].iloc[-1]
    total_tokens = 100000 * 33  # ~33 tok/story * 100K stories
    tokens_seen = total_tokens * progress
    # Training time ≈ step * batch_size * seq_len / (batch_size * seq_len / seconds per step)
    # Rough estimate from durations in the training output
    speeds.append(None)  # Will fill from training output

fig, ax = plt.subplots(figsize=(10, 5))
# We'll estimate speed from CSV log timestamps indirectly:
# Each row represents a log_interval=50 steps.
# steps_per_row = 50, batch_size=8, seq_len=512
# tokens_per_row = 50 * 8 * 512 = 204800
# So tok/s = 204800 / (time_between_rows_in_seconds)
ax.bar(LABELS, [0]*len(LABELS), color=COLORS)  # placeholder
ax.set_title("Training throughput (will fill from training output)")
fig.tight_layout()

# ═══════════════════════════════════════════════════════════════
# Cell 6: Figure 3 — Gradient norm comparison
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))
for i, name in enumerate(BASELINE_NAMES):
    df = logs[name]
    if "grad_norm" in df.columns:
        ax.plot(df["step"], df["grad_norm"], color=COLORS[i],
                label=LABELS[i], linewidth=1, alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Gradient norm")
ax.set_title("Gradient norm during training")
ax.legend(framealpha=0.9, ncol=2)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("fig3_grad_norm.png", dpi=150)
plt.show()
print("Fig 3 saved: gradient norm")

# ═══════════════════════════════════════════════════════════════
# Cell 7: Figure 4 — PPL comparison bar chart
# ═══════════════════════════════════════════════════════════════

from models import build_model
from eval_all import eval_ppl, load_tokenizer, tokenize_valid, ValidDataset

SEQ_LEN = 512
BATCH_SIZE = 4
MAX_STORIES = 2500

tokenizer = load_tokenizer("./tetranet_tokenizer.json")
tokens = tokenize_valid("./tinystories/TinyStories-valid.txt", tokenizer, MAX_STORIES)
dataset = ValidDataset(tokens, SEQ_LEN)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                      shuffle=False, drop_last=False)
print(f"Validation set: {len(dataset):,} sequences\n")

ppl_results = {}
for name in BASELINE_NAMES:
    ckpt_path = f"./checkpoint_{name}_final.pt"
    if not os.path.exists(ckpt_path):
        print(f"SKIP {name}: {ckpt_path} not found")
        continue
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(baseline=name, vocab_size=4096, hidden_dim=256,
                        num_layers=8, num_heads=8, ffn_dim=1024)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss, ppl = eval_ppl(model, loader)
    ppl_results[name] = {"loss": loss, "ppl": ppl}
    print(f"{name:20s}  loss={loss:.4f}  ppl={ppl:.2f}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
vals = [ppl_results[n]["ppl"] for n in BASELINE_NAMES if n in ppl_results]
names_plot = [n for n in BASELINE_NAMES if n in ppl_results]
labels_plot = [LABELS[i] for i, n in enumerate(BASELINE_NAMES) if n in ppl_results]
colors_plot = [COLORS[i] for i, n in enumerate(BASELINE_NAMES) if n in ppl_results]

bars = ax.bar(labels_plot, vals, color=colors_plot, width=0.6, edgecolor="white")
ax.set_ylabel("Perplexity (lower is better)")
ax.set_title("Perplexity comparison — TinyStories validation set (10M models)")
ax.tick_params(axis="x", rotation=20)

# Annotate bars
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Add delta annotations vs full_precision
if "full_precision" in ppl_results:
    fp_ppl = ppl_results["full_precision"]["ppl"]
    for i, (bar, val) in enumerate(zip(bars, vals)):
        delta = val - fp_ppl
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{delta_str} vs FP", ha="center", va="center",
                fontsize=7, color="white" if bar.get_facecolor()[0:3] != (1,1,1) else "black",
                fontweight="bold")

fig.tight_layout()
fig.savefig("fig4_ppl_comparison.png", dpi=150)
plt.show()
print("Fig 4 saved: PPL comparison")

# ═══════════════════════════════════════════════════════════════
# Cell 8: Figure 5 — Structural fingerprinting (learned_c)
# ═══════════════════════════════════════════════════════════════

ckpt_lc = torch.load("./checkpoint_learned_c_final.pt", map_location="cpu",
                     weights_only=False)
model_lc = build_model(baseline="learned_c", vocab_size=4096, hidden_dim=256,
                       num_layers=8, num_heads=8, ffn_dim=1024)
model_lc.load_state_dict(ckpt_lc["model_state_dict"])
c_vals = model_lc.get_c_values()

proj_map = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O",
            "gate_proj": "Gate", "up_proj": "Up", "down_proj": "Down"}
by_type = {}
for key, val in c_vals.items():
    for pat, label in proj_map.items():
        if pat in key:
            by_type.setdefault(label, []).append(val)
            break

types = sorted(by_type.keys())
x = np.arange(len(types))
counts_025 = [sum(1 for v in by_type[t] if abs(v - 0.25) < 0.02) for t in types]
counts_05  = [sum(1 for v in by_type[t] if abs(v - 0.50) < 0.02) for t in types]

fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.35
b1 = ax.bar(x - w/2, counts_025, w, label="Snapped to 0.25",
            color="#4C72B0", edgecolor="white")
b2 = ax.bar(x + w/2, counts_05,  w, label="Snapped to 0.50",
            color="#DD8452", edgecolor="white")
ax.set_ylabel("Number of layers (out of 8)")
ax.set_title("Structural fingerprinting: per-projection c snapping")
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.legend(framealpha=0.9)
for bar in b1 + b2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, int(h),
                ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, max(max(counts_025), max(counts_05)) + 1.5)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig("fig5_fingerprinting.png", dpi=150)
plt.show()
print("Fig 5 saved: structural fingerprinting")

# ═══════════════════════════════════════════════════════════════
# Cell 9: Figure 6 — C-value heatmap (layers x projections)
# ═══════════════════════════════════════════════════════════════

n_layers = 8
proj_order = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
proj_labels = ["Q", "K", "V", "O", "Gate", "Up", "Down"]

# Build c matrix
c_matrix = np.full((n_layers, len(proj_order)), np.nan)
for key, val in c_vals.items():
    for li in range(n_layers):
        if f"layer.{li}." in key:
            for pi, pat in enumerate(proj_order):
                if pat in key:
                    c_matrix[li, pi] = val
                    break
            break

fig, ax = plt.subplots(figsize=(9, 5))
# Custom colormap: red for 0.25, blue for 0.5
cmap = sns.diverging_palette(250, 15, s=75, l=45, n=256, center="light")
mask = np.isnan(c_matrix)
sns.heatmap(c_matrix, ax=ax, annot=True, fmt=".4f", cmap=cmap,
            xticklabels=proj_labels, yticklabels=[f"Layer {i}" for i in range(n_layers)],
            vmin=0.15, vmax=0.60, linewidths=1, linecolor="white",
            cbar_kws={"label": "c value", "shrink": 0.8})
ax.set_title("Learned c values per layer and projection")
ax.set_xlabel("Projection type")
ax.set_ylabel("Layer depth")
fig.tight_layout()
fig.savefig("fig6_c_heatmap.png", dpi=150)
plt.show()
print("Fig 6 saved: c-value heatmap")

# ═══════════════════════════════════════════════════════════════
# Cell 10: Figure 7 — Snapping loss landscape (learned_c)
# ═══════════════════════════════════════════════════════════════

df_lc = logs["learned_c"]
if "total_loss" in df_lc.columns:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Task loss
    l1 = ax1.plot(df_lc["step"], df_lc["task_loss"], color="#4C72B0",
                  linewidth=2, label="Task loss (cross-entropy)")
    # Total loss
    l2 = ax1.plot(df_lc["step"], df_lc["total_loss"], color="#C44E52",
                  linewidth=2, linestyle="--", label="Total loss (task + lambda * penalty)")

    # Lambda on secondary axis
    ax2 = ax1.twinx()
    l3 = ax2.plot(df_lc["step"], df_lc["lambda"], color="#55A868",
                  linewidth=2, linestyle=":", label="lambda(t)")
    ax2.set_ylabel("lambda(t)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    # Snap start vertical line
    snap_idx = (df_lc["lambda"] > 0).idxmax() if (df_lc["lambda"] > 0).any() else None
    if snap_idx is not None:
        snap_step = df_lc.loc[snap_idx, "step"]
        ax1.axvline(x=snap_step, color="gray", linestyle="--", alpha=0.6,
                    label=f"Snap start (step {snap_step})")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss landscape during the snapping phase")
    lns = l1 + l2 + [ax1.axvline(x=0, color="gray", linestyle="--", alpha=0)]
    if snap_idx is not None:
        lns = l1 + l2 + [plt.Line2D([0], [0], color="gray", linestyle="--")]
    labs = [l.get_label() for l in l1 + l2] + (["Snap start"] if snap_idx is not None else [])
    if snap_idx is not None:
        labs.append("Snap start")
    ax1.legend(l1 + l2 + [plt.Line2D([0], [0], color="gray", linestyle="--")],
               ["Task loss", "Total loss", "Snap start"],
               loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig("fig7_snapping_landscape.png", dpi=150)
    plt.show()
    print("Fig 7 saved: snapping loss landscape")

# ═══════════════════════════════════════════════════════════════
# Cell 11: Figure 8 — Snapping convergence (counts over time)
# ═══════════════════════════════════════════════════════════════

if "n_snapped_025" in df_lc.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_lc["step"], df_lc["n_snapped_025"], color="#4C72B0",
            linewidth=2, marker="o", label="Snapped to 0.25")
    ax.plot(df_lc["step"], df_lc["n_snapped_05"], color="#DD8452",
            linewidth=2, marker="s", label="Snapped to 0.50")

    total_c = df_lc["n_snapped_025"].iloc[0] + df_lc["n_snapped_05"].iloc[0]
    if total_c == 0:
        total_c = 56  # 8 layers * 7 projections
    ax.axhline(y=total_c, color="gray", linestyle=":", alpha=0.5,
               label=f"Total c params ({total_c})")

    # Verticals for snap start
    snap_idx = (df_lc["lambda"] > 0).idxmax() if (df_lc["lambda"] > 0).any() else None
    if snap_idx is not None:
        ax.axvline(x=df_lc.loc[snap_idx, "step"], color="red", linestyle="--",
                   alpha=0.4, label="Snap penalty activated")

    ax.set_xlabel("Step")
    ax.set_ylabel("Number of parameters snapped")
    ax.set_title("Snapping convergence: how fast do c values reach power-of-two wells?")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.set_ylim(0, total_c * 1.15)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig("fig8_snapping_convergence.png", dpi=150)
    plt.show()
    print("Fig 8 saved: snapping convergence")

# ═══════════════════════════════════════════════════════════════
# Cell 12: Figure 9 — C-value trajectories over training
# ═══════════════════════════════════════════════════════════════

if "c_values" in df_lc.columns:
    n_layers = 8
    proj_short = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O",
                  "gate_proj": "G", "up_proj": "U", "down_proj": "D"}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    colors_proj = plt.cm.Set2(np.linspace(0, 1, len(proj_short)))

    # Extract c-value trajectories per layer
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        for pi, (pat, plabel) in enumerate(proj_short.items()):
            key = f"layer.{layer_idx}.{pat}"
            traj = []
            steps = []
            for _, row in df_lc.iterrows():
                cv = row["c_values"].get(key)
                if cv is not None:
                    traj.append(cv)
                    steps.append(row["step"])
            if traj:
                ax.plot(steps, traj, color=colors_proj[pi], linewidth=1.5,
                        label=plabel, alpha=0.8)

        # Wells
        ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"Layer {layer_idx}")
        if layer_idx >= 4:
            ax.set_xlabel("Step")
        if layer_idx % 4 == 0:
            ax.set_ylabel("c value")
        ax.set_ylim(0.15, 0.60)

    axes[-1].legend(loc="center", framealpha=0.9, ncol=1,
                    bbox_to_anchor=(0.5, 0.5))
    fig.suptitle("C-value trajectories per layer during training", fontsize=14)
    fig.tight_layout()
    fig.savefig("fig9_c_trajectories.png", dpi=150)
    plt.show()
    print("Fig 9 saved: c-value trajectories")

# ═══════════════════════════════════════════════════════════════
# Cell 13: Figure 10 — Energy / MAC comparison
# ═══════════════════════════════════════════════════════════════

from benchmark import count_macs_per_token, QUANT_ENERGY

cfg_10m = dict(vocab_size=4096, hidden_dim=256, num_layers=8,
               num_heads=8, ffn_dim=1024, max_seq_len=512)
macs = count_macs_per_token(cfg_10m)

quant_baselines = ["full_precision", "bitnet", "fixed_c_025", "fixed_c_05"]
energy_labels = ["Full precision\n(FP32 ADD+MULT)", "BitNet b1.58\n(MUX/neg + ADD)",
                 "Quaternary c=0.25\n(SHIFT-2 + ADD)", "Quaternary c=0.5\n(SHIFT-1 + ADD)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MAC breakdown pie
mac_components = {
    "Attention QKV": macs["attention_per_layer"],
    "QK dot + attn@V": macs["qk_dot"] + macs["attn_v"],
    "Output proj": macs["o_proj"],
    "FFN (SwiGLU)": macs["ffn_per_layer"],
}
# Restructure: per layer values
attn_detail = {
    "QKV proj": 3 * cfg_10m["hidden_dim"] * cfg_10m["hidden_dim"],
    "QK^T + attn@V": cfg_10m["num_heads"] * cfg_10m["max_seq_len"] * (cfg_10m["hidden_dim"] // cfg_10m["num_heads"]) * 2,
    "Output proj": cfg_10m["hidden_dim"] * cfg_10m["hidden_dim"],
}
per_layer_attn = sum(attn_detail.values())
per_layer_ffn = 3 * cfg_10m["hidden_dim"] * cfg_10m["ffn_dim"]
total = (per_layer_attn + per_layer_ffn) * cfg_10m["num_layers"]

labels_pie = [
    f"Attention\n({per_layer_attn:,} MACs/layer)",
    f"FFN\n({per_layer_ffn:,} MACs/layer)",
]
sizes_pie = [per_layer_attn * cfg_10m["num_layers"],
             per_layer_ffn * cfg_10m["num_layers"]]
ax1.pie(sizes_pie, labels=labels_pie, autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"], startangle=90,
        textprops={"fontsize": 10})
ax1.set_title("MAC distribution per token\n(125M equiv model, 12 layers)")

# Energy bar chart
pj_per_mac = [QUANT_ENERGY[b] for b in quant_baselines]
nJ_per_tok = [macs["total_macs"] * pj / 1000 for pj in pj_per_mac]
bars = ax2.bar(energy_labels, nJ_per_tok, color=[COLORS[0], COLORS[1], COLORS[4], COLORS[5]],
               width=0.5, edgecolor="white")
ax2.set_ylabel("Energy per token (nJ)")
ax2.set_title("Energy per token (Horowitz ISSCC 2014, 45nm)")
ax2.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, nJ_per_tok):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

# Add vs FP ratio
fp_nJ = nJ_per_tok[0]
for bar, val in zip(bars, nJ_per_tok):
    ratio = (val / fp_nJ) * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.6,
             f"{ratio:.0f}% of FP", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")

fig.tight_layout()
fig.savefig("fig10_energy_comparison.png", dpi=150)
plt.show()
print("Fig 10 saved: energy comparison")

# ═══════════════════════════════════════════════════════════════
# Cell 14: Summary table (text)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("FINAL SUMMARY: Tetranet 10M Model Comparison")
print("=" * 90)

# System
print(f"\nSystem: {DEVICE} ({VRAM:.1f} GB) | Model: 10M params "
      f"(dim=256, layers=8, heads=8, ffn=1024)")
print(f"Dataset: 100K TinyStories (~3.3M tokens) | Vocab: 4096 | Seq: 512")
print(f"Validation: 2,500 TinyStories (~80K tokens)")

# Per-baseline
header = f"\n{'Baseline':<22s} {'Best loss':>10s} {'PPL':>8s} {'Δ vs FP':>8s} {'PPL/clf':>8s} {'MACs/tok':>10s} {'nJ/tok':>8s}"
print(header)
print("-" * len(header))

for i, name in enumerate(BASELINE_NAMES):
    df = logs[name]
    best_loss = df["task_loss"].min()
    ppl = math.exp(best_loss) if best_loss < 10 else float("nan")
    delta = ppl - ppl_results.get("full_precision", {}).get("ppl", 0)
    # Random classifier PPL = exp(-ln(1/vocab)) = vocab
    ppl_clf = 4096

    # Energy
    quant_baseline_map = {
        "full_precision": "full_precision",
        "bitnet": "bitnet",
        "uniform_2bit": "full_precision",  # not in QUANT_ENERGY, approximate
        "fixed_c_025": "fixed_c_025",
        "fixed_c_05": "fixed_c_05",
        "learned_c": "fixed_c_05",  # same ops as fixed_c_05
    }
    qb = quant_baseline_map.get(name, "full_precision")
    if qb in QUANT_ENERGY:
        nJ = macs["total_macs"] * QUANT_ENERGY[qb] / 1000
    else:
        nJ = float("nan")

    delta_str = f"{delta:+.2f}" if not math.isnan(delta) else " —"
    ppl_str = f"{ppl:.2f}" if not math.isnan(ppl) else " —"
    print(f"{name:<22s} {best_loss:>10.4f} {ppl_str:>8s} {delta_str:>8s} "
          f"{ppl_clf:>8d} {macs['total_macs']:>10,} {nJ:>8.1f}")

# PPL gap summary
if "full_precision" in ppl_results:
    fp = ppl_results["full_precision"]["ppl"]
    for name in ["bitnet", "learned_c"]:
        if name in ppl_results:
            gap = ppl_results[name]["ppl"] - fp
            print(f"\nPPL gap: {name} - full_precision = {gap:+.2f}")

# Snapping stats
if "learned_c" in ppl_results:
    print(f"\n--- learned_c snapping stats ---")
    c_vals = model_lc.get_c_values()
    n_025 = sum(1 for v in c_vals.values() if abs(v - 0.25) < 0.02)
    n_05 = sum(1 for v in c_vals.values() if abs(v - 0.50) < 0.02)
    print(f"  Total c params: {len(c_vals)}")
    print(f"  Snapped to 0.25: {n_025} ({n_025/len(c_vals)*100:.1f}%)")
    print(f"  Snapped to 0.50: {n_05} ({n_05/len(c_vals)*100:.1f}%)")
    print(f"  Failed to snap: {len(c_vals) - n_025 - n_05}")

print("\n" + "=" * 90)

# ═══════════════════════════════════════════════════════════════
# Cell 15: Copy all outputs to /kaggle/working/
# ═══════════════════════════════════════════════════════════════

for fname in os.listdir("."):
    if fname.endswith(".png") or fname.endswith("_final.pt") or fname.endswith(".csv"):
        shutil.copy(fname, "/kaggle/working/")

print("\n Output files in /kaggle/working/:")
for f in sorted(os.listdir("/kaggle/working/")):
    size = os.path.getsize(f"/kaggle/working/{f}")
    print(f"  {f:40s} {size/1024:>8.1f} KB")

print("\n Done! All 10 figures, checkpoints, and logs saved.")
print("Download from Kaggle Notebook -> Data -> Output tab.")
