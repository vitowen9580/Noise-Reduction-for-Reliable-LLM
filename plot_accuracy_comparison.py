import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 9.5,
    'axes.linewidth': 1.1,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

COND_ORDER = ['A_noisy_audio', 'C_enhanced_audio', 'B_separated_audio', 'D_clean_audio']
COND_MARKERS = {
    'A_noisy_audio': 's',
    'B_separated_audio': '^',
    'C_enhanced_audio': 'o',
    'D_clean_audio': 'D'
}

COND_LABELS = {
    'A_noisy_audio': 'Noisy',
    'B_separated_audio': 'Denoised',
    'C_enhanced_audio': 'Enhanced',
    'D_clean_audio': 'Clean'
}


def parse_by_folder_names(base_dir: str):
    root = Path(base_dir).resolve()
    if not root.exists():
        print(f"Error: Base directory '{base_dir}' does not exist!")
        return None

    json_files = sorted([
        p for p in root.rglob("four_conditions_comparison_results*.json")
        if p.name != "summary.json"
    ])

    if not json_files:
        print(f"Error: No 'four_conditions_comparison_results*.json' found under '{base_dir}'!")
        return None

    records = []
    for fpath in json_files:
        rel_parts = fpath.relative_to(root).parts
        exp_folder = rel_parts[0] if len(rel_parts) > 1 else fpath.parent.name

        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
                continue

        if not isinstance(data, list) or len(data) == 0:
            continue

        n = len(data)
        for cond in COND_ORDER:
            acc = sum(1 for q in data if q.get(f"{cond}_is_correct") is True) / n * 100
            snr = sum(q.get(f"{cond}_SNR_dB", 0) for q in data) / n
            records.append({
                'Path': str(fpath),
                'Experiment': exp_folder,
                'Condition': cond,
                'Accuracy': acc,
                'SNR_dB': snr
            })

    return pd.DataFrame(records)


def plot_accuracy_comparison(df, target_exps, exp_labels, exp_colors, title, output_filename, include_clean=True):
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=300)

    plot_conds = COND_ORDER if include_clean else COND_ORDER[:-1]
    curve_handles = []
    plotted_count = 0

    for color, exp_id in zip(exp_colors, target_exps):
        sub_df = df[df['Experiment'] == exp_id]
        if sub_df.empty:
            print(f"[WARNING] Folder '{exp_id}' not found or empty, skipping this curve.")
            continue

        label = exp_labels.get(exp_id, exp_id)

        points = []
        for cond in plot_conds:
            row = sub_df[sub_df['Condition'] == cond]
            if not row.empty:
                points.append((row['SNR_dB'].values[0], row['Accuracy'].values[0], cond))

        cond_order_map = {cond: i for i, cond in enumerate(plot_conds)}
        points.sort(key=lambda p: cond_order_map[p[2]])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        ax.plot(x_vals, y_vals, color=color, linewidth=2.4, linestyle='-', label=label, alpha=0.9, zorder=3)
        curve_handles.append(Line2D([0], [0], color=color, lw=2.4, label=label))

        for x, y, cond in points:
            ax.scatter(
                x, y,
                color=color,
                marker=COND_MARKERS[cond],
                s=85,
                edgecolors='black',
                linewidth=1.1,
                zorder=4
            )

            if plotted_count == 0:
                if cond == 'A_noisy_audio':
                    dy = 10
                elif cond == 'C_enhanced_audio':
                    dy = 10
                elif cond == 'B_separated_audio':
                    dy = -16
                else:
                    dy = 10

                ax.annotate(
                    COND_LABELS[cond],
                    (x, y),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    color='#1C2833',
                    weight='bold'
                )

        plotted_count += 1

    if plotted_count == 0:
        plt.close()
        print(f"[SKIPPED] No curves plotted for {output_filename}.")
        return

    ax.set_xlabel("Audio Quality: Average SNR (dB)", fontweight='semibold')
    ax.set_ylabel("Accuracy (%)", fontweight='semibold')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_ylim(10, 95)

    if include_clean:
        ax.set_xscale('symlog', linthresh=15)
        ax.set_xlim(-12, 160)
    else:
        ax.set_xlim(-10, 15)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.1)

    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=0)

    ax.legend(
        handles=curve_handles,
        loc='lower right',
        frameon=True,
        framealpha=0.95,
        shadow=False,
        fancybox=False,
        edgecolor='#B0B0B0',
        fontsize=9.5
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"[SAVED] {output_filename}")


if __name__ == "__main__":
    ######################
    #PARAM
    ######################
    BASE_DIR = "result/syntec_pdf_result"

    df = parse_by_folder_names(BASE_DIR)

    if df is not None and not df.empty:
        print("[LOADED] Detected experiment folders:")
        for exp_name, path_val in df[['Experiment', 'Path']].drop_duplicates().values:
            print(f"  - {exp_name} <- {path_val}")

        exps_1 = ['2_H', '4_H', '7_H']
        labels_1 = {
            '2_H': 'Human Voices',
            '4_H': 'Drilling Noise',
            '7_H': 'Jackhammer Noise'
        }
        colors_1 = ['#1F77B4', '#8E44AD', '#D35400']
        plot_accuracy_comparison(
            df=df,
            target_exps=exps_1,
            exp_labels=labels_1,
            exp_colors=colors_1,
            title="Chart 1: Audio Quality vs Accuracy Trajectory (Single Noise Types)",
            output_filename=os.path.join(BASE_DIR, "Chart1_Grouped_Single_Noises.png")
        )

        exps_2 = ['2_H', '2,7_H', '2,4,7_H']
        labels_2 = {
            '2_H': 'Human Voices',
            '2,7_H': 'Human Voices + Jackhammer',
            '2,4,7_H': 'Human Voices + Drilling + Jackhammer'
        }
        colors_2 = ['#27AE60', '#F39C12', '#C0392B']
        plot_accuracy_comparison(
            df=df,
            target_exps=exps_2,
            exp_labels=labels_2,
            exp_colors=colors_2,
            title="Chart 2: Audio Quality vs Accuracy Trajectory (Increasing Noise Complexity)",
            output_filename=os.path.join(BASE_DIR, "Chart2_Grouped_Noise_Complexity.png")
        )

        exps_3 = ['2,4,7_L', '2,4,7_M', '2,4,7_H']
        labels_3 = {
            '2,4,7_L': 'Human Voices + Drilling + Jackhammer (Low)',
            '2,4,7_M': 'Human Voices + Drilling + Jackhammer (Medium)',
            '2,4,7_H': 'Human Voices + Drilling + Jackhammer (High)'
        }
        colors_3 = ['#16A085', '#2980B9', '#8E44AD']
        plot_accuracy_comparison(
            df=df,
            target_exps=exps_3,
            exp_labels=labels_3,
            exp_colors=colors_3,
            title="Chart 3: Audio Quality vs Accuracy Trajectory (Across SNR Levels)",
            output_filename=os.path.join(BASE_DIR, "Chart3_Grouped_SNR_Levels.png")
        )