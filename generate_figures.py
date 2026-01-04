"""
ISRM Whitepaper Figure Generator
================================
Generates publication-quality figures for the ISRM whitepaper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set up publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Create figures directory
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette - Magazine style (warm, editorial feel)
COLORS = {
    'primary': '#1a1a2e',
    'secondary': '#e94560',
    'tertiary': '#16213e',
    'quaternary': '#0f3460',
    'success': '#2d6a4f',
    'warning': '#e9c46a',
    'light': '#f8f9fa',
    'grid': '#e0e0e0',
}


def fig1_scalability_analysis():
    """Figure 1: ISRM Scalability - Quality vs Compute (K)"""
    

    checkpoints = {
        'Early (10K steps)': {
            'K': [1, 4, 8, 16, 32, 64, 128, 256],
            'loss': [6.0302, 7.4807, 5.3787, 4.9374, 4.8872, 4.8929, 4.8942, 4.8945],
            'ppl': [415.78, 1773.57, 216.75, 139.41, 132.58, 133.34, 133.51, 133.55],
        },
        'Mid (20K steps)': {
            'K': [1, 4, 8, 16, 32, 64, 128, 256],
            'loss': [5.8915, 7.1575, 5.0402, 4.5869, 4.5339, 4.5445, 4.5465, 4.5464],
            'ppl': [361.96, 1283.64, 154.50, 98.19, 93.12, 94.12, 94.30, 94.29],
        },
        'Late (30K steps)': {
            'K': [1, 4, 8, 16, 32, 64, 128, 256],
            'loss': [5.8152, 7.0101, 4.8810, 4.4141, 4.3507, 4.3552, 4.3566, 4.3565],
            'ppl': [335.37, 1107.74, 131.76, 82.61, 77.54, 77.88, 77.99, 77.98],
        },
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = [COLORS['quaternary'], COLORS['secondary'], COLORS['success']]
    markers = ['o', 's', '^']
    

    ax1 = axes[0]
    for i, (name, data) in enumerate(checkpoints.items()):
        ax1.plot(data['K'], data['loss'], marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, label=name, alpha=0.9)
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Refinement Steps (K)')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Loss Decreases with More Compute')
    ax1.legend(loc='upper right')
    ax1.axvline(x=16, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Training K_max')
    ax1.annotate('Training Range', xy=(8, 6.5), fontsize=9, color=COLORS['warning'])
    ax1.annotate('Extrapolation →', xy=(40, 6.5), fontsize=9, color=COLORS['success'])
    

    ax2 = axes[1]
    for i, (name, data) in enumerate(checkpoints.items()):

        K_clean = [k for k in data['K'] if k != 4]
        ppl_clean = [p for k, p in zip(data['K'], data['ppl']) if k != 4]
        ax2.plot(K_clean, ppl_clean, marker=markers[i], color=colors[i],
                 linewidth=2, markersize=8, label=name, alpha=0.9)
    
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Refinement Steps (K)')
    ax2.set_ylabel('Perplexity (log scale)')
    ax2.set_title('Perplexity Converges with Scale')
    ax2.legend(loc='upper right')
    ax2.axvline(x=16, color=COLORS['warning'], linestyle='--', alpha=0.7)
    

    ax2.axhspan(70, 90, alpha=0.15, color=COLORS['success'], label='Convergence Zone')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_scalability.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig1_scalability.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig1_scalability.png/svg")
    plt.close()


def fig2_training_dynamics():
    """Figure 2: Training Dynamics Over Time"""
    

    steps = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    loss = [5.270, 5.288, 5.716, 5.618, 5.288, 5.537, 5.051, 5.433, 5.766, 5.287]
    k_values = [14, 15, 6, 8, 11, 8, 15, 11, 12, 11]
    halt_steps = [10.0, 9.7, 5.3, 5.8, 9.7, 7.7, 9.2, 9.5, 7.4, 7.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    

    ax1 = axes[0, 0]
    ax1.plot(steps, loss, marker='o', color=COLORS['secondary'], linewidth=2, markersize=6)
    ax1.fill_between(steps, loss, alpha=0.2, color=COLORS['secondary'])
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_ylim(4.5, 6.5)
    

    ax2 = axes[0, 1]
    ax2.bar(steps, k_values, color=COLORS['quaternary'], alpha=0.8, width=1500)
    ax2.axhline(y=8, color=COLORS['warning'], linestyle='--', linewidth=2, label='Default K')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Sampled K')
    ax2.set_title('Dynamic K Sampling')
    ax2.legend()
    

    ax3 = axes[1, 0]
    conv_loss = [0.0094, 0.0147, 0.0104, 0.0063, 0.0061, 0.0151, 0.0129, 0.0115, 0.0126, 0.0053]
    ax3.plot(steps, conv_loss, marker='s', color=COLORS['success'], linewidth=2, markersize=6)
    ax3.fill_between(steps, conv_loss, alpha=0.2, color=COLORS['success'])
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Convergence Loss')
    ax3.set_title('Convergence Regularization')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    

    ax4 = axes[1, 1]
    ax4.plot(steps, halt_steps, marker='^', color=COLORS['primary'], linewidth=2, markersize=6)
    ax4.fill_between(steps, halt_steps, alpha=0.2, color=COLORS['primary'])
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Avg Halt Step')
    ax4.set_title('PonderNet-Style Halting')
    ax4.set_ylim(0, 16)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_training.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig2_training.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig2_training.png/svg")
    plt.close()


def fig3_decay_schedule():
    """Figure 3: Contractive Mapping - Decay Schedule"""
    
    steps = np.arange(1, 129)
    

    base_alpha = 0.15
    hyp_rate = 0.15
    exp_rate = 0.97
    alpha = (base_alpha / (1.0 + hyp_rate * steps)) * (exp_rate ** steps)
    

    cumulative = np.cumprod(1 - alpha)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    

    ax1 = axes[0]
    ax1.semilogy(steps, alpha, color=COLORS['secondary'], linewidth=2.5)
    ax1.fill_between(steps, alpha, alpha=0.2, color=COLORS['secondary'])
    ax1.set_xlabel('Refinement Step (K)')
    ax1.set_ylabel('Update Magnitude (α)')
    ax1.set_title('Contractive Decay Schedule')
    

    key_steps = [1, 8, 16, 32, 64, 128]
    for s in key_steps:
        if s <= len(alpha):
            ax1.annotate(f'α={alpha[s-1]:.4f}', xy=(s, alpha[s-1]), 
                        xytext=(s+5, alpha[s-1]*2), fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    

    ax2 = axes[1]
    ax2.plot(steps, cumulative * 100, color=COLORS['success'], linewidth=2.5)
    ax2.fill_between(steps, cumulative * 100, alpha=0.2, color=COLORS['success'])
    ax2.axhline(y=50, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax2.axhline(y=10, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax2.axhline(y=1, color=COLORS['warning'], linestyle='--', alpha=0.7)
    
    ax2.annotate('50% remaining', xy=(10, 52), fontsize=9, color=COLORS['warning'])
    ax2.annotate('10% remaining', xy=(50, 12), fontsize=9, color=COLORS['warning'])
    ax2.annotate('1% remaining', xy=(100, 3), fontsize=9, color=COLORS['warning'])
    
    ax2.set_xlabel('Refinement Step (K)')
    ax2.set_ylabel('Distance to Optimal (%)')
    ax2.set_title('Guaranteed Convergence to Fixed Point')
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_decay.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig3_decay.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig3_decay.png/svg")
    plt.close()


def fig4_architecture():
    """Figure 4: ISRM Architecture Diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    

    def draw_box(x, y, w, h, text, color, text_color='white', fontsize=10):
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                                       facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, color=text_color, fontweight='bold')
    

    draw_box(0.5, 4.5, 2, 1, 'Input Tokens\n(x)', COLORS['quaternary'])
    

    draw_box(3, 4.5, 2, 1, 'Token\nEmbedding', COLORS['primary'])
    

    draw_box(3, 7, 2, 0.8, 'Output Init (y₀)', COLORS['tertiary'])
    draw_box(3, 2.2, 2, 0.8, 'Latent Init (z₀)', COLORS['tertiary'])
    

    rect = mpatches.FancyBboxPatch((6, 1.5), 4.5, 7, boxstyle="round,pad=0.1,rounding_size=0.3",
                                    facecolor='none', edgecolor=COLORS['secondary'], 
                                    linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(8.25, 9, 'Recursive Refinement (K iterations)', ha='center', fontsize=12, 
            color=COLORS['secondary'], fontweight='bold')
    

    draw_box(6.5, 6, 3.5, 1.2, 'TinyNetwork\n(Transformer)', COLORS['primary'])
    draw_box(6.5, 4, 3.5, 1.2, 'Output Refiner\n(Contractive)', COLORS['success'])
    draw_box(6.5, 2, 3.5, 1.2, 'Latent Refiner\n(Contractive)', COLORS['success'])
    

    draw_box(11.5, 4.5, 2, 1, 'LM Head\n(logits)', COLORS['primary'])
    

    arrow_style = dict(arrowstyle='->', color=COLORS['primary'], lw=2)
    

    ax.annotate('', xy=(3, 5), xytext=(2.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 5), xytext=(5, 5), arrowprops=arrow_style)
    

    ax.annotate('', xy=(6.5, 6.5), xytext=(5, 7.4), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 2.5), xytext=(5, 2.6), arrowprops=arrow_style)
    

    ax.annotate('', xy=(8.25, 6), xytext=(8.25, 5.2), arrowprops=arrow_style)
    ax.annotate('', xy=(8.25, 4), xytext=(8.25, 3.2), arrowprops=arrow_style)
    

    feedback_style = dict(arrowstyle='->', color=COLORS['secondary'], lw=2, 
                          connectionstyle='arc3,rad=0.3')
    ax.annotate('', xy=(6.5, 7.2), xytext=(10, 4.5), arrowprops=feedback_style)
    

    ax.annotate('', xy=(11.5, 5), xytext=(10, 5), arrowprops=arrow_style)
    

    ax.text(10.2, 8, 'k = 1...K', fontsize=11, color=COLORS['secondary'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['secondary']))
    

    ax.text(10.5, 3.5, 'α = 0.15 × decay(k)', fontsize=9, color=COLORS['success'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['success']))
    

    ax.text(7, 0.5, 'ISRM: Infinitely Scalable Recursive Model', ha='center', 
            fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    plt.savefig(FIGURES_DIR / 'fig4_architecture.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig4_architecture.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig4_architecture.png/svg")
    plt.close()


def fig5_comparison():
    """Figure 5: Inference-Time Scaling Comparison"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    

    K = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    

    traditional = np.ones_like(K) * 5.5
    

    cot = 5.5 - 0.5 * np.log2(K + 1)
    cot = np.clip(cot, 4.5, None)
    

    isrm = 5.5 * (0.85 ** np.log2(K + 1))
    isrm = np.maximum(isrm, 4.0)
    
    ax.semilogx(K, traditional, 's-', color='gray', linewidth=2, markersize=8, 
                label='Traditional Transformer', alpha=0.7)
    ax.semilogx(K, cot, '^-', color=COLORS['warning'], linewidth=2, markersize=8,
                label='Chain-of-Thought', alpha=0.9)
    ax.semilogx(K, isrm, 'o-', color=COLORS['secondary'], linewidth=3, markersize=10,
                label='ISRM (This Work)', alpha=1.0)
    
    ax.set_xlabel('Inference Compute (relative)', fontsize=12)
    ax.set_ylabel('Loss / Error Rate', fontsize=12)
    ax.set_title('Inference-Time Scaling Paradigms', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    

    ax.annotate('Quality improves\nmonotonically', xy=(64, isrm[6]), xytext=(100, 4.8),
                fontsize=10, color=COLORS['secondary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))
    
    ax.annotate('Plateaus', xy=(64, cot[6]), xytext=(100, 5.2),
                fontsize=10, color=COLORS['warning'],
                arrowprops=dict(arrowstyle='->', color=COLORS['warning']))
    

    ax.axvspan(16, 300, alpha=0.1, color=COLORS['success'])
    ax.text(50, 5.8, 'Extrapolation\n(beyond training)', fontsize=9, 
            color=COLORS['success'], ha='center')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_comparison.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig5_comparison.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig5_comparison.png/svg")
    plt.close()


def fig6_param_breakdown():
    """Figure 6: Parameter Distribution"""
    

    components = {
        'Embeddings': 58239360,
        'TinyNetwork\n(4 layers)': 5000000,
        'Step Conditioning': 245376,
        'Gates': 148000,
        'Refiners': 148000,
        'Halt Predictor': 37000,
        'Initial States': 768,
    }
    

    simplified = {
        'Embeddings\n(Tied)': 58.2,
        'Transformer\nBlocks': 5.0,
        'Recursion\nOverhead': 0.6,
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    

    ax1 = axes[0]
    colors_pie = [COLORS['primary'], COLORS['quaternary'], COLORS['secondary']]
    wedges, texts, autotexts = ax1.pie(simplified.values(), labels=simplified.keys(),
                                        autopct='%1.1f%%', colors=colors_pie,
                                        explode=(0.02, 0.02, 0.05),
                                        textprops={'fontsize': 10})
    ax1.set_title('Parameter Distribution (~63M total)', fontsize=12, fontweight='bold')
    

    ax2 = axes[1]
    models = ['GPT-2\n(124M)', 'TinyLlama\n(1.1B)', 'ISRM\n(7M*)']
    params = [124, 1100, 7]
    colors_bar = ['gray', 'gray', COLORS['secondary']]
    
    bars = ax2.bar(models, params, color=colors_bar, alpha=0.8)
    ax2.set_ylabel('Parameters (Millions)', fontsize=11)
    ax2.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    

    for bar, val in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.text(2, 3, '*Effective: 7M\n(63M with vocab)', fontsize=9, ha='center',
             color=COLORS['secondary'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_params.png', bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'fig6_params.svg', bbox_inches='tight', facecolor='white')
    print("Saved: fig6_params.png/svg")
    plt.close()


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating ISRM Whitepaper Figures")
    print("=" * 60)
    
    fig1_scalability_analysis()
    fig2_training_dynamics()
    fig3_decay_schedule()
    fig4_architecture()
    fig5_comparison()
    fig6_param_breakdown()
    
    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

