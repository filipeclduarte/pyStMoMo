import matplotlib.pyplot as plt


def set_style():
    """Apply a premium, modern aesthetic to matplotlib plots."""

    # Modern, clean color palette (harmonious HSL-based)
    # 0: Primary (Blue), 1: Secondary (Teal), 2: Accent (Indigo), 3: Danger (Red/Orange), 4: Neutral
    colors = [
        '#2563eb', # Blue 600
        '#0d9488', # Teal 600
        '#4f46e5', # Indigo 600
        '#dc2626', # Red 600
        '#d97706', # Amber 600
        '#7c3aed', # Violet 600
    ]

    plt.rcParams.update({
        # Figure and Layout
        'figure.figsize': (10, 6),
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'figure.autolayout': True,

        # Typography
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.titleweight': '600',
        'axes.labelsize': 11,
        'axes.labelweight': '500',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Spines and Grid
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.axisbelow': True,
        'axes.grid': True,
        'grid.color': '#f1f5f9', # Slate 100
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,

        # Colors and Lines
        'axes.prop_cycle': plt.cycler(color=colors),
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'patch.edgecolor': 'none',

        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',

        # Ticks
        'xtick.color': '#475569', # Slate 600
        'ytick.color': '#475569', # Slate 600
        'axes.labelcolor': '#1e293b', # Slate 800
        'axes.edgecolor': '#cbd5e1', # Slate 300
        'text.color': '#1e293b', # Slate 800
    })

def get_colors():
    """Return the primary color palette."""
    return [
        '#2563eb', # Blue 600
        '#0d9488', # Teal 600
        '#4f46e5', # Indigo 600
        '#dc2626', # Red 600
        '#d97706', # Amber 600
        '#7c3aed', # Violet 600
    ]
