"""Shared rendering for the equality-study figures in the analysis notebook."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from IPython.display import Image, display
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


def plot_facets(data, curves, networks, styles, *, style, output_path, zoom_steps=None):
    """Draw, export, and display the data on shared or zoomed y scales."""
    zoomed = zoom_steps is not None
    rc = {
        'text.usetex': False, 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 9 if zoomed else 10,
        'svg.fonttype': 'none',
    }
    with plt.style.context(style), plt.rc_context(rc):
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4.2) if zoomed else (10.8, 3.8), sharey=not zoomed,
        )
        fig.subplots_adjust(
            left=0.075 if zoomed else 0.085, right=0.985, top=0.90, bottom=0.30,
            wspace=0.48 if zoomed else 0.16,
        )
        for ax, (network, title) in zip(axes, networks.items()):
            points = data.loc[data['network'].eq(network)]
            predictions = curves.loc[curves['network'].eq(network)]
            for condition, appearance in styles.items():
                observed = points.loc[points['condition'].eq(condition)]
                fitted = predictions.loc[predictions['condition'].eq(condition)]
                if observed.empty:
                    continue
                ax.scatter(
                    observed['degree_gini_coefficient'], observed['reliability_pp'],
                    s=11, color=appearance['color'], alpha=0.48,
                    edgecolors='none', zorder=2, rasterized=True,
                )
                ax.fill_between(
                    fitted['gini'], fitted['lower_pp'], fitted['upper_pp'],
                    color=appearance['color'], alpha=0.18, linewidth=0, zorder=1,
                )
                ax.plot(
                    fitted['gini'], fitted['mean_pp'], color='black',
                    linestyle=appearance['linestyle'], linewidth=2.6, alpha=1, zorder=4,
                    path_effects=[
                        patheffects.Stroke(linewidth=4.2, foreground='white', alpha=0.9),
                        patheffects.Normal(),
                    ],
                )
            ax.set_title(title, pad=11)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            ax.grid(axis='y', color='#e6e6e6', linewidth=0.6, zorder=0)
            ax.margins(x=0.04)
            if zoomed:
                # Seven tick rows, anchored at 100%, with equal proportional padding.
                step = zoom_steps[network]
                span = 6 * step
                margin = 0.04 * span
                limits = (100 - span - margin, 100 + margin)
                low = min(points['reliability_pp'].min(), predictions['lower_pp'].min())
                high = max(points['reliability_pp'].max(), predictions['upper_pp'].max())
                if not limits[0] < low <= high < limits[1]:
                    raise ValueError(f'Increase the zoom tick step for {network} to include all data and CIs.')
                ax.set_ylim(limits)
                ax.set_yticks(100 - np.arange(6, -1, -1) * step)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}%'))
        axes[0].set_ylabel('Share of correct agents', labelpad=10)
        if zoomed:
            _shade_zoom_gaps(fig, axes)
        else:
            low = min(curves['lower_pp'].min(), data['reliability_pp'].min())
            high = max(curves['upper_pp'].max(), data['reliability_pp'].max())
            margin = 0.06 * (high - low)
            axes[0].set_ylim(low - margin, high + margin)
            axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        handles = [
            (
                Line2D([], [], linestyle='None', marker='o', markersize=10,
                       markerfacecolor=item['color'], markeredgewidth=0, zorder=1),
                Line2D([], [], color='black', linestyle=item['linestyle'], linewidth=2.6, zorder=2),
            )
            for item in styles.values()
        ]
        fig.legend(
            handles=handles, labels=[item['label'] for item in styles.values()],
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
            loc='lower center', bbox_to_anchor=(0.53 if zoomed else 0.535, 0.005),
            ncol=4, frameon=False, handlelength=2.7, columnspacing=1.3, fontsize=9.5,
        )
        fig.supxlabel(
            'Degree Gini coefficient (higher = more unequal)',
            x=fig.subplotpars.right, y=0.13, ha='right', fontsize=11, fontweight='bold',
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {extension: Path(f'{output_path}.{extension}') for extension in ('png', 'svg')}
        # Render inside the style context: automatic tick spacing depends on it.
        for extension, path in paths.items():
            fig.savefig(path, format=extension, dpi=300, bbox_inches='tight')
        display(fig)
        plt.close(fig)
    return fig, paths


def _shade_zoom_gaps(fig, axes):
    """Connect nested y ranges using fill confined to the spaces between panels."""
    to_figure = fig.transFigure.inverted()
    for left, right in zip(axes[:-1], axes[1:]):
        low, high = right.get_ylim()
        if not left.get_ylim()[0] < low < high < left.get_ylim()[1]:
            raise ValueError('Zoom tick steps must decrease from left to right.')
        bottom, top = to_figure.transform(left.get_yaxis_transform().transform([(1, low), (1, high)]))
        target = right.get_position()
        fig.add_artist(Polygon(
            [bottom, (target.x0, target.y0), (target.x0, target.y1), top],
            closed=True, transform=fig.transFigure, facecolor='#777777', alpha=0.055,
            edgecolor='none', linewidth=0, antialiased=False, clip_on=False, zorder=-1,
        ))


def _transform_pixels(rgb, mode):
    """Convert the rendered image in small batches to limit memory use."""
    result = np.empty(rgb.shape, dtype=np.uint8)
    for start in range(0, len(rgb), 128):
        pixels = rgb[start:start + 128]
        if mode == 'grayscale':
            linear = cspace_convert(pixels, 'sRGB1', 'sRGB1-linear')
            luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
            gray = np.repeat(luminance[..., None], 3, axis=-1)
            converted = cspace_convert(gray, 'sRGB1-linear', 'sRGB1')
        else:
            cvd = {
                'name': 'sRGB1+CVD',
                'cvd_type': 'protanomaly' if mode == 'protanopia' else 'deuteranomaly',
                'severity': 100,
            }
            # Simulation runs FROM the CVD space TO ordinary sRGB.
            converted = cspace_convert(pixels, cvd, 'sRGB1')
        result[start:start + 128] = np.rint(np.clip(converted, 0, 1) * 255).astype(np.uint8)
    return result


def finished_plot_previews(png_path, *, prefix=''):
    """Save/display grayscale and red–green simulations of a finished plot."""
    png_path = Path(png_path)
    pixels = mpimg.imread(png_path)
    rgb = pixels[..., :3]
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3:4]
        rgb = rgb * alpha + (1 - alpha)
    modes = {
        'grayscale': 'Black and white / grayscale',
        'protanopia': 'Protanopia simulation (full severity)',
        'deuteranopia': 'Deuteranopia simulation (full severity)',
    }
    paths = {}
    for mode, label in modes.items():
        path = png_path.with_name(f'{png_path.stem}_{mode}.png')
        mpimg.imsave(path, _transform_pixels(rgb, mode), dpi=300)
        paths[mode] = path
        print(prefix + label, flush=True)
        display(Image(filename=str(path), width=1080))
        print(f'Saved: {path}')
    print('Transformations applied to the finished figure, including its legend and shading.')
    return paths
