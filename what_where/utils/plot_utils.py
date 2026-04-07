

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

import what_where as ww


def check_helvetica_availability():
    """Check if Helvetica is available and what font will actually be used."""
    
    # Check if Helvetica is in the system font list
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    helvetica_available = 'Helvetica' in available_fonts
    
    # Get what font matplotlib will actually use from your sans-serif preference list
    sans_serif_list = plt.rcParams['font.sans-serif']
    actual_font_used = None
    
    for font in sans_serif_list:
        if font in available_fonts:
            actual_font_used = font
            break
    
    # If none of the preferred fonts are available, get the system default
    if actual_font_used is None:
        # Use default sans-serif resolution
        default_font_path = fm.findfont(fm.FontProperties())
        actual_font_used = fm.get_font(default_font_path).family_name
    
    if not helvetica_available:
        warnings.warn(
            f"Helvetica font not available on this system. "
            f"Matplotlib will use: '{actual_font_used}' instead."
        )
        return False, actual_font_used
    else:
        print("✓ Helvetica font is available")
        return True, actual_font_used


def init_plotting(cfg):
    # Clear the font cache
    fm.fontManager.__init__()

    # Or alternatively, rebuild the font list
    fm._load_fontmanager()

    # Clear matplotlib's cache
    plt.rcParams.clear()

    plotting_config = dict(cfg.analysis.plotting_config.rc_params)
    plt.rcParams.update(plotting_config)

    # Check after setting config
    check_helvetica_availability()


def get_figures_dir(cfg):
    figures_dir = ww.utils.ROOT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir