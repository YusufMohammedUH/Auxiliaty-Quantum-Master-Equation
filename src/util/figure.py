from typing import Union
import numpy as np
import colorcet as cc
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf  # saving pdflatex
from matplotlib.ticker import AutoMinorLocator

cm_in_inches = 0.393701
golden_ratio = 1.61803398875

cdict_blrddark = {
    'red': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    'blue': ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
}
cdict_rdbldark = {
    'red': ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    'blue': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0))
}
bl_rd_dark_cmap = mpl.colors.LinearSegmentedColormap(
    name="BlRdDark", segmentdata=cdict_blrddark)
rd_bl_dark_cmap = mpl.colors.LinearSegmentedColormap(
    name="RdBlDark", segmentdata=cdict_rdbldark)

# TODO: The code is taken from a previous project. It should be cleaned up.
#       I should save a list of axis as class attribute and access
#       them by index.


class FigureTheme:
    """Initializing object by passing a color map to class, style and size of
    the saved figure. The default is a with of 8.6 inches and a height
    according to the golden ratio.

    Parameters
    ----------
    cmap : cc.cm.bmy
        color map

    style : Union[str, None], optional
        style name as string, by default None

    width_inches : Union[float, None], optional
        width, by default None

    height_inches : Union[float, None], optional
        height, by default None

    height_multiplier : Union[float, None], optional
        factor with which the height is multiplied, by default None

    Attributes
    ----------
    fig : matplotlib.pyplot.figure
        figure object

    cmap : cc.cm.bmy
        color map

    ax : matplotlib.pyplot.axes
        axes object
    """

    def __init__(self,
                 cmap: cc.cm.bmy,
                 style: Union[str, None] = None,
                 width_inches: Union[float, None] = None,
                 height_inches: Union[float, None] = None,
                 height_multiplier: Union[float, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.cmap = cmap
        self.setRC()
        self.create_figure(style, width_inches, height_inches,
                           height_multiplier)
        self.projection = None
        self.ax = None

    def setRC(self) -> None:
        """
        General setting for plots are defined, e.g.
        use of LaTeX , LaTeX packages, fonts, ticks etc.
        """
        mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

        plt.rc('text', usetex=True)
        plt.rc("pgf", texsystem="lualatex")
        plt.rc("pgf",
               preamble=r'\usepackage{amsmath},'
               r'\usepackage[T1]{fontenc},'
               r'\usepackage[utf8]{inputenc},'
               r'\usepackage{yfonts},'
               r'\usepackage{amsmath},'
               r'\usepackage{amssymb},'
               r'\usepackage{txfonts},'
               r'\usepackage{fontspec},'
               r'\usepackage[Symbolsmallscale]{upgreek},'
               r'\usepackage{times},'
               r'\usepackage{blindtext}')

        plt.rc('font', **{'family': 'serif', 'size': 7.0})
        plt.rc('lines', linewidth=1.0)
        plt.rc('axes', linewidth=0.5)
        plt.rc('xtick', labelsize=7.0, direction='in')
        plt.rc('ytick', labelsize=7.0, direction='in')
        plt.rc('xtick.major', size=4.0, width=0.5)
        plt.rc('xtick.minor', size=2.0, width=0.5)
        plt.rc('ytick.major', size=4.0, width=0.5)
        plt.rc('ytick.minor', size=2.0, width=0.5)
        plt.rc('legend', fontsize='xx-small', loc='best')
        plt.rc('text', usetex=True)

    def create_figure(self,
                      style: Union[str, None] = None,
                      width_inches: Union[float, None] = None,
                      height_inches: Union[float, None] = None,
                      height_multiplier: Union[float, None] = None) -> None:
        """Given with, height and height multiplier in inch, returns a
        matplotlib.pyplot.figure object with these measures. Additionally a
        style is set(default is seborn-whitegrid).

        Parameters
        ----------
        style : Union[str, None], optional
            Style of figure, by default None
        width_inches : Union[float, None], optional
            Figure with, by default None
        height_inches : Union[float, None], optional
            Figure height, by default None
        height_multiplier : Union[float, None], optional
            Figure height multiplier, by default None
        """
        if style is None:
            plt.style.use('seaborn-whitegrid')
        else:
            plt.style.use(style)
        self.fig = plt.figure()
        self.fig.horizontal_merged = False
        self.fig.vertical_merged = False
        if width_inches is None:
            width_inches = 8.6 * cm_in_inches
        self.fig.default_height = width_inches / golden_ratio
        if height_inches is None:
            height_inches = self.fig.default_height
        if height_multiplier is not None:
            height_inches *= height_multiplier
        self.fig.set_size_inches(width_inches, height_inches)

    def set_default_spacing(self) -> None:
        """
        Set default subplot layout. See matplotlib.pyplot.subplot_adjust.
        These are a good start in most cases, but may require some manual
        """

        self.fig.subplots_adjust(bottom=0.15 * self.fig.default_height /
                                 self.fig.get_size_inches()[1])

        self.fig.subplots_adjust(
            top=1.0 -
            0.05 * self.fig.default_height / self.fig.get_size_inches()[1])
        if self.projection == '3d':
            self.fig.subplots_adjust(left=0.15)
            self.fig.subplots_adjust(right=0.95)
        elif self.projection == 'heatmap':
            self.fig.subplots_adjust(left=0.0)
            self.fig.subplots_adjust(right=0.95)
            self.fig.subplots_adjust(bottom=0.20 * self.fig.default_height /
                                     self.fig.get_size_inches()[1])

            self.fig.subplots_adjust(
                top=1.0 -
                0.05 * self.fig.default_height / self.fig.get_size_inches()[1])
        else:
            self.fig.subplots_adjust(left=0.10)
            self.fig.subplots_adjust(right=0.98)

    def create_single_panel(self,
                            projection_: Union[str, None] = None,
                            xlabel: Union[str, None] = None,
                            ylabel: Union[str, None] = None,
                            zlabel: Union[str, None] = None,
                            numcolors: Union[int, None] = 9) -> None:
        """Given parameters, creates a single panel and returns a subplot
        object.

        Parameters
        ----------
        projection_ : Union[str, None], optional
            string, currently for '3d', 'heatmap' or 'None', by default None

        xlabel : Union[str, None], optional
            x label string, by default None

        ylabel : Union[str, None], optional
            y label string, by default None

        zlabel : Union[str, None], optional
            z label string, by default None

        numcolors : Union[int, None], optional
            Number of colors for evenly spacing the colormap, by default 9
        """

        self.projection = projection_
        if projection_ == '3d':
            # necessary for recognizing 3d otherwise unused, therefore waring
            # can occur by ide
            from mpl_toolkits.mplot3d import Axes3D
            self.ax = self.fig.add_subplot(111, projection=projection_)
        else:
            self.ax = self.fig.add_subplot(111, projection=None)
            self.ax.grid(b=False)
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if zlabel is not None and projection_ == '3d':
            self.ax.set_zlabel(zlabel)
        # Comments from Cohen
        # This is a 2x1 plot, with the axes merged by default.
        # TODO: Can be generalized to Nx1.
        if isinstance(self.cmap, str):
            self.cmap = mpl.cm.get_cmap(self.cmap)
        self.ax.set_prop_cycle('color', self.cmap(np.linspace(0, 1,
                                                              numcolors)))
        self.set_default_spacing()

    def filled_plot(self, x: Union[np.ndarray, list],
                    y1: Union[np.ndarray, list],
                    y2: Union[np.ndarray, list], linestyle: str = "solid",
                    alpha: float = 0.5,
                    linealpha: float = 1.0, linewidth: float = 1.0,
                    label: Union[str, None] = None) -> None:
        """Plot a filled area between two curves y1 and y2.

        Parameters
        ----------
        x : Union[np.ndarray, list]
            x values

        y1 : Union[np.ndarray, list]
            y1 values

        y2 : Union[np.ndarray, list]
            y2 values

        linestyle : str, optional
            Linestyle, by default "solid"

        alpha : float, optional
            Opacity of filled area, by default 0.5

        linealpha : float, optional
            Opacity of lines, by default 1.0

        linewidth : float, optional
            Line width, by default 1.0

        label : Union[str, None], optional
            Plot label, by default None
        """
        base_line, = self.ax.plot(x, y1, linestyle=linestyle,
                                  linewidth=linewidth, label=label,
                                  alpha=linealpha)
        color = base_line.get_color()
        self.ax.plot(x, y2, color=color, linestyle=linestyle,
                     linewidth=linewidth, alpha=linealpha)
        self.ax.fill_between(x, y1, y2, color=color, linestyle=linestyle,
                             alpha=alpha, linewidth=linewidth)

    def filled_error_plot(self, x: Union[np.ndarray, list],
                          y: Union[np.ndarray, list],
                          err: Union[np.ndarray, list],
                          linestyle: str = "solid",
                          alpha: float = 0.5, linealpha: float = 1.0,
                          linewidth: float = 1.0,
                          label: Union[str, None] = None) -> None:
        """Plot a filled area between the curve y and the error of the
        function err.Function filled_plot is used for plotting.

        Parameters
        ----------
        x : Union[np.ndarray, list]
            X values

        y : Union[np.ndarray, list]
            Y values

        err : Union[np.ndarray, list]
            Error of y along x

        linestyle : str, optional
            Linestyle, by default "solid"

        alpha : float, optional
            Opacity of filled area, by default 0.5

        linealpha : float, optional
            Opacity of lines, by default 1.0

        linewidth : float, optional
            Line width, by default 1.0

        label : Union[str, None], optional
            Plot label, by default None
        """
        self.filled_plot(x, y - err, y + err, linestyle, alpha, linealpha,
                         linewidth, label)
    # This is a 2x1 plot, with the axes merged by default.
    # TODO: Can be generalized to Nx1.

    def create_vertical_split(self,
                              merged: bool = True,
                              xlabel: Union[str, None] = None,
                              ylabel: Union[str, None] = None,
                              palette: tuple = ('Set1', 'Set1'),
                              numcolors: tuple = (9, 9)):
        """Create a vertical split plot 2x1.

        Parameters
        ----------
        merged : bool, optional
            The borders are merged if True, by default True

        xlabel : Union[str, None], optional
            X label, by default None

        ylabel : Union[str, None], optional
            Y label, by default None

        palette : tuple, optional
            Color palette, by default ('Set1', 'Set1')

        numcolors : tuple, optional
            Number of equdistend colors used from color palettes, by default
            (9, 9)
        """
        ax1 = self.fig.add_subplot(121)

        if merged:
            self.fig.vertical_merged = True
            ax2 = self.fig.add_subplot(122, sharey=ax1)  # Share axes.
            self.fig.subplots_adjust(wspace=0)  # Merge axes.
            plt.setp([a.get_yticklabels() for a in self.fig.axes[1:]],
                     visible=False)  # Remove ticks from right axes.
            if ylabel is not None:
                ax1.set_ylabel(ylabel[0])  # No ylabel for right axes.
        else:
            self.fig.vertical_merged = False
            ax2 = self.fig.add_subplot(122)
            if ylabel is not None:
                ax1.set_ylabel(ylabel[0])
                ax2.set_ylabel(ylabel[1])
            self.fig.subplots_adjust(wspace=0.5)
        if xlabel is not None:
            ax1.set_xlabel(xlabel[0])
            ax2.set_xlabel(xlabel[1])

        ax1.set_prop_cycle(
            'color',
            plt.get_cmap(palette[0])(np.linspace(0, 1, numcolors[0])))
        ax2.set_prop_cycle(
            'color',
            plt.get_cmap(palette[1])(np.linspace(0, 1, numcolors[1])))
        self.set_default_spacing()
        return ax1, ax2

    # This is a 1x2 plot, with the axes merged by default.
    # TODO: Can be generalized to 1xN.

    def create_horizontal_split(self,
                                merged: bool = True,
                                xlabel: Union[str, None] = None,
                                ylabel: Union[str, None] = None,
                                palette: tuple = ('Set1', 'Set1'),
                                numcolors: tuple = (9, 9)):
        """Create a horizontal split plot 1x2.

        Parameters
        ----------
        merged : bool, optional
            The borders are merged if True , by default True

        xlabel : Union[str, None], optional
            X label, by default None

        ylabel : Union[str, None], optional
            Y label, by default None

        palette : tuple, optional
            Color palette, by default ('Set1', 'Set1')

        numcolors : tuple, optional
            Number of equdistend colors used from color palettes, by default
            (9, 9)
        """
        ax1 = self.fig.add_subplot(211)

        if merged:
            self.fig.horizontal_merged = True
            ax2 = self.fig.add_subplot(212, sharex=ax1)  # Share axes.
            self.fig.subplots_adjust(hspace=0)  # Merge axes.
            plt.setp([a.get_xticklabels() for a in self.fig.axes[:-1]],
                     visible=False)  # Remove ticks from top axes.
            if xlabel is not None:
                ax2.set_xlabel(xlabel[1])  # No ylabel for top axes.
        else:
            self.fig.horizontal_merged = False
            ax2 = self.fig.add_subplot(212)
            if ylabel is not None:
                ax1.set_xlabel(xlabel[0])
                ax2.set_xlabel(xlabel[1])
            self.fig.subplots_adjust(hspace=0.5)
        if ylabel is not None:
            ax1.set_ylabel(ylabel[0])
            ax2.set_ylabel(ylabel[1])

        ax1.set_prop_cycle(
            'color',
            plt.get_cmap(palette[0])(np.linspace(0, 1, numcolors[0])))
        ax2.set_prop_cycle(
            'color',
            plt.get_cmap(palette[1])(np.linspace(0, 1, numcolors[1])))
        self.set_default_spacing()
        return ax1, ax2

    # This is a 2x2 plot, with the axes always merged.
    # TODO: Can be generalized to NxM, partially merged.

    def create_quad_split(self,
                          xlabel: Union[str, None] = None,
                          ylabel: Union[str, None] = None,
                          palette: tuple = ('Set1', 'Set1', 'Set1', 'Set1'),
                          numcolors: tuple = (9, 9, 9, 9)):
        """Create a 2x2 plot with the axes merged.

        Parameters
        ----------
        xlabel : Union[str, None], optional
            X label, by default None

        ylabel : Union[str, None], optional
            Y label, by default None

        palette : tuple, optional
            Color palette, by default ('Set1', 'Set1', 'Set1', 'Set1')

        numcolors : tuple, optional
            Number of equdistend colors used from color palettes, by default
            (9, 9, 9, 9)

        Returns
        -------
        _type_
            _description_
        """
        self.fig.horizontal_merged = True
        self.fig.vertical_merged = True
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222, sharey=ax1)
        ax3 = self.fig.add_subplot(223, sharex=ax1)
        ax4 = self.fig.add_subplot(224, sharey=ax3)

        # Merge axes.
        self.fig.subplots_adjust(wspace=0)
        self.fig.subplots_adjust(hspace=0)

        # Remove ticks from top axes.
        for p in [0, 1]:
            plt.setp(self.fig.axes[p].get_xticklabels(), visible=False)
        # Remove ticks from right axes.
        for p in [1, 3]:
            plt.setp(self.fig.axes[p].get_yticklabels(), visible=False)

        # Set axis labels.
        if xlabel is not None:
            ax3.set_xlabel(xlabel[0])
            ax4.set_xlabel(xlabel[1])
        if ylabel is not None:
            ax1.set_ylabel(ylabel[0])
            ax3.set_ylabel(ylabel[1])

        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.set_prop_cycle(
                'color',
                plt.get_cmap(palette[i])(np.linspace(0, 1, numcolors[i])))
        self.set_default_spacing()
        return ax1, ax2, ax3, ax4

    def surface(self, ts: Union[np.ndarray, list],
                tps: Union[np.ndarray, list], gf: Union[np.ndarray, list]
                ) -> None:
        """Given a 2d function gf(x,y): R^2 ->R with eventually different
        length of arrays, e.g. triagonal in x,y, sets up a single panel and
        plots the function surface.

        Parameters
        ----------
        ts : Union[np.ndarray, list]
            X coordinate

        tps : Union[np.ndarray, list]
            Y coordinate

        gf : Union[np.ndarray, list]
            Function gf(ts,tps) on z coordinate
        """

        X = ts
        Y = tps
        X, Y = np.meshgrid(ts, tps)
        Z = gf

        self.ax.zaxis.set_rotate_label(False)
        self.ax.zaxis.label.set_rotation(90)
        surf = self.ax.plot_surface(X,
                                    Y,
                                    Z,
                                    rstride=1,
                                    cstride=1,
                                    cmap=self.cmap,
                                    vmin=np.nanmin(Z),
                                    vmax=np.nanmax(Z))

        self.fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.07)

        # ################### set z axis and view angles ######################
        tmp_planes = self.ax.zaxis._PLANES
        self.ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], tmp_planes[0],
                                 tmp_planes[1], tmp_planes[4], tmp_planes[5])
        view_1 = (40, -135)
        # view_2 = (25, -45)
        init_view = view_1
        self.ax.view_init(*init_view)
        # #####################################################################

    def finalize_and_save(self, filename: str = 'plot') -> None:
        """Finalizes the figure and saves it to filename.pdf.

        Parameters
        ----------
        filename : str, optional
            file name of pdf, by default 'plot'
        """

        axes = self.fig.get_axes()
        for ax in axes:
            legend = ax.legend(loc='best',
                               fontsize=5.0,
                               fancybox=True,
                               framealpha=0.5)
            if legend is not None:
                legend.get_frame().set_linewidth(0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.margins(y=0.02)
            ax.margins(x=0.0)
            ax.ticklabel_format(style="scientific",
                                scilimits=(0, 0),
                                useOffset=False)
            plt.tight_layout()
        if self.fig.vertical_merged and not self.fig.horizontal_merged:
            # Remove last label from left axes.
            plt.setp([a.get_xticklabels()[-1] for a in self.fig.axes[:-1]],
                     visible=False)
        if self.fig.horizontal_merged and not self.fig.vertical_merged:
            # Remove last label from bottom axes.
            plt.setp([a.get_yticklabels()[-1] for a in self.fig.axes[1:]],
                     visible=False)
        if self.fig.horizontal_merged and self.fig.vertical_merged:
            plt.setp(self.fig.axes[2].get_yticklabels()[-1], visible=False)
            plt.setp(self.fig.axes[2].get_xticklabels()[-1], visible=False)
        bbox = None
        if self.projection == 'heatmap':
            bbox = 'tight'

        self.fig.savefig(filename, dpi=900, bbox_inches=bbox)
        plt.close(self.fig)
        # TODO: filename can not have ending .pdf, therefore can not be
        #       saved to pdf. Find out why and is png is sufficient.


# TODO: include the rest from figures in my directory
