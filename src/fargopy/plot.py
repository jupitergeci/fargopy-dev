###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import matplotlib.pyplot as plt
import numpy as np

###############################################################
# Constants
###############################################################


###############################################################
# Classes
###############################################################
class Plot(object):
    """Plotting utilities and visualization helpers for FARGO3D data.

    The ``Plot`` class encapsulates static methods for common plotting tasks,
    such as adding watermarks to figures and creating standardized heatmaps
    for simulation fields.
    """

    @staticmethod
    def fargopy_mark(ax, frac=1/6, alpha=0.5):
        """Add a watermark to a 2D or 3D plot.

        Places a rotated "FARGOpy {version}" watermark in the top-right corner
        of the specified axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the watermark will be added.

        Returns
        -------
        matplotlib.text.Text
            The created text object.

        Examples
        --------
        Add watermark to a plot:

        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> fp.Plot.fargopy_mark(ax)
        """
        ax.grid(False, which="both")
        try:
            ax.minorticks_off()
        except Exception:
            pass

        # Get the height of axe
        axh = (
            ax.get_window_extent()
            .transformed(ax.get_figure().dpi_scale_trans.inverted())
            .height
        )
        fig_factor = frac * axh

        # Options of the water mark
        args = dict(
            rotation=270,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="pink",
            alpha=alpha,
            fontsize=10 * fig_factor,
            zorder=100,
        )

        # Text of the water mark
        mark = f"FARGOpy {fargopy.__version__}"

        # Choose the according to the fact it is a 2d or 3d plot
        try:
            ax.add_collection3d
            plt_text = ax.text2D
        except:
            plt_text = ax.text

        text = plt_text(1, 1, mark, **args)
        return text


    @staticmethod
    def mesh(
        sim,
        snapshot=0,
        slice="theta=1.56",
        planet=0,
        draw_hill=True,
        hill_frac=1.0,
        figsize=(8, 8),
        point_size=1,
        line_alpha=0.5,
        cmap="viridis",
        show=True,
    ):
        """
        Plot the simulation mesh in the XY plane and (optionally) the planet Hill circle.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        snapshot : int, optional
            Snapshot to plot, by default 0.
        slice : str, optional
            Slice definition, by default 'theta=1.56'.
        planet : int or str, optional
            Planet index or name to focus, by default 0.
        draw_hill : bool, optional
            Whether to draw the Hill sphere, by default True.
        hill_frac : float, optional
            Fraction of Hill radius to draw, by default 1.0.
        figsize : tuple, optional
            Figure size, by default (8,8).
        point_size : int, optional
            Size of mesh points, by default 1.
        line_alpha : float, optional
            Alpha transparency of mesh lines, by default 0.5.
        cmap : str, optional
            Colormap for points, by default 'viridis'.
        show : bool, optional
            Whether to show the plot, by default True.

        Returns
        -------
        tuple
            (fig, ax, nr_celdas_radial, nr_celdas_azimutal, n_inside)
            Matplotlib figure and axes, max contiguous radial cells, max contiguous azimuthal cells,
            and the number of mesh cells inside the hill_frac * Hill radius.

        Examples
        --------
        >>> fp.Plot.mesh(sim, snapshot=0)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        # Load a 2D interpolated field (keeps same interface used elsewhere)
        gasdens = sim.load_field(
            fields=["gasdens"], snapshot=snapshot, slice=slice
        )

        # Expect interpolator result with var1_mesh / var2_mesh (as used in plot_interactive)
        try:
            X = gasdens.var1_mesh[0]
            Y = gasdens.var2_mesh[0]
        except Exception:
            # Fallback: if a raw Field-like object is returned with mesh names var1_mesh/var2_mesh attributes
            X = getattr(gasdens, "var1_mesh", None)
            Y = getattr(gasdens, "var2_mesh", None)
            if X is None or Y is None:
                raise RuntimeError(
                    "Could not obtain var1_mesh/var2_mesh from loaded field. Use a valid slice."
                )

        # Prepare figure
        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points (convert to AU for axis if simulation units defined)
        scale = getattr(sim, "UL", 1.0) / getattr(sim, "AU", 1.0)
        ax.scatter(
            (X * scale).ravel(),
            (Y * scale).ravel(),
            s=point_size,
            c=(X * 0 + 0.5).ravel(),
            cmap=cmap,
            marker=".",
            linewidths=0,
        )

        # If mesh is 2D arrays, draw grid lines
        if X.ndim == 2 and Y.ndim == 2:
            # rows
            for i in range(X.shape[0]):
                ax.plot(
                    X[i, :] * scale,
                    Y[i, :] * scale,
                    color="gray",
                    linewidth=0.5,
                    alpha=line_alpha,
                )
            # columns
            for j in range(X.shape[1]):
                ax.plot(
                    X[:, j] * scale,
                    Y[:, j] * scale,
                    color="gray",
                    linewidth=0.5,
                    alpha=line_alpha,
                )

        # Planet selection
        planets = sim.load_planets(snapshot=snapshot)
        center_x = center_y = None
        radius = 0.0
        if planets:
            sel = None
            if isinstance(planet, int):
                try:
                    sel = planets[planet]
                except Exception:
                    sel = planets[0]
            else:
                # name lookup
                for p in planets:
                    if getattr(p, "name", None) == planet:
                        sel = p
                        break
                if sel is None:
                    sel = planets[0]
            # planet object expected to have pos.x / pos.y and hill_radius property
            center_x = sel.pos.x
            center_y = sel.pos.y
            if draw_hill:
                radius = hill_frac * getattr(sel, "hill_radius", 0.0)

        # Draw Hill circle if requested and compute counts
        nr_celdas_radial = 0
        nr_celdas_azimutal = 0
        n_inside = 0
        if draw_hill and center_x is not None and center_y is not None and radius > 0:
            circle = patches.Circle(
                (center_x * scale, center_y * scale),
                radius * scale,
                edgecolor="red",
                facecolor="lightblue",
                linestyle="-",
                linewidth=1.5,
            )
            ax.add_patch(circle)

            # Count mesh cells (points) inside the requested fraction of Hill radius
            try:
                # X,Y are in simulation length units (same as center_x, center_y)
                mask_inside = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= (radius**2)
                n_inside = int(np.count_nonzero(mask_inside))

                # If mesh is structured 2D array, compute contiguous runs:
                if X.ndim == 2 and Y.ndim == 2:
                    # Helper to get max contiguous True length in a 1D boolean array
                    def max_contiguous_true(arr1d):
                        idx = np.flatnonzero(arr1d)
                        if idx.size == 0:
                            return 0
                        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                        lengths = [s.size for s in splits]
                        return max(lengths) if lengths else 0

                    # Azimutal: along rows (axis 1) -> for each row find longest contiguous True segment
                    max_az = 0
                    for i in range(mask_inside.shape[0]):
                        l = max_contiguous_true(mask_inside[i, :])
                        if l > max_az:
                            max_az = l
                    nr_celdas_azimutal = max_az

                    # Radial: along cols (axis 0) -> for each col find longest contiguous True segment
                    max_rad = 0
                    for j in range(mask_inside.shape[1]):
                        l = max_contiguous_true(mask_inside[:, j])
                        if l > max_rad:
                            max_rad = l
                    nr_celdas_radial = max_rad

            except Exception as e:
                print(f"Warning computing counts: {e}")

        fargopy.Plot.fargopy_mark(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")

        if show:
            plt.show()

        return fig, ax, nr_celdas_radial, nr_celdas_azimutal, n_inside
