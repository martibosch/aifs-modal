"""Plotting utils."""

from collections.abc import Mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure


def make_cartopy_plot(
    da: xr.DataArray,
    projection: ccrs.Projection | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    **kwargs: object,
) -> Figure:
    """
    Plot a 2D field on a Cartopy map.

    Parameters
    ----------
    da : xr.DataArray
        Data to plot. It should have `lat` and `lon` coordinates.
    projection : ccrs.Projection, optional
        Cartopy projection used for the target axes. Defaults to ``ccrs.PlateCarree()``.
    cmap : str, optional
        Colormap passed to ``xarray.DataArray.plot``. Defaults to "viridis"
    title : str | None, optional
        Plot title, optional.
    **kwargs : object
        Additional keyword arguments passed to ``xarray.DataArray.plot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if projection is None:
        projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection=projection)

    data_transform = ccrs.PlateCarree()

    da.plot(
        ax=ax,
        transform=data_transform,
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={"shrink": 0.7, "orientation": "horizontal", "pad": 0.05},
        **kwargs,
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title("" if title is None else title)

    return fig


def make_cartopy_facet_plot(
    da: xr.DataArray,
    col: str = "lead_time",
    *,
    projection: ccrs.Projection | None = None,
    cmap: str = "viridis",
    col_wrap: int = 4,
    cbar_kwargs: Mapping[str, object] | None = None,
    **kwargs,
):
    """
    Create faceted cartopy plots of an xarray.DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Data to plot. It must contain the provided `facet_coord`, plus `lat` and `lon`.
    col : str, optional
        Coordinate/dimension used for faceting. Defaults to "lead_time".
    projection : ccrs.Projection | None, optional
        Cartopy projection for each facet. Defaults to ``ccrs.PlateCarree()``.
    cmap : str, optional
        Colormap to use in xarray plotting.
    col_wrap : int, optional
        Number of columns before wrapping to the next row.
    cbar_kwargs : kwargs-like, optional
        Colorbar kwargs passed to xarray plotting.
    **kwargs
        Additional keyword arguments passed to `da.plot`.
    """
    if col not in da.coords and col not in da.dims:
        msg = (
            f"facet_coord {col!r} not found in DataArray coords/dims: {tuple(da.dims)}"
        )
        raise ValueError(msg)

    if projection is None:
        projection = ccrs.PlateCarree()
    data_transform = ccrs.PlateCarree()

    g = da.plot(
        col=col,
        col_wrap=col_wrap,
        subplot_kws={"projection": projection},
        transform=data_transform,
        cmap=cmap,
        cbar_kwargs=cbar_kwargs,
        **kwargs,
    )

    for ax in g.axs.flat:
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")

    return g
