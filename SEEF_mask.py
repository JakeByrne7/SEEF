# masking_tool.py
"""
Interactive ellipse masking tool reworked to run as an importable function.
Call run_masking_tool(image_data_out, id_no, vmin, vmax, crop=True, hdulist=None)
Returns modified image_data_out (same array object, possibly modified in-place).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, EllipseSelector
from matplotlib.patches import Ellipse as MplEllipse
from scipy.ndimage import rotate
from astropy import units as u
from regions import PixCoord, EllipsePixelRegion


def run_masking_tool(image_data_out, id_no, vmin=None, vmax=None, crop=True, hdulist=None):
    """
    Launch interactive ellipse masking tool.

    Parameters
    ----------
    image_data_out : ndarray
        Array-like with shape (N, Y, X) or similar where image_data_out[id_no] is the image.
    id_no : int
        Index of the image to operate on.
    vmin, vmax : float or None
        Display limits passed to imshow.
    crop : bool
        If True, create the EllipseSelector and buttons (keeps the original behaviour).
    hdulist : astropy.io.fits.HDUList or None
        Optional: if provided the function will close it at the end (same as original script).

    Returns
    -------
    image_data_out : ndarray
        The (possibly modified) array (modified in-place).
    """

    # ---- local copies / state ----
    original_data = np.copy(image_data_out[id_no])  # Keep the original for reference

    # figure + image
    fig, ax = plt.subplots()
    im = ax.imshow(image_data_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    text = ax.text(0.98, 0.98, '', size='small', color='yellow',
                   ha='right', va='top', transform=ax.transAxes)
    plt.title("Interactive Ellipse Placement")

    # state variables (these replace globals from the single-file version)
    ellipses = []                # store matplotlib ellipse patches
    ellipse_regions = []         # store tuples (x_center, y_center, width, height, angle)
    current_mode = "inner"       # "inner", "outer", "ring"
    pending_ellipse = None       # for ring mode selection
    rotation_angle_mask = 0      # cumulative rotation applied to image_data_out[id_no]

    # (kept from original - optional helpers/settings)
    angles_to_plot = [i for i in range(360) if i % 20 == 0]
    biny_n = "yes"
    limits = True
    lines = True
    xborder, yborder = 50, 50
    plot_lines = "Yes"
    binned_outlier_removal = "Yes"

    # ---- Draggable ellipse helper class (captures local lists via closure) ----
    class DraggableEllipse:
        """A class to handle dragging and resizing of an ellipse (inner scope)."""
        def __init__(self, ellipse_patch, idx):
            self.ellipse = ellipse_patch
            self.idx = idx
            self.press = None

        def connect(self):
            self.cidpress = self.ellipse.figure.canvas.mpl_connect('button_press_event', self.on_press)
            self.cidrelease = self.ellipse.figure.canvas.mpl_connect('button_release_event', self.on_release)
            self.cidmotion = self.ellipse.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        def on_press(self, event):
            if event.inaxes != self.ellipse.axes:
                return
            contains, _ = self.ellipse.contains(event)
            if not contains:
                return
            self.press = (self.ellipse.center, event.xdata, event.ydata)

        def on_release(self, event):
            self.press = None
            self.ellipse.figure.canvas.draw()

        def on_motion(self, event):
            if self.press is None or event.inaxes != self.ellipse.axes:
                return
            center, xpress, ypress = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            new_center = (center[0] + dx, center[1] + dy)
            self.ellipse.center = new_center
            # update the stored region entry (width, height unchanged; angle set to 0 in original)
            ellipse_regions[self.idx] = (*new_center, self.ellipse.width, self.ellipse.height, 0)
            self.ellipse.figure.canvas.draw()

        def disconnect(self):
            self.ellipse.figure.canvas.mpl_disconnect(self.cidpress)
            self.ellipse.figure.canvas.mpl_disconnect(self.cidrelease)
            self.ellipse.figure.canvas.mpl_disconnect(self.cidmotion)

    draggable_ellipses = []  # list of DraggableEllipse instances

    # ---- helper functions (use local closure variables) ----
    def create_ellipse(region_params, color, idx):
        """Create and draw a moveable ellipse; returns DraggableEllipse instance."""
        x_center, y_center, width, height, angle = region_params
        ellipse_patch = MplEllipse((x_center, y_center), width, height, angle=angle,
                                  edgecolor=color, facecolor='none', lw=2)
        ellipse = ax.add_patch(ellipse_patch)
        draggable = DraggableEllipse(ellipse, idx)
        draggable.connect()
        ellipses.append(ellipse)
        draggable_ellipses.append(draggable)
        plt.draw()
        return draggable

    def clear_ellipses():
        """Remove and disconnect all ellipses."""
        for draggable in list(draggable_ellipses):
            try:
                draggable.disconnect()
            except Exception:
                pass
        for e in list(ellipses):
            try:
                e.remove()
            except Exception:
                pass
        ellipses.clear()
        draggable_ellipses.clear()
        ellipse_regions.clear()
        plt.draw()

    def reset_image(event=None):
        """Reset image to original and clear ellipses."""
        nonlocal image_data_out
        clear_ellipses()
        image_data_out[id_no] = np.copy(original_data)
        im.set_data(image_data_out[id_no])
        text.set_text("Image reset to its original state.")
        plt.draw()

    # Masking functions using regions
    def mask_inside(region_params):
        x_center, y_center, width, height, angle = region_params
        region = EllipsePixelRegion(center=PixCoord(x=x_center, y=y_center),
                                    width=width, height=height, angle=angle * u.deg)
        mask = region.to_mask(mode='center')
        mask_image = mask.to_image(image_data_out[id_no].shape)
        image_data_out[id_no][mask_image == 1] = 0

    def mask_outside(region_params):
        x_center, y_center, width, height, angle = region_params
        region = EllipsePixelRegion(center=PixCoord(x=x_center, y=y_center),
                                    width=width, height=height, angle=angle * u.deg)
        mask = region.to_mask(mode='center')
        mask_image = mask.to_image(image_data_out[id_no].shape)
        outside_mask = np.ones_like(image_data_out[id_no], dtype=bool)
        outside_mask[mask_image == 1] = False
        image_data_out[id_no][outside_mask] = 0

    def mask_ring(inner_params, outer_params):
        # inner
        x_center, y_center, width, height, angle = inner_params
        inner_region = EllipsePixelRegion(center=PixCoord(x=x_center, y=y_center),
                                          width=width, height=height, angle=angle * u.deg)
        # outer
        x_center, y_center, width, height, angle = outer_params
        outer_region = EllipsePixelRegion(center=PixCoord(x=x_center, y=y_center),
                                          width=width, height=height, angle=angle * u.deg)
        inner_mask = inner_region.to_mask(mode='center').to_image(image_data_out[id_no].shape)
        outer_mask = outer_region.to_mask(mode='center').to_image(image_data_out[id_no].shape)
        ring_mask = (outer_mask == 1) & (inner_mask != 1)
        image_data_out[id_no][ring_mask] = 0

    # ---- Callbacks (these must be inner definitions so they capture closure) ----
    def on_select(eclick, erelease):
        nonlocal pending_ellipse
        # compute bounding box, center, dims
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        angle = 0  # unchanged behaviour - no rotation via selector

        if current_mode in ["inner", "outer"]:
            if ellipses:
                clear_ellipses()
            ellipse_regions.append((x_center, y_center, width, height, angle))
            create_ellipse(ellipse_regions[-1], color="yellow", idx=len(ellipse_regions) - 1)
            text.set_text(f"Ellipse placed for {current_mode.capitalize()} mode.")
        elif current_mode == "ring":
            pending_ellipse = (x_center, y_center, width, height, angle)
            text.set_text("Ellipse dimensions selected. Click 'Place Ellipse' to confirm.")
        plt.draw()

    def apply_mask(event):
        """Apply the mask based on current_mode and ellipse_regions (same checks as original)."""
        if current_mode in ["inner", "outer"] and len(ellipse_regions) != 1:
            text.set_text("Place exactly one ellipse for this mode.")
            return
        if current_mode == "ring" and len(ellipse_regions) != 2:
            text.set_text("Place exactly two ellipses for ring mode.")
            return

        if current_mode == "inner":
            mask_inside(ellipse_regions[0])
            text.set_text("Masked inside the ellipse.")
        elif current_mode == "outer":
            mask_outside(ellipse_regions[0])
            text.set_text("Masked outside the ellipse.")
        elif current_mode == "ring":
            mask_ring(ellipse_regions[0], ellipse_regions[1])
            text.set_text("Masked between the ellipses (ring).")

        im.set_data(image_data_out[id_no])
        plt.draw()

    def toggle_mode(event):
        nonlocal current_mode
        modes = ["inner", "outer", "ring"]
        current_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
        clear_ellipses()
        text.set_text(f"Mode switched to: {current_mode.capitalize()}")
        plt.draw()

    def place_ellipse(event):
        nonlocal pending_ellipse
        if current_mode != "ring":
            text.set_text("This button is only available in Ring mode.")
            return
        if pending_ellipse is None:
            text.set_text("No ellipse dimensions selected. Use the selector first.")
            return
        ellipse_regions.append(pending_ellipse)
        color = "yellow" if len(ellipse_regions) == 1 else "red"
        create_ellipse(pending_ellipse, color, idx=len(ellipse_regions) - 1)
        text.set_text(f"Ellipse {len(ellipse_regions)} placed for Ring mode.")
        pending_ellipse = None
        if len(ellipse_regions) > 2:
            text.set_text("Maximum of 2 ellipses allowed for Ring mode.")
        plt.draw()

    def rotate_clockwise(event):
        nonlocal rotation_angle_mask
        rotation_angle_mask += 5
        image_data_out[id_no] = rotate(image_data_out[id_no], 5, reshape=False)
        im.set_data(image_data_out[id_no])
        plt.draw()

    def rotate_counterclockwise(event):
        nonlocal rotation_angle_mask
        rotation_angle_mask -= 5
        image_data_out[id_no] = rotate(image_data_out[id_no], -5, reshape=False)
        im.set_data(image_data_out[id_no])
        plt.draw()

    def zoom(factor):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * factor / 2
        y_range = (ylim[1] - ylim[0]) * factor / 2
        ax.set_xlim([x_center - x_range, x_center + x_range])
        ax.set_ylim([y_center - y_range, y_center + y_range])
        plt.draw()

    def zoom_in(event):
        zoom(0.8)

    def zoom_out(event):
        zoom(1.25)

    def finalize(event):
        nonlocal rotation_angle_mask
        # rotate back (same behaviour as original)
        image_data_out[id_no] = rotate(image_data_out[id_no], -rotation_angle_mask, reshape=False)
        plt.close(fig)

    # ---- UI wiring ----
    if crop:
        selector = EllipseSelector(ax, on_select, interactive=True)

        # buttons layout (same positions / sizes your original used)
        button_width = 0.09
        button_height = 0.045
        button_y = 0.01
        positions = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89]

        ax_apply = plt.axes([positions[0], button_y, button_width, button_height])
        button_apply = Button(ax_apply, "Apply Mask")
        button_apply.on_clicked(apply_mask)

        ax_place_ellipse = plt.axes([positions[1], button_y, button_width, button_height])
        button_place_ellipse = Button(ax_place_ellipse, "Place Ellipse")
        button_place_ellipse.on_clicked(place_ellipse)

        ax_toggle = plt.axes([positions[2], button_y, button_width, button_height])
        button_toggle = Button(ax_toggle, "Toggle Mode")
        button_toggle.on_clicked(toggle_mode)

        ax_finalize = plt.axes([positions[3], button_y, button_width, button_height])
        button_finalize = Button(ax_finalize, "Finalize")
        button_finalize.on_clicked(finalize)

        ax_reset = plt.axes([positions[4], button_y, button_width, button_height])
        button_reset = Button(ax_reset, "Reset Image")
        button_reset.on_clicked(reset_image)

        ax_rotate_clockwise = plt.axes([positions[5], button_y, button_width, button_height])
        button_rotate_clockwise = Button(ax_rotate_clockwise, "Rotate +5°")
        button_rotate_clockwise.on_clicked(rotate_clockwise)

        ax_rotate_counterclockwise = plt.axes([positions[6], button_y, button_width, button_height])
        button_rotate_counterclockwise = Button(ax_rotate_counterclockwise, "Rotate -5°")
        button_rotate_counterclockwise.on_clicked(rotate_counterclockwise)

        ax_zoom_in = plt.axes([positions[7], button_y, button_width, button_height])
        button_zoom_in = Button(ax_zoom_in, "Zoom In")
        button_zoom_in.on_clicked(zoom_in)

        ax_zoom_out = plt.axes([positions[8], button_y, button_width, button_height])
        button_zoom_out = Button(ax_zoom_out, "Zoom Out")
        button_zoom_out.on_clicked(zoom_out)

        # show interactive UI
        plt.show()

        # final display as in original script
        plt.imshow(image_data_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
        plt.title("Final Masked Image")
        plt.xlabel("X axis pixels")
        plt.ylabel("Y axis pixels")
        plt.colorbar()
        plt.show()

        # close the fits handle only if provided (keeps original behaviour if hdulist was used)
        if hdulist is not None:
            try:
                hdulist.close()
            except Exception:
                pass

    return image_data_out


# quick standalone test if run directly (keeps the same signature)
if __name__ == "__main__":
    import numpy as _np
    dummy = _np.random.random((1, 200, 200))
    run_masking_tool(dummy, id_no=0, vmin=0, vmax=1, crop=True)
