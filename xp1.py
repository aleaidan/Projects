
import time
import os
import math
from pathlib import Path
from typing import Optional, List, Tuple

import arcade
from pythonosc.udp_client import SimpleUDPClient
from PIL import Image
import numpy as np
import subprocess
import tempfile

# Try to import imageio for video playback (more reliable than cv2)
try:
    import imageio
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("[WARNING] Video playback disabled.  pip install imageio[ffmpeg]")

# --- DEFINE ASSETS DIRECTORY HERE ---
# Resolve assets directory from a shortlist of common names so this works
BASE_DIR = Path(__file__).resolve().parent
_ASSET_CANDIDATES = ["media_assets", "Media_Assets", "media", "Media", "media assets", "Media Assets", "assets", "Assets", "Media Assets"]
ASSETS = None
for _n in _ASSET_CANDIDATES:
    _p = BASE_DIR / _n
    if _p.exists() and _p.is_dir():
        ASSETS = _p
        break
if ASSETS is None:
    ASSETS = BASE_DIR / "media"

def asset(name):
    return str(ASSETS / name)

print("RUNNING FILE:", __file__)
print("ASSET PATH IN CODE:", ASSETS)

# Toggle visuals
# Set to False to hide icon labels and the helper text at the bottom
SHOW_ICON_LABELS = False
SHOW_HELPER_TEXT = False
# Global icon size (pixels) — change this value to resize all desktop icons
ICON_SIZE = 98
# Global orbit speed (radians per second) used for icon orbits
ORBIT_SPEED = 0.2

# -----------------------
# Compatibility helpers
# -----------------------

def draw_texture_compat(cx, cy, w, h, texture):
    """
    Draw  texture at window center (cx, cy) with size (w, h).
    """
    # Guard against invalid dimensions
    if w <= 0 or h <= 0:
        return None
    
    try:
        # Arcade 3.x uses draw_texture_rect with a Rect object
        if hasattr(arcade, 'draw_texture_rect'):
            from arcade.types import Rect
            # Rect is a NamedTuple with: left, right, bottom, top, width, height, x, y
            left = cx - w/2
            right = cx + w/2
            bottom = cy - h/2
            top = cy + h/2
            # Create Rect with all 8 fields
            rect = Rect(left, right, bottom, top, w, h, cx, cy)
            return arcade.draw_texture_rect(texture, rect)
    except Exception as e:
        print(f"[ERROR] draw_texture_compat failed: {e}")
        import traceback
        traceback.print_exc()
    return None


def draw_lrtb_rectangle_filled(left, right, top, bottom, color):
    # Guard against invalid dimensions
    if left >= right or bottom >= top:
        return None
    
    if hasattr(arcade, 'draw_lrtb_rectangle_filled'):
        try:
            return arcade.draw_lrtb_rectangle_filled(left, right, top, bottom, color)
        except Exception:
            pass
    if hasattr(arcade, 'draw_lrbt_rectangle_filled'):
        try:
            return arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, color)
        except Exception:
            pass
    # fallback:
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    w = right - left
    h = top - bottom
    return arcade.draw_rectangle_filled(cx, cy, w, h, color)


def draw_lrtb_rectangle_outline(left, right, top, bottom, color, border_thickness=1):
    if hasattr(arcade, 'draw_lrtb_rectangle_outline'):
        try:
            return arcade.draw_lrtb_rectangle_outline(left, right, top, bottom, color, border_thickness)
        except Exception:
            pass
    if hasattr(arcade, 'draw_lrbt_rectangle_outline'):
        try:
            return arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, color, border_thickness)
        except Exception:
            pass
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    w = right - left
    h = top - bottom
    return arcade.draw_rectangle_outline(cx, cy, w, h, color, border_thickness)


def draw_rect_compat(cx, cy, w, h, color):
    # Guard against invalid dimensions
    if w <= 0 or h <= 0:
        return None
    
    if hasattr(arcade, 'draw_rectangle_filled'):
        try:
            return arcade.draw_rectangle_filled(cx, cy, w, h, color)
        except Exception:
            pass
    if hasattr(arcade, 'draw_rect_filled'):
        try:
            return arcade.draw_rect_filled(cx, cy, w, h, color)
        except Exception:
            pass
    left = cx - w/2
    right = cx + w/2
    top = cy + h/2
    bottom = cy - h/2
    return draw_lrtb_rectangle_filled(left, right, top, bottom, color)

# -----------------------
# Config & asset helper
# (assets directory resolved earlier in the file)
# -----------------------

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_TITLE = "XP Windowing Demo"
OSC_IP = "127.0.0.1"
OSC_PORT = 57120
OSC_ADDRESS = "/cursor"
NORMALIZE_OSC = True

# -----------------------
# UI constants
# -----------------------
TITLEBAR_HEIGHT = 30
BORDER_THICKNESS = 4
BUTTON_SIZE = 4
BUTTON_PADDING = 6
MIN_WINDOW_SIZE = (180, 120)
RESIZE_HANDLE_SIZE = 12

# Colors (XP-ish)
COLOR_TITLE = (0x00, 0x56, 0xa6)    # blue
COLOR_TITLE_TEXT = arcade.color.WHITE
COLOR_WINDOW_BG = (0xf0, 0xf0, 0xe0)
COLOR_BORDER = (0x00, 0x00, 0x00)
COLOR_BUTTON_CLOSE = (0xd9, 0x3b, 0x3b)  # red
COLOR_BUTTON_MIN = (0xff, 0xc7, 0x00)    # yellow
COLOR_BUTTON_MAX = (0x44, 0xc7, 0x44)    # green
COLOR_ICON_LABEL = arcade.color.BLACK

# -----------------------
# Small helper utils
# -----------------------
def point_in_rect(px, py, cx, cy, w, h):
    left = cx - w/2
    right = cx + w/2
    bottom = cy - h/2
    top = cy + h/2
    return left <= px <= right and bottom <= py <= top

# -----------------------
# Video Player class
# -----------------------
class VideoPlayer:
    """Simple video player using imageio for playback."""
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.reader = None
        self.current_frame_index = -1
        self.texture = None
        self.is_playing = False
        self.fps = 30
        self.total_frames = None
        self.start_time = None

        # Lazy-loading approach: don't preload all frames (avoids long startup pause).
        if VIDEO_AVAILABLE and os.path.exists(video_path):
            try:
                self.reader = imageio.get_reader(video_path)
                meta = self.reader.get_meta_data()
                self.fps = meta.get('fps', 30)
                # Some backends provide frame count; treat infinite as unknown
                self.total_frames = meta.get('nframes', None)
                try:
                    import math
                    if isinstance(self.total_frames, float) and math.isinf(self.total_frames):
                        self.total_frames = None
                except Exception:
                    pass
                self.is_playing = True
                print(f"[VIDEO] Opened video: {video_path} ({'{} frames'.format(self.total_frames) if self.total_frames else 'unknown frames'}) at {self.fps} fps")
            except Exception as e:
                print(f"[ERROR] VideoPlayer init failed for {video_path}: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_frame(self, frame_index: int):
        """Convert frame to arcade texture."""
        try:
            # Try random-access read first (may be supported by ffmpeg backend)
            frame = None
            # Ensure we use an integer frame index for reader and range
            idx = int(frame_index)
            try:
                frame = self.reader.get_data(idx)
            except Exception:
                # Fall back to iterating up to the requested frame (slower)
                it = iter(self.reader)
                for i in range(idx + 1):
                    frame = next(it)

            # Ensure frame is RGB numpy array
            if frame is None:
                return False
            if isinstance(frame, Image.Image):
                pil_img = frame.convert('RGB')
            else:
                arr = frame
                if len(arr.shape) == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                elif len(arr.shape) == 2:
                    arr = Image.fromarray(arr).convert('RGB')
                    arr = np.array(arr)
                pil_img = Image.fromarray(arr)

            # Convert PIL image to arcade texture (PNG in-memory)
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)
            self.texture = arcade.load_texture(buf)
            self.current_frame_index = frame_index
            return True
        except Exception as e:
            print(f"[ERROR] Failed to convert frame {frame_index} to texture: {e}")
            return False
    
    def update(self, delta_time: float):
        """Update video playback based on elapsed time."""
        if not self.is_playing:
            return

        # Initialize start time on first update so we use arcade's delta_time rhythm
        if self.start_time is None:
            self.start_time = time.time()

        # Compute which frame should be shown according to elapsed time
        elapsed = time.time() - self.start_time
        if self.fps <= 0:
            return
        frame_index = int(elapsed * self.fps)
        if self.total_frames:
            frame_index = frame_index % self.total_frames

        if frame_index != self.current_frame_index:
            self._load_frame(frame_index)
    
    def draw(self, cx: float, cy: float, w: float, h: float):
        """Draw the current video frame at the specified position."""
        if self.texture:
            try:
                draw_texture_compat(cx, cy, w, h, self.texture)
            except Exception as e:
                print(f"[ERROR] Failed to draw video frame: {e}")

    def _stop_audio(self):
        """No-op placeholder for audio cleanup (audio disabled)."""
        pass
    
    def close(self):
        """Release video resources."""
        # Stop audio first (safest)
        try:
            self._stop_audio()
        except Exception:
            pass
        
        # Close reader
        if self.reader:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        
        self.is_playing = False
        self.texture = None

# -----------------------
# Window class
# -----------------------
class XPWindow:
    def __init__(self, title: str, center_x: float, center_y: float, width: int, height: int, content=None, background_image: Optional[str] = None, custom_style: bool = False, video_path: Optional[str] = None):
        self.title = title
        self.center_x = center_x
        self.center_y = center_y
        self.width = max(width, MIN_WINDOW_SIZE[0])
        self.height = max(height, MIN_WINDOW_SIZE[1])
        # Enforce 4:3 aspect ratio on initialization
        self._enforce_aspect_ratio()
        self.content = content  # can be callable draw_content(window, x, y, w, h)
        self.is_dragging = False
        self.drag_offset = (0, 0)
        self.is_resizing = False
        self.resize_dir = None  # 'se', 'e', 's', etc.
        self.minimized = False
        self.maximized = False
        self.prev_geometry = None  # store (cx, cy, w, h) for un-maximize
        self.is_focused = False
        # Custom styling for popup windows
        self.custom_style = custom_style  # Use PNG background and simple close button
        self.background_image = None
        self.bg_texture = None
        if background_image and os.path.exists(background_image):
            try:
                self.bg_texture = arcade.load_texture(background_image)
                self.background_image = background_image
            except Exception as e:
                print(f"[ERROR] Failed to load background image {background_image}: {e}")
        # Video player for custom windows
        self.video_player = None
        if video_path and os.path.exists(video_path) and VIDEO_AVAILABLE:
            try:
                self.video_player = VideoPlayer(video_path)
            except Exception as e:
                print(f"[ERROR] Failed to create video player for {video_path}: {e}")
        # Close button position for custom windows (top-right corner)
        self.close_button_x = 0
        self.close_button_y = 0
        self._update_close_button_pos()
    
    def _enforce_aspect_ratio(self):
        """Maintain 4:3 aspect ratio by adjusting height based on width."""
        target_height = (self.width * 3) / 4
        if target_height < MIN_WINDOW_SIZE[1]:
            # If calculated height is too small, adjust width instead
            self.width = (MIN_WINDOW_SIZE[1] * 4) / 3
            self.height = MIN_WINDOW_SIZE[1]
        else:
            self.height = target_height
    
    def _update_close_button_pos(self):
        """Update close button position based on window location."""
        # Position close button in the top-right corner of the window with small padding
        # Use the same padding concept as other UI elements so it sits just inside the edge.
        padding = 0
        radius = 12
        # Place the close button flush against the top-right inner corner
        self.close_button_x = self.right() - padding - radius
        self.close_button_y = self.top() - padding - radius

    # Titlebar rectangle center and dims
    @property
    def title_center(self):
        return (self.center_x, self.center_y + (self.height/2 - TITLEBAR_HEIGHT/2))

    @property
    def title_rect(self):
        cx, cy = self.title_center
        return (cx, cy, self.width, TITLEBAR_HEIGHT)

    def left(self):
        return self.center_x - self.width/2

    def right(self):
        return self.center_x + self.width/2

    def top(self):
        return self.center_y + self.height/2

    def bottom(self):
        return self.center_y - self.height/2

    def contains_point(self, x: float, y: float) -> bool:
        return point_in_rect(x, y, self.center_x, self.center_y, self.width, self.height)

    def point_in_titlebar(self, x: float, y: float) -> bool:
        cx, cy, w, h = *self.title_center, self.width, TITLEBAR_HEIGHT
        return point_in_rect(x, y, cx, cy, w, h)

    def get_button_positions(self):
        # right side buttons: close, max, min
        right_x = self.right() - BUTTON_PADDING - BUTTON_SIZE/2
        y = self.top() - TITLEBAR_HEIGHT/2
        close_pos = (right_x, y)
        max_pos = (right_x - (BUTTON_SIZE + 4), y)
        min_pos = (right_x - 2*(BUTTON_SIZE + 4), y)
        return {'close': close_pos, 'max': max_pos, 'min': min_pos}

    def resize_handle_rect(self):
        # bottom-right corner handle
        hx = self.right() - RESIZE_HANDLE_SIZE/2
        hy = self.bottom() + RESIZE_HANDLE_SIZE/2
        return (hx, hy, RESIZE_HANDLE_SIZE, RESIZE_HANDLE_SIZE)

    def on_pointer_down(self, x: float, y: float):
        # returns a string action or None
        
        # Custom styled windows: simple close button
        if self.custom_style:
            # Check close button click (top-right corner)
            if point_in_rect(x, y, self.close_button_x, self.close_button_y, 24, 24):
                return "button_close"
            # Allow dragging from anywhere in window
            if self.contains_point(x, y):
                self.is_dragging = True
                self.drag_offset = (x - self.center_x, y - self.center_y)
                return "drag"
            return None
        
        if self.minimized:
            # clicking the titlebar of minimized window should restore it
            if self.point_in_titlebar(x, y):
                self.minimized = False
                return "focus"
            return None

        # buttons
        btns = self.get_button_positions()
        for name, pos in btns.items():
            bx, by = pos
            if point_in_rect(x, y, bx, by, BUTTON_SIZE, BUTTON_SIZE):
                return f"button_{name}"

        # title drag
        if self.point_in_titlebar(x, y):
            self.is_dragging = True
            self.drag_offset = (x - self.center_x, y - self.center_y)
            return "drag"

        # resize handle
        hx, hy, hw, hh = self.resize_handle_rect()
        if point_in_rect(x, y, hx, hy, hw, hh):
            self.is_resizing = True
            self.resize_dir = 'se'  # for now only se handle
            self.drag_offset = (self.right() - x, self.top() - y)
            return "resize"

        # content click: focus
        if self.contains_point(x, y):
            return "focus"

        return None

    def on_pointer_up(self, x: float, y: float):
        self.is_dragging = False
        self.is_resizing = False
        self.resize_dir = None
        # Ensure dimensions are valid after resize
        if self.width <= 0 or self.height <= 0:
            self.width = MIN_WINDOW_SIZE[0]
            self.height = MIN_WINDOW_SIZE[1]
        # Re-enforce aspect ratio after any resize
        self._enforce_aspect_ratio()

    def on_pointer_move(self, x: float, y: float, dx: float, dy: float):
        if self.is_dragging:
            # move window but clamp to screen
            new_cx = x - self.drag_offset[0]
            new_cy = y - self.drag_offset[1]
            half_w, half_h = self.width/2, self.height/2
            new_cx = max(half_w, min(new_cx, WINDOW_WIDTH - half_w))
            new_cy = max(half_h, min(new_cy, WINDOW_HEIGHT - half_h))
            self.center_x = new_cx
            self.center_y = new_cy
        elif self.is_resizing:
            # simple bottom-right resizing; compute new width/height with 4:3 aspect ratio
            new_right = x + self.drag_offset[0]
            new_width = max(MIN_WINDOW_SIZE[0], new_right - self.left())
            
            # Maintain 4:3 aspect ratio (width:height = 4:3)
            new_height = (new_width * 3) / 4
            new_height = max(MIN_WINDOW_SIZE[1], new_height)
            
            # If height hit minimum, adjust width to maintain ratio
            if new_height == MIN_WINDOW_SIZE[1]:
                new_width = (new_height * 4) / 3
            
            # Ensure dimensions are valid before assignment
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height

    def close(self):
        # placeholder for cleanup if needed
        pass

    def minimize(self):
        self.minimized = True

    def maximize(self):
        if not self.maximized:
            self.prev_geometry = (self.center_x, self.center_y, self.width, self.height)
            self.center_x = WINDOW_WIDTH / 2
            self.center_y = WINDOW_HEIGHT / 2
            self.width = WINDOW_WIDTH
            self.height = WINDOW_HEIGHT
            self.maximized = True
        else:
            # restore
            if self.prev_geometry:
                self.center_x, self.center_y, self.width, self.height = self.prev_geometry
            self.maximized = False

    def draw(self):
        # If minimized, draw only titlebar as a small strip at its current center_x
        if self.minimized:
            cx, cy = self.title_center
            draw_rect_compat(cx, cy, self.width, TITLEBAR_HEIGHT, COLOR_TITLE)
            arcade.draw_text(self.title, cx - self.width/2 + 6, cy - TITLEBAR_HEIGHT/2 + 6, COLOR_TITLE_TEXT, 12)
            return

        # Custom styled windows (with PNG background)
        if self.custom_style and self.bg_texture:
            try:
                # Draw background image
                draw_texture_compat(self.center_x, self.center_y, self.width, self.height, self.bg_texture)
                # Draw video on top of background if available
                if self.video_player:
                    video_w = self.width * 0.8  # 80% of window width
                    video_h = video_w * 3 / 4    # Maintain 4:3 ratio
                    self.video_player.draw(self.center_x, self.center_y, video_w, video_h)
                # Update close button position (keep hit area) but do not draw it (invisible)
                self._update_close_button_pos()
                # Intentionally not drawing the close button so it's invisible but still clickable
            except Exception as e:
                print(f"[ERROR] Failed to draw custom window: {e}")
                # Fallback to standard window
                self._draw_standard_window()
        else:
            # Standard XP-style window
            self._draw_standard_window()
    
    def _draw_standard_window(self):
        """Draw standard XP-style window with titlebar and buttons."""
        # Window background (use compatibility wrapper)
        draw_lrtb_rectangle_filled(self.left(), self.right(), self.top(), self.bottom(), COLOR_WINDOW_BG)
        # Border
        draw_lrtb_rectangle_outline(self.left(), self.right(), self.top(), self.bottom(), COLOR_BORDER, BORDER_THICKNESS)

        # Titlebar
        tcx, tcy = self.title_center
        draw_rect_compat(tcx, tcy, self.width, TITLEBAR_HEIGHT, COLOR_TITLE)
        arcade.draw_text(self.title, self.left() + 8, tcy - 8, COLOR_TITLE_TEXT, 12)

        # Buttons
        btns = self.get_button_positions()
        # Close
        bx, by = btns['close']
        arcade.draw_circle_filled(bx, by, BUTTON_SIZE/2, COLOR_BUTTON_CLOSE)
        arcade.draw_text("X", bx - 5, by - 6, arcade.color.WHITE, 12)
        # Max
        bx, by = btns['max']
        arcade.draw_circle_filled(bx, by, BUTTON_SIZE/2, COLOR_BUTTON_MAX)
        arcade.draw_text("□", bx - 6, by - 6, arcade.color.WHITE, 10)
        # Min
        bx, by = btns['min']
        arcade.draw_circle_filled(bx, by, BUTTON_SIZE/2, COLOR_BUTTON_MIN)
        arcade.draw_text("_", bx - 6, by - 7, arcade.color.WHITE, 16)

        # Resize handle (bottom-right)
        hx, hy, hw, hh = self.resize_handle_rect()
        draw_rect_compat(hx, hy, hw, hh, COLOR_BORDER)

        # Content area: call content draw function if present
        if self.content and not self.minimized:
            left = self.left()
            bottom = self.bottom()
            content_w = self.width
            content_h = self.height - TITLEBAR_HEIGHT
            content_left = left + 6
            content_bottom = bottom + 6
            content_w -= 12
            content_h -= TITLEBAR_HEIGHT + 12
            try:
                # content can be a callable or a mapping specifying type
                if callable(self.content):
                    self.content(self, content_left, content_bottom, content_w, content_h)
                else:
                    # backward-compat: if content is dict but content function assigned elsewhere
                    # try to detect 'type' and handle basic types
                    if isinstance(self.content, dict) and self.content.get('type') == 'seq':
                        content_image_sequence(self, content_left, content_bottom, content_w, content_h)
            except Exception as e:
                arcade.draw_text("Content error", self.center_x, self.center_y, arcade.color.RED, 16, anchor_x="center", anchor_y="center")
                print("Window content error:", e)

# -----------------------
# Example window content helpers
# -----------------------
def content_text(window: XPWindow, left: float, bottom: float, w: float, h: float):
    arcade.draw_text(
        f"Window: {window.title}\nSize: {int(w)}x{int(h)}",
        left + 10, bottom + h - 24,
        arcade.color.BLACK, 14
    )

def content_image(window: XPWindow, left: float, bottom: float, w: float, h: float):
    img_path = asset("fish.png")
    if os.path.exists(img_path):
        try:
            tex = arcade.load_texture(img_path)
            tw, th = tex.width, tex.height
            scale = min(w / tw, h / th, 1.0)
            draw_texture_compat(left + w/2, bottom + h/2, tw*scale, th*scale, tex)
        except Exception:
            arcade.draw_text("Error drawing image", left + 10, bottom + h/2, arcade.color.BLACK, 12)
    else:
        arcade.draw_text("No image (fish.png) in media assets", left + 10, bottom + h/2, arcade.color.BLACK, 12)

def content_image_sequence(window: XPWindow, left: float, bottom: float, w: float, h: float, seq=None):
    # Support lazy-loading frames onto the window (saved as window.content_seq)
    if not hasattr(window, "content_seq"):
        if isinstance(window.content, dict) and 'sequence_files' in window.content:
            files = window.content['sequence_files']
            seq_tex = []
            for f in files:
                if os.path.exists(f):
                    try:
                        seq_tex.append(arcade.load_texture(f))
                    except Exception:
                        pass
            window.content_seq = seq_tex
            window.content_seq_index = 0
            window.content_seq_time = 0.0
    seq_tex = getattr(window, "content_seq", None)
    if seq_tex:
        # Very simple frame advance based on constant dt approximation
        dt = 1/60.0
        window.content_seq_time += dt
        if window.content_seq_time > 1/15.0:
            window.content_seq_time = 0
            window.content_seq_index = (window.content_seq_index + 1) % len(seq_tex)
        tex = seq_tex[window.content_seq_index]
        scale = min(w / tex.width, h / tex.height, 1.0)
        draw_texture_compat(left + w/2, bottom + h/2, tex.width*scale, tex.height*scale, tex)
    else:
        arcade.draw_text("No frames", left + 10, bottom + h/2, arcade.color.BLACK, 12)

# -----------------------
# DesktopIcon now uses arcade.Sprite
# -----------------------
# -----------------------
# DesktopIcon now uses arcade.Sprite
# -----------------------
class DesktopIcon:
    def __init__(self, label: str, image_file: Optional[str], center_x: float, center_y: float, on_click=None,
                 orbit_center: Tuple[float, float] = None, orbit_radius: float = 0.0, orbit_speed: float = 0.0,
                 initial_angle: float = 0.0):
        self.size = ICON_SIZE  # Default icon size (pixels)
        self.label = label
        # base_center_* is the current draw center for the icon (updated by orbit)
        self.base_center_x = center_x
        self.base_center_y = center_y
        self.on_click = on_click
        # Orbit parameters (optional)
        self.orbit_center = orbit_center if orbit_center is not None else (center_x, center_y)
        self.orbit_radius = float(orbit_radius)
        self.orbit_speed = float(orbit_speed)
        self.angle = float(initial_angle)
        from PIL import Image
        import arcade
        import os
        import tempfile

        # Store the texture directly instead of a sprite
        self.texture = None
        self.sprite = None

        if image_file and os.path.exists(image_file):
            try:
                # Load with Pillow first (fixes P3, CMYK, palette issues)
                pil_img = Image.open(image_file).convert("RGBA")

                # Save normalized image to a temporary PNG file
                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)   # we only need the path, not the open file descriptor
                pil_img.save(temp_path, format="PNG")

                # Let Arcade load the cleaned texture from disk
                self.texture = arcade.load_texture(temp_path)

                # (Optional) delete the temp file later
                # os.remove(temp_path)

            except Exception as e:
                print(f"[ERROR] Failed to load image {label}: {image_file} → {e}")
                import traceback
                traceback.print_exc()
                self.texture = None

        # Create fallback sprite (solid gray) if no texture
        if self.texture is None:
            try:
                self.sprite = arcade.SpriteSolidColor(self.size, self.size, arcade.color.GRAY)
                self.sprite.center_x = self.base_center_x
                self.sprite.center_y = self.base_center_y - 8
            except Exception:
                self.sprite = arcade.Sprite()
                self.sprite.width = self.size
                self.sprite.height = self.size
                self.sprite.center_x = self.base_center_x
                self.sprite.center_y = self.base_center_y - 8

    def update(self, dt: float):
        """Advance orbit position by dt seconds if orbit is enabled."""
        if self.orbit_radius and self.orbit_speed:
            # advance angle
            self.angle += self.orbit_speed * dt
            cx, cy = self.orbit_center
            # match previous code's vertical sign convention (y inverted)
            self.base_center_x = int(cx + self.orbit_radius * math.cos(self.angle))
            self.base_center_y = int(cy - self.orbit_radius * math.sin(self.angle))
            # update fallback sprite position
            if self.sprite is not None:
                try:
                    self.sprite.center_x = self.base_center_x
                    self.sprite.center_y = self.base_center_y - 8
                except Exception:
                    pass

    def draw(self):
        try:
            if self.texture:
                # Use draw_texture_compat to draw the texture
                draw_texture_compat(self.base_center_x, self.base_center_y - 8, self.size, self.size, self.texture)
            else:
                # Fallback: draw a solid gray rectangle for the icon
                draw_rect_compat(self.base_center_x, self.base_center_y - 8, self.size, self.size, arcade.color.GRAY)
        except Exception as e:
            print(f"[ERROR] Failed to draw {self.label}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback fallback: draw a rectangle if all else fails
            draw_rect_compat(self.base_center_x, self.base_center_y - 8, self.size, self.size, arcade.color.GRAY)
        # Icon labels are optional; controlled by SHOW_ICON_LABELS flag
        try:
            if SHOW_ICON_LABELS:
                arcade.draw_text(self.label, self.base_center_x, self.base_center_y - 48, COLOR_ICON_LABEL, 12, anchor_x="center")
        except NameError:
            # If the flag is not defined for any reason, default to showing labels
            arcade.draw_text(self.label, self.base_center_x, self.base_center_y - 48, COLOR_ICON_LABEL, 12, anchor_x="center")

    def contains_point(self, x, y):
        # Simple rectangle hit test
        left = self.base_center_x - self.size / 2
        right = self.base_center_x + self.size / 2
        bottom = self.base_center_y - 8 - self.size / 2
        top = self.base_center_y - 8 + self.size / 2
        return left <= x <= right and bottom <= y <= top

# -----------------------
# Main Window (Arcade)
# -----------------------
class DesktopApp(arcade.Window):
    def __init__(self, width, height, title):
        # Don't make window resizable due to macOS/Pyglet issues
        # Use a fixed window size instead 
        super().__init__(width, height, title, resizable=False)
        # default wallpaper fallback color
        self.default_bg_color = (0x9b, 0xc8, 0xff)
        arcade.set_background_color(self.default_bg_color)

        # z-ordered list of windows (last is on top)
        self.windows: List[XPWindow] = []
        self.icons: List[DesktopIcon] = []
        self.focused_window: Optional[XPWindow] = None

        # mouse state
        self.mouse_x = width // 2
        self.mouse_y = height // 2

        # OSC (optional)
        try:
            self.osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
        except Exception:
            self.osc_client = None
        self.last_osc = 0.0

        # wallpaper texture (optional)
        self.wallpaper = None
        wallpaper_path = asset("wallpaper.png")
        if os.path.exists(wallpaper_path):
            try:
                self.wallpaper = arcade.load_texture(wallpaper_path)
            except Exception:
                self.wallpaper = None

        # create some icons
        self.setup_icons()

    def setup_icons(self):
        # Position icons initially near top-left, but enable orbit around the window center
        cx = 80
        cy = self.height - 80
        padding = 120
        # Orbit parameters: orbit around the current window center
        center = (self.width / 2, self.height / 2)
        orbit_radius = min(self.width, self.height) * 0.34
        # Create three icons spaced around the orbit
        fish_icon = DesktopIcon("Fish", asset("fish.png"), cx, cy, on_click=self.open_fish_window,
                    orbit_center=center, orbit_radius=orbit_radius, orbit_speed=ORBIT_SPEED, initial_angle=0.0)
        trash_icon = DesktopIcon("Trash", asset("trash_can.png"), cx + padding, cy, on_click=self.open_trash_window,
                     orbit_center=center, orbit_radius=orbit_radius, orbit_speed=ORBIT_SPEED, initial_angle=2.094)
        media_icon = DesktopIcon("Media", asset("media_player.png"), cx + padding*2, cy, on_click=self.open_media_window,
                     orbit_center=center, orbit_radius=orbit_radius, orbit_speed=ORBIT_SPEED, initial_angle=4.188)
        self.icons = [fish_icon, trash_icon, media_icon]

    # Icon actions
    def open_fish_window(self):
        bg_path = asset("fish 4-3 window.png")
        video_path = asset("videoplayback.mp4")
        center_x = self.width / 2
        center_y = self.height / 2
        w = XPWindow("Fish", center_x, center_y, 320, 240, 
                     content=None, background_image=bg_path, custom_style=True, video_path=video_path)
        self.add_window(w)

    def open_trash_window(self):
        bg_path = asset("trashpopupbackground.png")
        video_path = asset("Live_at_the_Landfill.mp4")
        center_x = self.width / 2
        center_y = self.height / 2
        w = XPWindow("Trash", center_x, center_y, 320, 240, 
                     content=None, background_image=bg_path, custom_style=True, video_path=video_path)
        self.add_window(w)

    def open_media_window(self):
        bg_path = asset("dancebackground.png")
        video_path = asset("evolution of dance.mp4")
        center_x = self.width / 2
        center_y = self.height / 2
        w = XPWindow("Media Player", center_x, center_y, 320, 240, 
                     content=None, background_image=bg_path, custom_style=True, video_path=video_path)
        self.add_window(w)

    def add_window(self, window: XPWindow):
        # Clamp window position to ensure it stays fully within screen bounds
        half_w, half_h = window.width / 2, window.height / 2
        window.center_x = max(half_w, min(window.center_x, self.width - half_w))
        window.center_y = max(half_h, min(window.center_y, self.height - half_h))
        window._update_close_button_pos()
        
        self.windows.append(window)
        self.bring_to_front(window)

    def set_icon_size(self, new_size: int):
        """Update global ICON_SIZE and apply to existing icons.

        This updates both texture-based icons and fallback solid-color sprites.
        """
        global ICON_SIZE
        ICON_SIZE = int(new_size)
        for icon in self.icons:
            icon.size = ICON_SIZE
            # Update fallback sprite size if present
            if icon.sprite is not None:
                try:
                    icon.sprite.width = ICON_SIZE
                    icon.sprite.height = ICON_SIZE
                    # keep sprite centered vertically similar to constructor
                    icon.sprite.center_x = icon.base_center_x
                    icon.sprite.center_y = icon.base_center_y - 8
                except Exception:
                    pass

    def bring_to_front(self, window: XPWindow):
        if window in self.windows:
            self.windows.remove(window)
            self.windows.append(window)
        self.focused_window = window
        for w in self.windows:
            w.is_focused = (w is window)

    # -----------------
    # Arcade event handlers
    # -----------------
    def draw_wallpaper_fullscreen(self):
        if self.wallpaper:
            try:
                # Use draw_texture_compat to draw the wallpaper
                tw, th = self.wallpaper.width, self.wallpaper.height
                scale = max(self.width / tw, self.height / th)
                draw_w, draw_h = tw * scale, th * scale
                draw_texture_compat(self.width/2, self.height/2, draw_w, draw_h, self.wallpaper)
            except Exception as e:
                print(f"[WARNING] Failed to draw wallpaper: {e}")
                # If any failure, fall back to solid color background
                arcade.set_background_color(self.default_bg_color)
        else:
            # ensure background color is set
            arcade.set_background_color(self.default_bg_color)

    def on_draw(self):
        # Clear the window / start render
        if hasattr(self, 'clear'):
            try:
                self.clear()
            except Exception:
                pass
        elif hasattr(arcade, 'clear'):
            try:
                arcade.clear()
            except Exception:
                pass
        elif hasattr(arcade, 'start_render'):
            try:
                arcade.start_render()
            except Exception:
                pass

        # Draw wallpaper (image if available)
        self.draw_wallpaper_fullscreen()

        # draw desktop icons
        for icon in self.icons:
            icon.draw()

        # draw windows in z-order
        for w in self.windows:
            w.draw()

        # draw some helper text (optional)
        try:
            if SHOW_HELPER_TEXT:
                arcade.draw_text("XP-style Windowing Demo — Click icons to open windows. Drag titlebar. Resize via bottom-right handle.",
                                 10, 10, arcade.color.BLACK, 12)
        except NameError:
            # If the flag isn't defined, default to showing the helper text
            arcade.draw_text("XP-style Windowing Demo — Click icons to open windows. Drag titlebar. Resize via bottom-right handle.",
                             10, 10, arcade.color.BLACK, 12)

    def on_update(self, delta_time: float):
        now = time.time()
        # update icon orbits
        try:
            for icon in self.icons:
                try:
                    icon.update(delta_time)
                except Exception:
                    pass
        except Exception:
            pass
        # update video playback in windows
        try:
            for w in self.windows:
                if w.video_player:
                    try:
                        w.video_player.update(delta_time)
                    except Exception:
                        pass
        except Exception:
            pass
        if now - self.last_osc > 1/60:
            try:
                if self.osc_client:
                    if NORMALIZE_OSC:
                        self.osc_client.send_message(OSC_ADDRESS, [self.mouse_x / float(self.width), self.mouse_y / float(self.height)])
                    else:
                        self.osc_client.send_message(OSC_ADDRESS, [self.mouse_x, self.mouse_y])
                self.last_osc = now
            except Exception:
                # ignore OSC issues
                pass

    def find_top_window_at(self, x, y) -> Optional[XPWindow]:
        for w in reversed(self.windows):
            if w.contains_point(x, y) or w.point_in_titlebar(x, y) or point_in_rect(x, y, *w.resize_handle_rect()):
                return w
        return None

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        self.mouse_x = int(x)
        self.mouse_y = int(y)
        if self.focused_window and (self.focused_window.is_dragging or self.focused_window.is_resizing):
            self.focused_window.on_pointer_move(x, y, dx, dy)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        self.mouse_x = int(x)
        self.mouse_y = int(y)

        # Send an immediate OSC trigger on left-click (maps from previous Map19 behavior)
        try:
            if button == getattr(arcade, 'MOUSE_BUTTON_LEFT', 1) and self.osc_client:
                # Send cursor message (normalized or raw)
                if NORMALIZE_OSC:
                    self.osc_client.send_message(OSC_ADDRESS, [self.mouse_x / float(self.width), self.mouse_y / float(self.height)])
                else:
                    self.osc_client.send_message(OSC_ADDRESS, [self.mouse_x, self.mouse_y])

                # Also trigger the SFX message used in Map 19
                try:
                    self.osc_client.send_message('/play_sfx', [])
                except Exception:
                    # ignore SFX send failures
                    pass
        except Exception:
            # Don't allow OSC issues to interrupt UI
            pass

        # First, check if any window is covering this point
        # If so, don't allow icon clicks
        window_covering = self.find_top_window_at(x, y)
        
        # Check desktop icons only if no window is covering them
        if not window_covering:
            for icon in self.icons:
                if icon.contains_point(x, y):
                    if icon.on_click:
                        icon.on_click()
                    return

        # find window under pointer (topmost)
        target = window_covering
        if target:
            action = target.on_pointer_down(x, y)
            # bring to top if we clicked inside (or titlebar)
            self.bring_to_front(target)
            # handle button actions
            if isinstance(action, str) and action.startswith("button_"):
                name = action.split("_", 1)[1]
                if name == "close":
                    if target in self.windows:
                        # Clean up video resources before removing window
                        if target.video_player:
                            try:
                                target.video_player.close()
                            except Exception:
                                pass
                        self.windows.remove(target)
                        if self.windows:
                            self.focused_window = self.windows[-1]
                        else:
                            self.focused_window = None
                elif name == "min":
                    target.minimize()
                elif name == "max":
                    target.maximize()
            return
        else:
            # Clicked empty desktop: unfocus windows
            self.focused_window = None
            for w in self.windows:
                w.is_focused = False

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        self.mouse_x = int(x)
        self.mouse_y = int(y)
        if self.focused_window:
            self.focused_window.on_pointer_up(x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.ESCAPE:
            if self.focused_window:
                if self.focused_window.minimized:
                    self.focused_window.minimized = False
                else:
                    if self.focused_window in self.windows:
                        self.windows.remove(self.focused_window)
                    self.focused_window = self.windows[-1] if self.windows else None
            else:
                try:
                    arcade.close_window()
                except Exception:
                    pass
        # Increase / decrease icon size at runtime: I = increase, K = decrease
        elif symbol == arcade.key.I:
            try:
                # bump by 8 pixels
                self.set_icon_size(ICON_SIZE + 8)
                print(f"ICON_SIZE set to {ICON_SIZE}")
            except Exception:
                pass
        elif symbol == arcade.key.K:
            try:
                self.set_icon_size(max(8, ICON_SIZE - 8))
                print(f"ICON_SIZE set to {ICON_SIZE}")
            except Exception:
                pass

    def on_resize(self, width: int, height: int):
        try:
            # Guard against invalid resize dimensions
            if width < MIN_WINDOW_SIZE[0] or height < MIN_WINDOW_SIZE[1]:
                return
            
            # keep windows and icons within bounds when the desktop resizes
            for w in self.windows:
                try:
                    half_w, half_h = max(w.width/2, 0), max(w.height/2, 0)
                    # Ensure window stays visible on screen
                    w.center_x = max(half_w, min(w.center_x, width - half_w))
                    w.center_y = max(half_h, min(w.center_y, height - half_h))
                except Exception as e:
                    print(f"[ERROR] Failed to update window on resize: {e}")
                    
            # reposition icons vertically so they stay near top margin
            # keep their base x but recompute base y relative to new height
            # (simple behavior; you can refine as needed)
            top_margin = 80
            for i, icon in enumerate(self.icons):
                try:
                    icon.base_center_y = max(top_margin, height - top_margin)
                    if icon.sprite:
                        icon.sprite.center_y = icon.base_center_y - 8
                except Exception as e:
                    print(f"[ERROR] Failed to update icon on resize: {e}")
        except Exception as e:
            print(f"[ERROR] on_resize failed: {e}")
            import traceback
            traceback.print_exc()

# -----------------------
# Run
# -----------------------
def main():
    app = DesktopApp(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()
