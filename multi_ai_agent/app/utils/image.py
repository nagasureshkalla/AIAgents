"""Image validation and loading."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import SecurityConfig
from app.core.exceptions import InvalidImageError

_FORMAT_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "BMP": "bmp", "GIF": "gif", "TIFF": "tiff"}


def _check_header(img: Image.Image, security: SecurityConfig) -> str:
    fmt = (img.format or "").upper()
    if fmt not in security.allowed_image_formats:
        raise InvalidImageError(
            f"Unsupported image format '{fmt or 'unknown'}'. Allowed: {', '.join(security.allowed_image_formats)}"
        )
    width, height = img.size
    if width <= 0 or height <= 0:
        raise InvalidImageError("Image has invalid dimensions")
    if width * height > security.max_image_pixels:
        raise InvalidImageError(
            f"Image too large: {width}x{height} exceeds {security.max_image_pixels} pixels"
        )
    return fmt


def validate_image_bytes(data: bytes, security: SecurityConfig) -> str:
    """Validate uploaded bytes. Returns the detected format."""
    if not data:
        raise InvalidImageError("Empty image upload")
    if len(data) > security.max_image_mb * 1024 * 1024:
        raise InvalidImageError(f"Image exceeds {security.max_image_mb} MB")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = _check_header(img, security)
            img.verify()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Invalid or corrupted image: {exc}") from exc
    return fmt


def validate_image_file(path: Path, security: SecurityConfig) -> str:
    if not path.is_file():
        raise InvalidImageError(f"Image not found: {path.name}")
    if path.stat().st_size > security.max_image_mb * 1024 * 1024:
        raise InvalidImageError(f"Image exceeds {security.max_image_mb} MB")
    return validate_image_bytes(path.read_bytes(), security)


def extension_for(fmt: str) -> str:
    return _FORMAT_EXTENSIONS.get(fmt.upper(), "img")


def load_image(path: str | Path, security: SecurityConfig) -> Image.Image:
    """Load an RGB image ready for a vision processor, downscaled to `max_image_side`."""
    path = Path(path)
    validate_image_file(path, security)
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if getattr(img, "is_animated", False):
                img.seek(0)
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    max_side = security.max_image_side
    if max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return rgb
