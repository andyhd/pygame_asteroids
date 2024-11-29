from typing import NamedTuple
from typing import cast
from typing import overload

import pygame


class Circle(NamedTuple):
    """A circle shape."""

    radius: float = 1.0
    center: pygame.Vector2 = pygame.Vector2()


class Polygon(NamedTuple):
    """A polygon shape."""

    points: list[tuple[float, float]]
    center: pygame.Vector2 = pygame.Vector2()


class ComplexShape(NamedTuple):
    """A complex shape."""

    parts: list[Polygon]
    center: pygame.Vector2 = pygame.Vector2()


Shape = Circle | Polygon | ComplexShape


@overload
def transform(
    obj: Circle,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
) -> Circle: ...
@overload
def transform(
    obj: Polygon,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
) -> Polygon: ...
@overload
def transform(
    obj: ComplexShape,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
) -> ComplexShape: ...
@overload
def transform(
    obj: tuple,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
) -> tuple: ...
@overload
def transform(
    obj: float,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
) -> float: ...
def transform(
    obj,
    translate: pygame.Vector2 | None = None,
    scale: float = 1.0,
    rotate: float = 0.0,
):
    """Transform a shape."""
    if translate is None:
        translate = pygame.Vector2()

    match obj:
        case ComplexShape(parts, center):
            return ComplexShape(
                [cast(Polygon, transform(p, translate, scale, rotate)) for p in parts],
                center + translate,
            )
        case Polygon(points, center):
            return Polygon(
                [(r * scale, phi + rotate) for r, phi in points],
                center + translate,
            )
        case Circle(radius, center):
            return Circle(radius * scale, center + translate)
        case (r, phi):
            return (r * scale, phi + rotate)
        case float(r):
            return r * scale

    raise TypeError(f"Unsupported type: {type(obj)}")


def draw_shape(surface: pygame.Surface, color: pygame.Color, shape: Shape) -> None:
    """Draw a shape."""
    width, height = screen_size = surface.get_size()
    match shape:
        case Circle(radius, center):
            center = round(center.elementwise() * screen_size)
            pygame.draw.circle(surface, color, center, radius * width, width=1)
        case Polygon(points, center):
            points = [
                (center + pygame.Vector2(0, r).rotate(a)).elementwise() * screen_size
                for r, a in points
            ]
            pygame.draw.polygon(surface, color, points, width=1)
        case ComplexShape(parts, center):
            for part in parts:
                draw_shape(surface, color, part)
