import math
import random
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from functools import cache
from functools import partial
from inspect import getmembers
from typing import Any
from typing import cast

import pygame
import pygame.locals as pg
from pygame import Surface
from pygame.event import Event

from pygskin import imgui
from pygskin.animation import animate
from pygskin.assets import Assets
from pygskin.clock import Clock
from pygskin.clock import Timer
from pygskin.ecs import get_ecs_update_fn
from pygskin.game import run_game
from pygskin.imgui import label
from pygskin.screen import ScreenFn
from pygskin.screen import ScreenManager
from pygskin.screen import screen_manager
from pygskin.stylesheet import get_styles
from pygskin.utils import angle_between

from shapes import Circle
from shapes import ComplexShape
from shapes import Polygon
from shapes import Shape
from shapes import draw_shape
from shapes import transform


assets = Assets()
stylesheet = partial(
    get_styles,
    {
        "*": {
            "color": "white",
            "font": assets.Hyperspace.size(40),
            "padding": [10],
        },
    },
)
gui = imgui.IMGUI()


def asteroids():
    return screen_manager(
        {
            main_menu: [start_game],
            play_game(): [return_to_main_menu],
        }
    )


def main_menu(surface: Surface, events: list[Event], manager: ScreenManager) -> None:
    """The main menu screen."""
    rect = surface.get_rect()
    surface.blit(assets.main_menu, (0, 0))

    with imgui.render(gui, surface, stylesheet) as render:
        render(
            label("Press any key to start"),
            centerx=rect.centerx,
            y=rect.bottom - 100,
        )

    if any(event.type == pg.KEYDOWN for event in events):
        manager.send("start_game")


def start_game(input) -> ScreenFn | None:
    return play_game() if input == "start_game" else None


@cache
def play_game() -> ScreenFn:
    state: dict[str, Any] = {}
    entities: list[Any] = []
    ship = Ship()
    wave = Wave(0)

    ecs_update = get_ecs_update_fn(
        [
            update_timers,
            control_ship,
            steer_drone,
            steer_saucer,
            shoot_saucer,
            start_next_wave,
            apply_physics,
            collide_bullet_asteroid,
            collide_bullet_saucer,
            collide_asteroid_ship,
            collide_drone_ship,
            collide_bullet_ship,
            collide_saucer_ship,
            award_extra_lives,
            spawn_saucer,
            timeout_bullets,
            timeout_explosions,
            respawn_ship,
            play_heartbeat,
        ]
    )

    def reset_game():
        state["game_over"] = False
        state["paused"] = False
        state["ship"] = ship
        state["wave"] = wave
        entities.clear()
        entities.append(ship)
        entities.append(wave)
        ship.score = 0
        ship.lives = 3
        respawn_ship(ship, [])
        wave.number = 0
        wave.__post_init__()

    reset_game()

    def _play_game(surface: Surface, events: list[Event], manager: ScreenManager) -> None:
        """The main game screen."""
        surface.fill((0, 0, 0))
        surface.blit(assets.background, (0, 0))
        rect = surface.get_rect()
        layer = Surface(rect.size, pygame.SRCALPHA)

        if any(event.type == pg.KEYDOWN and event.key == pg.K_p for event in events):
            state["paused"] = not state["paused"]

        if not state["paused"]:
            ecs_update(entities, events=events, state=state)

        state["game_over"] = not ship.alive and ship.respawn_timer.finished
        if state["game_over"] and any(event.type == pg.KEYDOWN for event in events):
            entities.clear()
            reset_game()
            manager.send("game_over")

        for entity in entities:
            if isinstance(entity, Mob) and entity.alive:
                draw_mob(surface, entity)

        draw_lives(layer, ship.lives)
        with imgui.render(gui, layer, stylesheet) as render:
            render(label(f"{ship.score:7d}"), topleft=(0.025 * rect.width, 0))
            if state["game_over"]:
                render(label("GAME OVER"), center=rect.center)
            elif state["paused"]:
                render(label("PAUSED"), center=rect.center)
            elif not wave.get_ready_cooldown.finished:
                render(label("GET READY"), center=rect.center)

        surface.blit(layer, (0, 0))

    return _play_game


def return_to_main_menu(input) -> ScreenFn | None:
    return main_menu if input == "game_over" else None


def random_vector(magnitude: float = 1.0) -> pygame.Vector2:
    """Return a random vector with a magnitude of 1."""
    return pygame.Vector2(0, 1).rotate(random.random() * 360) * magnitude


# @dataclass
# class Shield(Timer):
#     """A shield that can be activated and deactivated."""

#     active: bool = False

#     FULL = pygame.Color(255, 255, 255, 255)
#     LOW = pygame.Color(255, 255, 255, 64)

#     @property
#     def color(self) -> pygame.Color:
#         """Return the color of the shield based on its remaining duration."""
#         return Shield.FULL.lerp(Shield.LOW, self.quotient)


@dataclass
class Mob:
    """A mobile object with position, velocity, and acceleration."""

    pos: pygame.Vector2
    radius: float = 0.1
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    acceleration: pygame.Vector2 = field(default_factory=pygame.Vector2)
    angle: float = 0.0
    spin: float = 0.0
    alive: bool = True
    color = pygame.Color("white")
    shape: Shape = Circle()


def collide(a: Mob, b: Mob) -> bool:
    """Check if two mobs are colliding."""
    if not (a.alive and b.alive):
        return False
    return a.pos.distance_to(b.pos) < (a.radius + b.radius)


@dataclass
class Ship(Mob):
    """A player-controlled ship with additional properties."""

    pos: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0.5))
    radius: float = 0.02
    angle: float = 180.0
    # shield: Shield = field(default_factory=lambda: Shield(10000))
    thruster: bool = False
    heartbeat: Timer = field(default_factory=lambda: Timer(1000))
    invulnerability: Timer = field(default_factory=lambda: Timer(2000))
    respawn_timer: Timer = field(default_factory=lambda: Timer(3000, paused=True))
    lives: int = 3
    score: int = 0
    extra_life_trigger: int = 1

    THRUST_VECTOR = pygame.Vector2(0, 0.005)
    FULL_SHIELD_COLOR = pygame.Color(255, 255, 255, 255)
    LOW_SHIELD_COLOR = pygame.Color(255, 255, 255, 64)

    shape: Shape = ComplexShape(
        [
            Polygon([(1, 0), (1, -150), (1, 150)]),
            Polygon([(0, 0), (1.2, -120), (1, -150), (1, 150), (1.2, 120)]),
        ],
    )
    exhaust = Polygon([(1, -150), (1.2, -180), (1, 150)])


class Size(IntEnum):
    """An enumeration of mob sizes."""

    big = 2
    medium = 1
    small = 0

    def smaller(self) -> "Size":
        """Return the next smaller size."""
        return Size(self.value - 1)


@dataclass
class Saucer(Mob):
    """An enemy saucer with additional properties."""

    pos: pygame.Vector2 = field(default_factory=pygame.Vector2)
    size: Size = Size.big
    firing_cooldown: Timer = field(default_factory=lambda: Timer(3000))
    color = pygame.Color("green")

    shape: Shape = ComplexShape(
        [
            Polygon([(1, 90), (0.5, 45), (0.5, -45), (1, -90)]),
            Polygon([(1, 90), (0.5, 135), (0.5, -135), (1, -90)]),
            Polygon([(0.5, 135), (0.75, 160), (0.75, -160), (0.5, -135)]),
        ],
    )

    def __post_init__(self) -> None:
        self.radius, speed, self.score = {
            Size.big: (0.04, 0.015, 200),
            Size.small: (0.025, 0.01, 1000),
        }[self.size]
        self.pos = pygame.Vector2(random.choice((0, 1)), random.uniform(0.1, 0.9))
        self.velocity = pygame.Vector2(math.copysign(speed, self.pos.x - 0.5), 0)


@dataclass
class Bullet(Mob):
    """A projectile fired by a ship."""

    source: Mob | None = None
    radius: float = 0.005
    ttl: Timer = field(default_factory=lambda: Timer(1000))
    color = pygame.Color("orange")

    SPEED = 0.15

    def __post_init__(self) -> None:
        self.ttl.elapsed = 0
        if isinstance(self.source, Saucer):
            self.color = pygame.Color("greenyellow")


class Explosion(Mob):
    """An explosion effect with a growing radius."""

    def __init__(self, pos: pygame.Vector2, size: Size = Size.big) -> None:
        super().__init__(pos)
        self.size = size
        self.timer = Timer(200)
        self._radius = animate({0.0: 0.0, 1.0: 0.375}, self.timer.quotient)
        self.color = pygame.Color("orange")

        assets[f"bang_{self.size.name}"].play()

    @property
    def radius(self) -> float:
        return next(self._radius)

    @radius.setter
    def radius(self, value: float) -> None:
        pass


@dataclass
class Asteroid(Mob):
    """An asteroid with additional properties."""

    size: Size = Size.big
    spin: float = field(default_factory=lambda: random.uniform(-10, 10))

    def __post_init__(self) -> None:
        self.radius, speed, self.score, self.num_fragments = {
            Size.big: (0.08, 0.01, 20, 2),
            Size.medium: (0.04, 0.02, 50, 2),
            Size.small: (0.02, 0.03, 100, 0),
        }[self.size]
        self.velocity = pygame.Vector2(
            random.uniform(-speed, speed),
            random.uniform(-speed, speed),
        )
        self.shape = Polygon([(random.uniform(0.8, 1), i * 18) for i in range(20)])


@dataclass
class Drone(Mob):
    """An enemy drone with additional properties."""

    size: Size = Size.big
    color = pygame.Color("lightblue")
    angle: float = random.random() * 360
    velocity: pygame.Vector2 = field(default_factory=lambda: random_vector(0.01))
    shape: Shape = Polygon([(1, 0), (1, -120), (0.1, 180), (1, 120)])
    THRUST_VECTOR = pygame.Vector2(0, 0.0075)

    def __post_init__(self) -> None:
        self.radius, self.speed, self.score, self.num_fragments = {
            Size.big: (0.05, 0.01, 0, 3),
            Size.medium: (0.04, 0.0125, 0, 2),
            Size.small: (0.025, 0.0175, 200, 0),
        }[self.size]


@dataclass
class Wave:
    """A wave of asteroids to spawn."""

    number: int = 0
    started: bool = False
    asteroids: list = field(default_factory=list)
    saucer: Saucer | None = None
    saucer_timer: Timer = field(default_factory=lambda: Timer(60000))
    get_ready_cooldown: Timer = field(default_factory=lambda: Timer(3000))

    def __post_init__(self) -> None:
        self.get_ready_cooldown.elapsed = 0
        self.saucer_timer.elapsed = 0

        self.asteroids[:] = list(
            Asteroid(random_vector(random.uniform(0.7, 0.9)))
            for _ in range(min(self.number + 3, 6))
        )

        if self.number > 3:
            self.asteroids.append(Drone(random_vector(random.uniform(0.7, 0.9))))

    @property
    def completed(self) -> bool:
        """Check if the wave is completed."""
        return self.started and not self.asteroids and not self.saucer


def draw_lives(surface: Surface, lives: int) -> None:
    """Draw the remaining lives as ships."""
    for i in range(lives):
        ship = transform(
            cast(ComplexShape, Ship.shape),
            pygame.Vector2(0.06 + (i * 0.05), 0.1),
            scale=Ship.radius,
            rotate=180,
        )
        draw_shape(surface, "white", ship)


def update_timers(entity: object, **_) -> None:
    """Update all timers in an entity."""
    delta_time = Clock.get_time()
    for name, timer in getmembers(entity, lambda attr: isinstance(attr, Timer)):
        timer.tick(delta_time)


def apply_physics(mob: Mob, **_) -> None:
    """Apply physics to a mob."""
    if not mob.alive:
        return
    delta_time = Clock.get_time() / 100
    mob.velocity = mob.velocity + mob.acceleration * delta_time
    mob.pos += mob.velocity * delta_time
    # wrap around screen
    mob.pos = mob.pos.elementwise() % 1.0
    mob.angle = (mob.angle + mob.spin * delta_time) % 360


def collide_bullet_asteroid(bullet: Bullet, entities: list, state: dict, **_) -> None:
    """Check for collisions between bullets and asteroids or drones."""
    if not bullet.alive:
        return
    wave = state["wave"]
    for asteroid in list(wave.asteroids):
        if collide(bullet, asteroid):
            if isinstance(bullet.source, Ship):
                bullet.source.score += asteroid.score
            entities.append(Explosion(asteroid.pos.copy(), size=asteroid.size))
            fragments = get_fragments(asteroid)
            wave.asteroids.extend(fragments)
            entities.extend(fragments)
            asteroid.alive = False
            wave.asteroids.remove(asteroid)
            bullet.alive = False
            entities.remove(bullet)


def get_fragments(mob: Asteroid | Drone) -> list[Asteroid | Drone]:
    """Return a list of fragments from a mob."""
    return [
        mob.__class__(mob.pos.copy(), size=mob.size.smaller())
        for _ in range(mob.num_fragments)
    ]


def collide_bullet_saucer(bullet: Bullet, entities: list, state: dict, **_) -> None:
    """Check for collisions between bullets and saucers."""
    wave = state["wave"]
    if (
        (saucer := wave.saucer)
        and collide(bullet, saucer)
        and isinstance(bullet.source, Ship)
    ):
        bullet.source.score += saucer.score
        entities.append(Explosion(saucer.pos.copy(), size=saucer.size))
        assets[f"saucer_{saucer.size.name}"].fadeout(200)
        saucer.alive = False
        entities.remove(saucer)
        wave.saucer = None
        wave.saucer_timer.elapsed = 0
        bullet.alive = False
        entities.remove(bullet)


def collide_asteroid_ship(asteroid: Asteroid, entities: list, state: dict, **_) -> None:
    """Check for collisions between asteroids and the ship."""
    ship = state["ship"]
    if ship.alive and collide(ship, asteroid):
        # if ship.shield.active:
        #     asteroid.velocity = asteroid.velocity.reflect(asteroid.pos - ship.pos)
        #     asteroid.pos += asteroid.velocity
        if ship.invulnerability.finished:
            entities.append(Explosion(ship.pos.copy()))
            ship.alive = False


def collide_drone_ship(drone: Drone, entities: list, state: dict, **_) -> None:
    """Check for collisions between drones and the ship."""
    ship = state["ship"]
    if ship.alive and collide(ship, drone):
        # if ship.shield.active:
        #     entities.append(Explosion(drone.pos.copy(), size=drone.size))
        #     drone.alive = False
        # world.remove(drone)
        if ship.invulnerability.finished:
            entities.append(Explosion(ship.pos.copy()))
            ship.alive = False


def collide_bullet_ship(bullet: Bullet, entities: list, state: dict, **_) -> None:
    """Check for collisions between bullets and the ship."""
    ship = state["ship"]
    if (
        ship.alive
        and ship.invulnerability.finished
        and bullet.source is not ship
        and collide(bullet, ship)
    ):
        bullet.alive = False
        entities.remove(bullet)
        if ship.invulnerability.finished:  # not ship.shield.active:
            entities.append(Explosion(ship.pos.copy()))
            ship.alive = False


def collide_saucer_ship(saucer: Saucer, entities: list, state: dict, **_) -> None:
    """Check for collisions between saucers and the ship."""
    ship = state["ship"]
    if collide(ship, saucer):
        # if ship.shield.active:
        #     world.add(Explosion(saucer.pos.copy(), size=saucer.size))
        #     world.remove(saucer)
        # else:
        if ship.invulnerability.finished:
            entities.append(Explosion(ship.pos.copy()))
            ship.alive = False


def spawn_saucer(wave: Wave, entities: list, **_) -> None:
    """Spawn a saucer if the timer has elapsed."""
    if not wave.saucer and wave.saucer_timer.finished:
        wave.saucer = Saucer(size=random.choice((Size.big, Size.small)))
        entities.append(wave.saucer)
        assets[f"saucer_{wave.saucer.size.name}"].play(loops=-1, fade_ms=100)


def control_ship(ship: Ship, events: list[Event], entities: list, **_) -> None:
    """Control the ship with keyboard input."""
    if not ship.alive:
        return

    for event in events:
        if event.type == pg.KEYDOWN:
            key = event.key
            if key == pg.K_UP:
                ship.thruster = True
                assets.thrust.play(loops=-1, fade_ms=100)
            if key == pg.K_LEFT:
                ship.spin = -20
            if key == pg.K_RIGHT:
                ship.spin = 20
            if key == pg.K_SPACE:
                assets.fire.play()
                entities.append(
                    Bullet(
                        ship.pos,
                        source=ship,
                        velocity=ship.velocity
                        + pygame.Vector2(0, Bullet.SPEED).rotate(ship.angle),
                    )
                )
            # if key == pg.K_s:
            #     ship.shield.active = True

        if event.type == pg.KEYUP:
            key = event.key
            if key == pg.K_UP:
                ship.thruster = False
                ship.acceleration = pygame.Vector2(0)
                assets.thrust.fadeout(200)
            if key in (pg.K_LEFT, pg.K_RIGHT):
                ship.spin = 0
            # if key == pg.K_s:
            #     ship.shield.active = False

    if ship.thruster:
        ship.acceleration = Ship.THRUST_VECTOR.rotate(ship.angle)


def steer_drone(drone: Drone, state: dict, **_) -> None:
    """Steer the drone towards the ship."""
    if drone.size != Size.big:
        angle_to_ship = 180 - angle_between(drone.pos, state["ship"].pos)
        drone.spin = -10 if (angle_to_ship - drone.angle) % 360 < 180 else 10
        drone.acceleration = Drone.THRUST_VECTOR.rotate(drone.angle)
        drone.velocity.clamp_magnitude_ip(drone.speed)


def steer_saucer(saucer: Saucer, **_) -> None:
    """Steer the saucer in a sine wave pattern."""
    saucer.velocity.y = math.sin(math.radians(saucer.pos.x * 360 * 3)) * 0.015


def shoot_saucer(saucer: Saucer, entities: list, state: dict, **_) -> None:
    """Shoot a bullet from the saucer."""
    if saucer.firing_cooldown.finished:
        if saucer.size == Size.big:
            velocity = saucer.velocity + random_vector(Bullet.SPEED)
        else:
            # aim at ship with slight spread
            velocity = ((state["ship"].pos - saucer.pos).normalize() * 0.15).rotate(
                random.uniform(-10, 10)
            )
        entities.append(Bullet(saucer.pos, source=saucer, velocity=velocity))
        assets.saucer_fire.play()
        saucer.firing_cooldown.elapsed = 0


def start_next_wave(wave: Wave, entities: list, state: dict, **_) -> None:
    """Start the next wave when all asteroids are destroyed."""
    if wave.completed:
        wave.number += 1
        wave.started = False
        wave.__post_init__()

    if not wave.started and wave.get_ready_cooldown.finished:
        wave.started = True
        entities.extend(wave.asteroids)
        # avoid spawning on top of ship
        for asteroid in wave.asteroids:
            if collide(asteroid, state["ship"]):
                asteroid.pos += state["ship"].pos


def timeout_bullets(bullet: Bullet, entities: list, **_) -> None:
    """Remove bullets that have exceeded their time-to-live."""
    if bullet.alive and bullet.ttl.finished:
        bullet.alive = False
        entities.remove(bullet)


def timeout_explosions(explosion: Explosion, entities: list, **_) -> None:
    """Remove explosions that have completed their animation."""
    if explosion.alive and explosion.timer.finished:
        explosion.alive = False
        entities.remove(explosion)


def respawn_ship(ship: Ship, entities: list, **_) -> None:
    """Remove the ship if destroyed."""
    if ship.alive:
        return

    if ship.respawn_timer.paused:
        assets.thrust.fadeout(200)
        ship.respawn_timer.elapsed = 0
        ship.respawn_timer.paused = False
        ship.lives -= 1
        return

    if ship.respawn_timer.finished:
        if ship.lives == 0:
            entities.remove(ship)
            return

        ship.pos.update(0.5, 0.5)
        ship.velocity.update(0, 0)
        ship.acceleration.update(0, 0)
        ship.spin = 0.0
        ship.thruster = False
        # ship.shield.active = False
        ship.alive = True
        ship.respawn_timer.paused = True
        return


def award_extra_lives(ship: Ship, state: dict, **_) -> None:
    """Award an extra life every 10,000 points."""
    if ship.score > ship.extra_life_trigger * 10000:
        if ship.lives < 10:
            ship.lives += 1
            assets.life.play()
        ship.extra_life_trigger += 1


def play_heartbeat(ship: Ship, **_) -> None:
    """Play the heartbeat sound effect at regular intervals."""
    if ship.heartbeat.finished:
        assets.beat1.play()
        ship.heartbeat.duration = 1000 + int(min(1.0, ship.score / 1000000) * 750)
        ship.heartbeat.elapsed = 0


def draw_mob(surface: Surface, mob: Mob) -> None:
    """Draw a mob."""
    to_world = partial(transform, translate=mob.pos, scale=mob.radius, rotate=mob.angle)

    match mob:
        case Drone(size=Size.medium) as drone:
            offset = pygame.Vector2(0, 0.013).rotate(drone.angle)
            front = transform(drone.shape, translate=offset, scale=0.625)
            back = transform(drone.shape, translate=-offset, scale=0.625, rotate=180)
            draw_shape(surface, drone.color, to_world(front))
            draw_shape(surface, drone.color, to_world(back))

        case Drone(size=Size.big) as drone:
            for angle in range(0, 360, 60):
                segment = transform(
                    cast(Polygon, drone.shape),
                    translate=pygame.Vector2(0, 0.025).rotate(angle),
                    scale=drone.radius * 0.5,
                    rotate=angle - (60 if angle % 120 == 0 else -60),
                )
                draw_shape(surface, drone.color, transform(segment, mob.pos))

        case Explosion() as explosion:
            explode = partial(transform, scale=explosion.radius)
            for i in range(3):
                blast_wave = explode(Circle(random.uniform(1, 1.1875)))
                draw_shape(surface, explosion.color, to_world(blast_wave))

        case Ship() as ship:
            draw_shape(surface, ship.color, to_world(ship.shape))

            if ship.thruster:
                draw_shape(surface, ship.color, to_world(ship.exhaust))

            # if ship.shield.active:
            #     shield_shape = to_world(Circle(random.uniform(1.2, 1.3875)))
            #     draw_shape(surface, ship.shield.color, shield_shape)

        case _:
            draw_shape(surface, mob.color, to_world(mob.shape))


if __name__ == "__main__":
    run_game(
        pygame.Window("Asteroids", (800, 800)),
        asteroids(),
    )
