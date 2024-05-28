from __future__ import annotations

import math
import random
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from inspect import getmembers
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Iterable

import pygame
import pygame.locals as pg
from pygame.event import Event

from pygskin import ecs
from pygskin.animation import Player, KeyframeAnimation
from pygskin.assets import Assets
from pygskin.clock import Clock
from pygskin.events import EventDispatch
from pygskin.game import Game
from pygskin.screen import Screen
from pygskin.text import Text
from pygskin.utils import angle_between
from pygskin.window import Window

WIDTH, HEIGHT = 800, 800

assets = Assets()

Points = list[tuple[float, float]]


class Settings:
    """Game settings."""

    mute: bool = False


def translate(v: pygame.Vector2) -> pygame.Vector2:
    """Translate a vector from 0-1 space to screen space."""
    return round(v.elementwise() * (WIDTH, HEIGHT))


def radial_point(origin: pygame.Vector2, radius: float, angle: float) -> pygame.Vector2:
    """Return a point on a circle given an origin, radius, and angle."""
    return origin + pygame.Vector2(0, radius).rotate(angle)


def random_vector(magnitude: float = 1.0) -> pygame.Vector2:
    """Return a random vector with a magnitude of 1."""
    return pygame.Vector2(0, 1).rotate(random.random() * 360) * magnitude


def draw_radial_points(
    surface: pygame.Surface,
    color: pygame.Color,
    points: Points,
    pos: pygame.Vector2 = pygame.Vector2(0),
    radius: float = 1.0,
    angle: float = 0.0,
) -> None:
    """Draw a polygon of points around a center point."""
    pygame.draw.polygon(
        surface,
        color,
        [translate(radial_point(pos, radius * r, angle + phi)) for r, phi in points],
        width=1,
    )


def collide(a: Mob, b: Mob) -> bool:
    """Check if two mobs are colliding."""
    if not (a.alive and b.alive):
        return False
    return a.pos.distance_to(b.pos) < (a.radius + b.radius)


@dataclass
class Cooldown:
    """A cooldown timer that can be started and checked for completion."""

    duration: int
    remaining: int = 0
    active: bool = True

    def start(self) -> None:
        self.remaining = self.duration


@dataclass
class Shield(Cooldown):
    """A shield that can be activated and deactivated."""

    active: bool = False

    FULL = pygame.Color(255, 255, 255, 255)
    LOW = pygame.Color(255, 255, 255, 64)

    @property
    def color(self) -> pygame.Color:
        """Return the color of the shield based on its remaining duration."""
        return Shield.LOW.lerp(Shield.FULL, self.remaining / self.duration)


@dataclass
class Mob:
    """A mobile object with position, velocity, and acceleration."""

    pos: pygame.Vector2
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    acceleration: pygame.Vector2 = field(default_factory=pygame.Vector2)
    angle: float = 0.0
    spin: float = 0.0
    radius: float = 0.1
    alive: bool = True
    color = pygame.Color("white")


@dataclass
class Ship(Mob):
    """A player-controlled ship with additional properties."""

    pos: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0.5))
    radius: float = 0.02
    angle: float = 180.0
    shield: Shield = field(default_factory=lambda: Shield(10000))
    thruster: bool = False
    heartbeat: Cooldown = field(default_factory=lambda: Cooldown(1000))
    invulnerability: Cooldown = field(default_factory=lambda: Cooldown(2000))
    respawn_timer: Cooldown = field(default_factory=lambda: Cooldown(3000))
    lives: int = 3
    extra_life_trigger: int = 1

    THRUST_VECTOR = pygame.Vector2(0, 0.005)
    FULL_SHIELD_COLOR = pygame.Color(255, 255, 255, 255)
    LOW_SHIELD_COLOR = pygame.Color(255, 255, 255, 64)

    FUSELAGE: ClassVar[Points] = [(1, 0), (1, -150), (1, 150)]
    WINGS: ClassVar[Points] = [(0, 0), (1.2, -120), (1, -150), (1, 150), (1.2, 120)]
    EXHAUST: ClassVar[Points] = [(1, -150), (1.2, -180), (1, 150)]

    def respawn(self) -> None:
        """Reset the ship's position and state."""
        self.pos.update(0.5, 0.5)
        self.velocity.update(0, 0)
        self.acceleration.update(0, 0)
        self.spin = 0.0
        self.thruster = False
        self.shield.active = False
        self.alive = True
        self.shield.remaining = self.shield.duration
        self.invulnerability.start()


class Size(IntEnum):
    """An enumeration of mob sizes."""

    big = 0
    medium = 1
    small = 2

    def smaller(self) -> Size:
        """Return the next smaller size."""
        return Size(self.value + 1)


@dataclass
class Saucer(Mob):
    """An enemy saucer with additional properties."""

    pos: pygame.Vector2 = field(default_factory=pygame.Vector2)
    size: ClassVar[Size] = Size.big
    radius: float = 0.04
    speed: float = 0.015
    score: int = 200
    firing_cooldown: Cooldown = field(default_factory=lambda: Cooldown(3000))

    SECTIONS: ClassVar[list[Points]] = [
        [(1, 90), (0.5, 45), (0.5, -45), (1, -90)],
        [(1, 90), (0.5, 135), (0.5, -135), (1, -90)],
        [(0.5, 135), (0.75, 160), (0.75, -160), (0.5, -135)],
    ]

    def __post_init__(self) -> None:
        self.pos = pygame.Vector2(random.choice((0, 1)), random.uniform(0.1, 0.9))
        self.velocity = pygame.Vector2(math.copysign(self.speed, self.pos.x - 0.5), 0)


@dataclass
class SmallSaucer(Saucer):
    """A smaller variant of the saucer."""

    size: ClassVar[Size] = Size.small
    radius: float = 0.025
    speed: float = 0.01
    score: int = 1000


@dataclass
class Bullet(Mob):
    """A projectile fired by a ship."""

    source: ClassVar[type[Mob]] = Ship
    radius: float = 0.005
    ttl: Cooldown = field(default_factory=lambda: Cooldown(1000))
    color = pygame.Color("orange")

    SPEED = 0.15

    def __post_init__(self) -> None:
        self.ttl.start()


@dataclass
class SaucerBullet(Bullet):
    """A projectile fired by a saucer."""

    source: ClassVar[type[Mob]] = Saucer
    color = pygame.Color("greenyellow")


@dataclass
class Explosion(Mob):
    """An explosion effect with a growing radius."""

    size: Size = Size.big
    color = pygame.Color("orange")
    anim: Player = field(
        default_factory=lambda: Player(KeyframeAnimation({0: 0.0, 200: 0.375}))
    )

    def __post_init__(self) -> None:
        self.anim.send("start")
        if not Settings.mute:
            assets[f"bang_{self.size.name}"].play()


@dataclass
class Asteroid(Mob):
    """An asteroid with additional properties."""

    size: Size = Size.big
    spin: float = field(default_factory=lambda: random.uniform(-10, 10))
    points: Points = field(
        default_factory=lambda: [
            pygame.Vector2(random.uniform(0.8, 1), i * 18) for i in range(20)
        ]
    )

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


@dataclass
class Drone(Mob):
    """An enemy drone with additional properties."""

    size: Size = Size.big
    color = pygame.Color("lightblue")

    SEGMENT: ClassVar[Points] = [(1, 0), (1, -120), (0.1, 180), (1, 120)]
    THRUST_VECTOR = pygame.Vector2(0, 0.0075)

    def __post_init__(self) -> None:
        self.radius, self.speed, self.score, self.num_fragments = {
            Size.big: (0.05, 0.01, 0, 3),
            Size.medium: (0.04, 0.0125, 0, 2),
            Size.small: (0.025, 0.0175, 200, 0),
        }[self.size]
        self.velocity = pygame.Vector2(0, self.speed).rotate(random.random() * 360)
        if self.size != Size.big:
            self.angle = random.random() * 360


@dataclass
class Wave:
    """A wave of asteroids to spawn."""

    number: int = 0
    started: bool = False
    asteroids: list[Asteroid | Drone] = field(default_factory=list)
    saucer: Saucer | None = None
    saucer_timer: Cooldown = field(default_factory=lambda: Cooldown(60000))
    get_ready_cooldown: Cooldown = field(default_factory=lambda: Cooldown(3000))

    def __post_init__(self) -> None:
        self.get_ready_cooldown.start()
        self.saucer_timer.start()

        self.asteroids.extend(
            Asteroid(random_vector(random.uniform(0.7, 0.9)))
            for _ in range(min(self.number + 3, 6))
        )

        if self.number > 3:
            self.asteroids.append(Drone(random_vector(random.uniform(0.7, 0.9))))

    def remove(self, mob: Asteroid | Drone) -> None:
        """Remove an asteroid or drone from the wave."""
        self.asteroids.remove(mob)

    @property
    def completed(self) -> bool:
        """Check if the wave is completed."""
        return self.started and not self.asteroids and not self.saucer


class DynamicLabel:
    """A label that can be updated with a new value."""

    def __init__(self, value_fn: Callable[[], Any], **kwargs) -> None:
        self.value_fn = value_fn
        self.value = value_fn()
        self.label = Text(str(self.value), **kwargs)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the label."""
        if (value := self.value_fn()) != self.value:
            self.value = value
            self.label.text = str(value)
            with suppress(AttributeError):
                del self.label.image
                del self.label.rect
        surface.blit(self.label.image, self.label.rect)


class Lives:
    """Display of remaining lives."""

    def __init__(self, value_fn: Callable[[], int]) -> None:
        self.value_fn = value_fn

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the remaining lives as ships."""
        for i in range(self.value_fn()):
            self.draw_ship(surface, pygame.Vector2(0.06 + (i * 0.05), 0.1))

    def draw_ship(self, surface: pygame.Surface, pos: pygame.Vector2) -> None:
        """Draw a ship icon at a given position."""
        for part in (Ship.FUSELAGE, Ship.WINGS):
            draw_radial_points(surface, "white", part, pos=pos, radius=0.015, angle=180)


class World(ecs.Container):
    """The game world containing all entities and systems."""

    def __init__(self) -> None:
        super().__init__()

        self.systems += [
            update_cooldowns,
            EventDispatch(),
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
            cull_bullets,
            cull_explosions,
            respawn_ship,
            play_heartbeat,
            draw_asteroid,
            draw_bullet,
            draw_drone,
            draw_explosion,
            draw_saucer,
            draw_ship,
        ]

        self.ship = Ship()
        self.add(self.ship)

        self.paused = False
        self.score = 0

    @property
    def asteroids(self) -> Iterable[Asteroid | Drone]:
        """Iterate over all asteroids and drones."""
        return (mob for mob in self.entities if isinstance(mob, (Asteroid, Drone)))

    def update(self, events: list[Event], surface: pygame.Surface) -> None:
        """Update the world state."""
        if any(event.type == pg.KEYDOWN and event.key == pg.K_p for event in events):
            self.paused = not self.paused
        if not self.paused:
            super().update(events=events, surface=surface, world=self)

    def add(self, mobs: Mob | Iterable[Mob]) -> None:
        """Add multiple mobs to the world."""
        if isinstance(mobs, Mob):
            mobs = [mobs]
        self.entities.extend(mobs)

    def remove(self, entity: Any) -> None:
        """Remove an entity from the world."""
        entity.alive = False
        match entity:
            case (Asteroid() | Drone()) as mob:
                self.entities.remove(mob)
                self.wave.remove(mob)

            case Saucer() as saucer:
                if not Settings.mute:
                    assets[f"saucer_{saucer.size.name}"].fadeout(200)
                self.entities.remove(saucer)
                self.wave.saucer = None
                self.wave.saucer_timer.start()

            case Ship() as ship:
                if not Settings.mute:
                    assets.thrust.fadeout(200)
                ship.lives -= 1
                ship.respawn_timer.start()

            case _ as entity:
                self.entities.remove(entity)

    def remove_all(self) -> None:
        """Remove all entities from the world."""
        for entity in self.entities:
            self.remove(entity)

    @property
    def wave(self) -> Wave:
        """The current wave of asteroids."""
        if not hasattr(self, "_wave"):
            self._wave = Wave(0)
            self.entities.append(self._wave)
        return self._wave

    @wave.setter
    def wave(self, value: Wave) -> None:
        """Set the current wave of asteroids."""
        self.entities.remove(self._wave)
        self._wave = value
        self.entities.append(value)


def any_key_pressed(events: list[Event]) -> bool:
    """Check if any key was pressed in a list of events."""
    return any(event.type == pg.KEYDOWN for event in events)


class MainMenu(Screen):
    """The main menu screen."""

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the main menu screen."""
        surface.blit(assets.main_menu, (0, 0))
        prompt = Text(
            "Press any key to start",
            font=assets.Hyperspace.size(30),
            padding=[20],
            midbottom=surface.get_rect().midbottom,
        )
        surface.blit(prompt.image, prompt.rect)

    def update(self, events: list[Event]) -> None:
        """Update the main menu screen."""
        if any_key_pressed(events):
            self.exit()

    def start_game(self) -> type[Play]:
        """Transition to the play screen."""
        return Play


class Play(Screen):
    """The main game screen."""

    def setup(self) -> None:
        """Initialize the game world and labels."""
        self.world = World()
        self.surface = pygame.Surface(Window.size).convert_alpha()

        text_params = {
            "color": "white",
            "padding": [10],
            "font": assets.Hyperspace.size(40),
        }
        self.pause_label = Text(
            "PAUSED",
            center=translate(pygame.Vector2(0.5, 0.6)),
            **text_params,
        )
        self.get_ready_label = Text(
            "GET READY",
            center=translate(pygame.Vector2(0.5, 0.6)),
            **text_params,
        )
        self.game_over_label = Text(
            "GAME OVER",
            center=translate(pygame.Vector2(0.5, 0.4)),
            **text_params,
        )
        self.score_label = DynamicLabel(
            lambda: self.world.score,
            topleft=translate(pygame.Vector2(0.025, 0)),
            **text_params,
        )
        self.lives_meter = Lives(lambda: self.world.ship.lives)

        self.game_over = False

    def update(self, events: list[Event]) -> None:
        """Update the game state."""
        self.surface.blit(assets.background, (0, 0))
        self.world.update(events, surface=self.surface)
        self.game_over = (
            not self.world.ship.alive and not self.world.ship.respawn_timer.remaining
        )
        if self.game_over and any_key_pressed(events):
            self.world.remove_all()
            self.exit()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the game screen."""
        self.score_label.draw(self.surface)
        self.lives_meter.draw(self.surface)
        if self.game_over:
            self.surface.blit(self.game_over_label.image, self.game_over_label.rect)
        elif self.world.paused:
            self.surface.blit(self.pause_label.image, self.pause_label.rect)
        elif self.world.wave.get_ready_cooldown.remaining > 0:
            self.surface.blit(self.get_ready_label.image, self.get_ready_label.rect)
        surface.blit(self.surface, (0, 0))

    def back_to_main_menu(self) -> type[MainMenu]:
        """Transition back to the main menu screen."""
        return MainMenu


@ecs.System
def update_cooldowns(entity: object, **_) -> None:
    """Update cooldown timers."""
    for name, attr in getmembers(entity, lambda attr: isinstance(attr, Cooldown)):
        if attr.active:
            attr.remaining = max(0, attr.remaining - Clock.delta_time)


@ecs.System
def apply_physics(mob: Mob, **_) -> None:
    """Apply physics to a mob."""
    if not mob.alive:
        return
    delta_time = Clock.delta_time / 100
    mob.velocity = mob.velocity + mob.acceleration * delta_time
    mob.pos += mob.velocity * delta_time
    # wrap around screen
    mob.pos = mob.pos.elementwise() % 1.0
    mob.angle = (mob.angle + mob.spin * delta_time) % 360


@ecs.System
def collide_bullet_asteroid(bullet: Bullet, world: World, **_) -> None:
    """Check for collisions between bullets and asteroids or drones."""
    for asteroid in world.asteroids:
        if collide(bullet, asteroid):
            if bullet.source is Ship:
                world.score += asteroid.score
            world.add(Explosion(asteroid.pos.copy(), size=asteroid.size))
            fragments = get_fragments(asteroid)
            world.wave.asteroids.extend(fragments)
            world.add(fragments)
            world.remove(asteroid)
            world.remove(bullet)


def get_fragments(mob: Asteroid | Drone) -> list[Asteroid | Drone]:
    """Return a list of fragments from a mob."""
    return [
        mob.__class__(mob.pos.copy(), size=mob.size.smaller())
        for _ in range(mob.num_fragments)
    ]


@ecs.System
def collide_bullet_saucer(bullet: Bullet, world: World, **_) -> None:
    """Check for collisions between bullets and saucers."""
    if (
        (saucer := world.wave.saucer)
        and collide(bullet, saucer)
        and bullet.source is Ship
    ):
        world.score += saucer.score
        world.add(Explosion(saucer.pos.copy(), size=saucer.size))
        world.remove(saucer)
        world.remove(bullet)


@ecs.System
def collide_asteroid_ship(asteroid: Asteroid, world: World, **_) -> None:
    """Check for collisions between asteroids and the ship."""
    ship = world.ship
    if ship.alive and collide(ship, asteroid):
        if ship.shield.active:
            asteroid.velocity = asteroid.velocity.reflect(asteroid.pos - ship.pos)
            asteroid.pos += asteroid.velocity
        elif not ship.invulnerability.remaining:
            world.add(Explosion(ship.pos.copy()))
            world.remove(ship)


@ecs.System
def collide_drone_ship(drone: Drone, world: World, **_) -> None:
    """Check for collisions between drones and the ship."""
    ship = world.ship
    if ship.alive and collide(ship, drone):
        if ship.shield.active:
            world.add(Explosion(drone.pos.copy(), size=drone.size))
            world.remove(drone)
        elif not ship.invulnerability.remaining:
            world.add(Explosion(ship.pos.copy()))
            world.remove(ship)


@ecs.System
def collide_bullet_ship(bullet: Bullet, world: World, **_) -> None:
    """Check for collisions between bullets and the ship."""
    ship = world.ship
    if (
        ship.alive
        and not ship.invulnerability.remaining
        and bullet.source is not Ship
        and collide(bullet, ship)
    ):
        world.remove(bullet)
        if not ship.shield.active:
            world.add(Explosion(ship.pos.copy()))
            world.remove(ship)


@ecs.System
def collide_saucer_ship(saucer: Saucer, world: World, **_) -> None:
    """Check for collisions between saucers and the ship."""
    ship = world.ship
    if collide(ship, saucer):
        if ship.shield.active:
            world.add(Explosion(saucer.pos.copy(), size=saucer.size))
            world.remove(saucer)
        else:
            world.add(Explosion(ship.pos.copy()))
            world.remove(ship)


@ecs.System
def respawn_ship(ship: Ship, world: World, **_) -> None:
    """Respawn the ship if it is dead and has lives remaining."""
    if not ship.alive and not ship.respawn_timer.remaining and ship.lives > 0:
        ship.respawn()


@ecs.System
def spawn_saucer(wave: Wave, world: World, **_) -> None:
    """Spawn a saucer if the timer has elapsed."""
    if not wave.saucer and wave.saucer_timer.remaining <= 0:
        wave.saucer = random.choice((Saucer, SmallSaucer))()
        world.add(wave.saucer)
        if not Settings.mute:
            assets[f"saucer_{wave.saucer.size.name}"].play(loops=-1, fade_ms=100)


@ecs.System
def control_ship(ship: Ship, events: list[Event], world: World, **_) -> None:
    """Control the ship with keyboard input."""
    if not ship.alive:
        return

    for event in events:
        if event.type == pg.KEYDOWN:
            key = event.key
            if key == pg.K_UP:
                ship.thruster = True
                if not Settings.mute:
                    assets.thrust.play(loops=-1, fade_ms=100)
            if key == pg.K_LEFT:
                ship.spin = -20
            if key == pg.K_RIGHT:
                ship.spin = 20
            if key == pg.K_SPACE:
                if not Settings.mute:
                    assets.fire.play()
                world.add(
                    Bullet(
                        ship.pos,
                        velocity=radial_point(ship.velocity, Bullet.SPEED, ship.angle),
                    )
                )
            if key == pg.K_s:
                ship.shield.active = True

        if event.type == pg.KEYUP:
            key = event.key
            if key == pg.K_UP:
                ship.thruster = False
                ship.acceleration = pygame.Vector2(0)
                if not Settings.mute:
                    assets.thrust.fadeout(200)
            if key in (pg.K_LEFT, pg.K_RIGHT):
                ship.spin = 0
            if key == pg.K_s:
                ship.shield.active = False

    if ship.thruster:
        ship.acceleration = Ship.THRUST_VECTOR.rotate(ship.angle)


@ecs.System
def steer_drone(drone: Drone, world: World, **_) -> None:
    """Steer the drone towards the ship."""
    if drone.size != Size.big:
        angle_to_ship = 180 - angle_between(drone.pos, world.ship.pos)
        drone.spin = -10 if (angle_to_ship - drone.angle) % 360 < 180 else 10
        drone.acceleration = Drone.THRUST_VECTOR.rotate(drone.angle)
        drone.velocity.clamp_magnitude_ip(drone.speed)


@ecs.System
def steer_saucer(saucer: Saucer, **_) -> None:
    """Steer the saucer in a sine wave pattern."""
    saucer.velocity.y = math.sin(math.radians(saucer.pos.x * 360 * 3)) * 0.015


@ecs.System
def shoot_saucer(saucer: Saucer, world: World, **_) -> None:
    """Shoot a bullet from the saucer."""
    if saucer.firing_cooldown.remaining <= 0:
        if saucer.size == Size.big:
            velocity = saucer.velocity + random_vector(Bullet.SPEED)
        else:
            # aim at ship with slight spread
            velocity = ((world.ship.pos - saucer.pos).normalize() * 0.15).rotate(
                random.uniform(-10, 10)
            )
        world.add(SaucerBullet(saucer.pos, velocity=velocity))
        if not Settings.mute:
            assets.saucer_fire.play()
        saucer.firing_cooldown.start()


@ecs.System
def start_next_wave(wave: Wave, world: World, **_) -> None:
    """Start the next wave when all asteroids are destroyed."""
    if wave.completed:
        world.wave = Wave(wave.number + 1)

    if not wave.started and not wave.get_ready_cooldown.remaining:
        wave.started = True
        world.add(wave.asteroids)
        # avoid spawning on top of ship
        for asteroid in wave.asteroids:
            asteroid.pos += world.ship.pos
        wave.saucer_timer.start()


@ecs.System
def cull_bullets(bullet: Bullet, world: World, **_) -> None:
    """Remove bullets that have exceeded their time-to-live."""
    if not bullet.ttl.remaining:
        world.remove(bullet)


@ecs.System
def cull_explosions(explosion: Explosion, world: World, **_) -> None:
    """Remove explosions that have completed their animation."""
    if explosion.anim.elapsed >= explosion.anim.duration:
        world.remove(explosion)


@ecs.System
def award_extra_lives(ship: Ship, world: World, **_) -> None:
    """Award an extra life every 10,000 points."""
    if world.score > ship.extra_life_trigger * 10000:
        if ship.lives < 10:
            ship.lives += 1
            if not Settings.mute:
                assets.life.play()
        ship.extra_life_trigger += 1


@ecs.System
def play_heartbeat(ship: Ship, world: World, **_) -> None:
    """Play the heartbeat sound effect at regular intervals."""
    if not ship.heartbeat.remaining:
        if not Settings.mute:
            assets.beat1.play()
        ship.heartbeat.duration = 1000 + int(min(1.0, world.score / 1000000) * 750)
        ship.heartbeat.start()


@ecs.System
def draw_asteroid(asteroid: Asteroid, surface: pygame.Surface, **_) -> None:
    """Draw an asteroid."""
    if asteroid.alive:
        draw_radial_points(
            surface,
            asteroid.color,
            asteroid.points,
            pos=asteroid.pos,
            radius=asteroid.radius,
            angle=asteroid.angle,
        )


@ecs.System
def draw_bullet(bullet: Bullet, surface: pygame.Surface, **_) -> None:
    """Draw a bullet."""
    if bullet.alive:
        pygame.draw.circle(
            surface,
            bullet.color,
            translate(bullet.pos),
            radius=bullet.radius * WIDTH,
            width=0,
        )


@ecs.System
def draw_drone(drone: Drone, surface: pygame.Surface, **_) -> None:
    """Draw a drone."""
    if drone.alive:
        if drone.size == Size.small:
            draw_drone_segment(surface, drone.pos, drone.radius, drone.angle)
        if drone.size == Size.medium:
            offset = math.tan(math.radians(50)) * 0.01
            for a_add in (0, 180):
                pos = radial_point(drone.pos, offset, drone.angle + a_add)
                draw_drone_segment(surface, pos, 0.025, drone.angle + a_add)
        if drone.size == Size.big:
            for angle in range(0, 360, 60):
                pos = radial_point(drone.pos, 0.025, angle)
                angle = angle - (60 if angle % 120 == 0 else -60)
                draw_drone_segment(surface, pos, 0.025, angle)


def draw_drone_segment(
    surface: pygame.Surface, pos: pygame.Vector2, radius: float, angle: float
) -> None:
    """Draw a drone segment."""
    draw_radial_points(
        surface,
        Drone.color,
        Drone.SEGMENT,
        pos=pos,
        radius=radius,
        angle=angle,
    )


@ecs.System
def draw_explosion(explosion: Explosion, surface: pygame.Surface, **_) -> None:
    """Draw an explosion."""
    if explosion.alive:
        radius = explosion.anim.current_frame
        for i in range(1, 3):
            pygame.draw.circle(
                surface,
                explosion.color,
                translate(explosion.pos),
                radius=(radius + i * random.uniform(0.0, 0.00375)) * WIDTH,
                width=1,
            )


@ecs.System
def draw_saucer(saucer: Saucer, surface: pygame.Surface, **_) -> None:
    """Draw a saucer."""
    if saucer.alive:
        for section in Saucer.SECTIONS:
            draw_radial_points(
                surface, saucer.color, section, pos=saucer.pos, radius=saucer.radius
            )


@ecs.System
def draw_ship(ship: Ship, surface: pygame.Surface, **_) -> None:
    """Draw a ship."""
    if (
        ship.alive
        # flash while invulnerable
        and (ship.invulnerability.remaining // 100) % 3 != 1
    ):
        for part in (Ship.FUSELAGE, Ship.WINGS, Ship.EXHAUST):
            if part is Ship.EXHAUST and not ship.thruster:
                continue
            draw_radial_points(
                surface,
                ship.color,
                part,
                pos=ship.pos,
                radius=ship.radius,
                angle=ship.angle,
            )

        if ship.shield.active:
            pygame.draw.circle(
                surface,
                ship.shield.color,
                translate(ship.pos),
                radius=(ship.radius + random.uniform(0, 0.00375)) * WIDTH,
                width=1,
            )


if __name__ == "__main__":
    # Settings.mute = True
    Game(
        initial_screen=MainMenu,
        window_size=(WIDTH, HEIGHT),
        window_title="Asteroids",
    ).start()
