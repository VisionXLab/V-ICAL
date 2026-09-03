"""One-time initialization for modules that are lazily loaded by dependencies."""


def ensure_pygame_surfarray():
    """Eagerly finish pygame.surfarray loading before concurrent workers start."""
    import pygame
    import pygame.surfarray

    pixels3d = getattr(pygame.surfarray, "pixels3d", None)
    if not callable(pixels3d):
        raise RuntimeError(
            "pygame.surfarray failed to initialize pixels3d; "
            "check the pygame/numpy installation"
        )

    return pygame.surfarray
