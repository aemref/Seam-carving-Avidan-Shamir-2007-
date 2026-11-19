"""Command line interface. Owner: Mirza.

    python -m seamcarving resize INPUT OUTPUT [--width W | --dw N] [--height H | --dh N]
                                              [--energy sobel] [--protect MASK]
    python -m seamcarving remove INPUT MASK OUTPUT [--protect MASK] [--no-restore]
    python -m seamcarving energy INPUT OUTPUT [--method sobel]
    python -m seamcarving seams  INPUT OUTPUT --dw N
"""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the process exit code."""
    raise NotImplementedError
