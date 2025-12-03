from collections.abc import Sequence
from typing import Callable, TypeVar, cast

import questionary
import typer
from rich.console import Console

# Initialize Rich console
console = Console()


S = TypeVar("S", bound=str)
T = TypeVar("T")


def select(message: str, choices: Sequence[S]) -> S | None:
    return cast(S | None, questionary.select(message, choices).ask())


def checkbox_str(
    message: str,
    choices: Sequence[S],
    instruction="Use spacebar to select/deselect options, 'a' to select all, 'i' to invert selection, enter to confirm",
    validate: Callable[[list[str]], str | bool] = lambda _: True,
) -> list[S]:
    return questionary.checkbox(message, choices=choices, instruction=instruction, validate=validate).ask()


def handle_cancellation(result: T | None, message: str = "Operation cancelled by user.") -> T:
    """Handle user cancellation and exit cleanly"""
    if result is None:
        console.print(f"\n[red]{message}[/red]")
        raise typer.Exit(0) from None
    return result
