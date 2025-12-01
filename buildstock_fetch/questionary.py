from collections.abc import Sequence
from enum import Enum
from typing import TypeVar, cast

import questionary
import typer
from rich.console import Console

# Initialize Rich console
console = Console()


S = TypeVar("S", bound=str)
E = TypeVar("E", bound=Enum)
T = TypeVar("T")


def select(message: str, choices: Sequence[S]) -> S | None:
    return cast(S | None, questionary.select(message, choices).ask())


def select_enum(message: str, choices: type[E]) -> E | None:
    return questionary.select(
        message, choices=[questionary.Choice(title=str(_.value()), value=_) for _ in choices]
    ).ask()


def select_with_cancel(message: str, choices: Sequence[S], cancel_message="Operation cancelled by user.") -> S:
    return handle_cancellation(select(message, choices), cancel_message)


def handle_cancellation(result: T | None, message: str = "Operation cancelled by user.") -> T:
    """Handle user cancellation and exit cleanly"""
    if result is None:
        console.print(f"\n[red]{message}[/red]")
        raise typer.Exit(0) from None
    return result
