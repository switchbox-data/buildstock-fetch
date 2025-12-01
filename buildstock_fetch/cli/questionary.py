from collections.abc import Sequence
from enum import EnumType
from typing import Any, Callable, TypeVar, cast, overload

import questionary
import typer
from rich.console import Console

# Initialize Rich console
console = Console()


S = TypeVar("S", bound=str)
E = TypeVar("E", bound=EnumType)
T = TypeVar("T")


def select_str(message: str, choices: Sequence[S]) -> S | None:
    return cast(S | None, questionary.select(message, choices).ask())


def select_enum(message: str, choices: type[E]) -> E | None:
    return questionary.select(
        message, choices=[questionary.Choice(title=str(_.value()), value=_) for _ in choices]
    ).ask()


@overload
def select(message: str, choices: Sequence[S]) -> S | None: ...


@overload
def select(message: str, choices: type[E]) -> E | None: ...


def select(message: str, choices: Any) -> Any:
    if type(choices) is EnumType:
        return select_enum(message, choices)
    if hasattr(choices, "__iter__") and all(isinstance(_, str) for _ in choices):
        return select_str(message, choices)
    raise ValueError("choices")


def checkbox(
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
