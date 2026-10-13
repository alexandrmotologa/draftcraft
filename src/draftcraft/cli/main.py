"""DraftCraft CLI entrypoint."""

import typer
from draftcraft.cli.commands import (
    check_tokenizer_cmd,
    matrix_cmd,
    profile_cmd,
    serve_cmd,
    show_info,
    train_cmd,
)

app = typer.Typer(
    name="draftcraft",
    help="Turnkey speculative decoding matrix evaluator and EAGLE draft head trainer for local LLMs.",
    add_completion=False,
)

app.command("info")(show_info)
app.command("check-tokenizer")(check_tokenizer_cmd)
app.command("profile")(profile_cmd)
app.command("matrix")(matrix_cmd)
app.command("train")(train_cmd)
app.command("serve")(serve_cmd)


if __name__ == "__main__":
    app()
