"""DraftCraft CLI entrypoint."""

import typer
from draftcraft.cli.commands import (
    check_tokenizer_cmd,
    export_cmd,
    matrix_cmd,
    profile_cmd,
    serve_cmd,
    show_info,
    temp_profile_cmd,
    train_cmd,
    vram_cmd,
    wizard_cmd,
)

app = typer.Typer(
    name="draftcraft",
    help="Turnkey speculative decoding matrix evaluator and EAGLE draft head trainer for local LLMs.",
    add_completion=False,
)

app.command("info")(show_info)
app.command("wizard")(wizard_cmd)
app.command("check-tokenizer")(check_tokenizer_cmd)
app.command("profile")(profile_cmd)
app.command("matrix")(matrix_cmd)
app.command("vram")(vram_cmd)
app.command("temp-profile")(temp_profile_cmd)
app.command("export")(export_cmd)
app.command("train")(train_cmd)
app.command("serve")(serve_cmd)


if __name__ == "__main__":
    app()
