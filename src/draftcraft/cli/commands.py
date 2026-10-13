import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import torch

from draftcraft import __version__
from draftcraft.alignment.file_loader import TokenizerFileLoader
from draftcraft.alignment.tokenizer_verifier import TokenizerAlignmentVerifier
from draftcraft.core.config import DraftCraftConfig, EagleDraftHeadConfig, EagleTrainConfig, ProfilerConfig
from draftcraft.core.types import AlignmentStatus, DomainCategory
from draftcraft.models.draft_head import EagleDraftHead
from draftcraft.profiler.acceptance_math import AcceptanceMath
from draftcraft.profiler.exporter import BenchmarkExporter
from draftcraft.profiler.matrix_evaluator import MatrixEvaluator
from draftcraft.profiler.speculative_profiler import SpeculativeProfiler
from draftcraft.profiler.temperature_profiler import TemperatureProfiler
from draftcraft.profiler.vram_calculator import HARDWARE_PRESETS, PrecisionType, VRAMCalculator
from draftcraft.trainer.dataset import SyntheticFeatureGenerator
from draftcraft.trainer.eagle_trainer import EagleTrainer

console = Console()


def show_info() -> None:
    """Print system environment, PyTorch configuration, and DraftCraft metadata."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"

    table = Table(title="DraftCraft System Environment", border_style="cyan")
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("DraftCraft Version", __version__)
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Acceleration", "Enabled" if cuda_avail else "Disabled (CPU mode)")
    table.add_row("Hardware Device", device_name)
    table.add_row("PyTorch Threads", str(torch.get_num_threads()))

    console.print(table)


def check_tokenizer_cmd(
    target_vocab_size: int = typer.Option(128256, "--target-size", "-t", help="Target model vocabulary size"),
    draft_vocab_size: int = typer.Option(128256, "--draft-size", "-d", help="Draft model or head vocabulary size"),
    target_hidden_dim: int = typer.Option(4096, "--target-dim", help="Target hidden state dimension"),
    head_hidden_dim: int = typer.Option(1024, "--head-dim", help="Draft head hidden state dimension"),
    eagle_head: bool = typer.Option(True, "--eagle/--separate-llm", help="Speculate with EAGLE head vs standalone LLM"),
    target_file: Optional[Path] = typer.Option(None, "--target-file", help="Path to target tokenizer.json file"),
    draft_file: Optional[Path] = typer.Option(None, "--draft-file", help="Path to draft tokenizer.json file"),
    export_remap: Optional[Path] = typer.Option(None, "--export-remap", help="Path to export vocab_remap.json"),
) -> None:
    """Verify tokenizer and vocabulary alignment for speculative decoding."""
    console.print(Panel("[bold cyan]DraftCraft Tokenizer Alignment Verifier[/bold cyan]", border_style="cyan"))

    if target_file and draft_file:
        console.print(f"[dim]Loading tokenizer files:\n  Target: {target_file}\n  Draft: {draft_file}[/dim]")
        summary = TokenizerFileLoader.cross_verify_files(
            target_file=target_file,
            draft_file=draft_file,
            export_remap_path=export_remap,
        )
    elif eagle_head:
        summary = TokenizerAlignmentVerifier.verify_eagle_head(
            target_vocab_size=target_vocab_size,
            head_vocab_size=draft_vocab_size,
            target_hidden_dim=target_hidden_dim,
            head_hidden_dim=head_hidden_dim,
        )
    else:
        target_vocab = {f"tok_{i}": i for i in range(target_vocab_size)}
        draft_vocab = {f"tok_{i}": i for i in range(draft_vocab_size)}
        verifier = TokenizerAlignmentVerifier(target_vocab, draft_vocab)
        summary = verifier.verify_vocabularies()

    status_color = "green" if summary.status == AlignmentStatus.FULLY_ALIGNED else "yellow" if summary.status == AlignmentStatus.SUBSET_COMPATIBLE else "red"

    table = Table(title="Alignment Verification Results", border_style="white")
    table.add_column("Metric", style="bold")
    table.add_column("Result", style=status_color)

    table.add_row("Alignment Status", summary.status.value.upper())
    table.add_row("Target Vocab Size", f"{summary.target_vocab_size:,}")
    table.add_row("Draft Vocab Size", f"{summary.draft_vocab_size:,}")
    table.add_row("Vocabulary Overlap Ratio", f"{summary.overlap_ratio * 100:.2f}%")
    table.add_row("Shared Tokens Count", f"{summary.shared_tokens_count:,}")
    table.add_row("Divergent Sample Count", str(summary.divergent_sample_count))
    table.add_row("Special Tokens Matched", str(summary.special_tokens_matched))
    table.add_row("Prefix Consistency Score", f"{summary.subword_prefix_consistency * 100:.2f}%")

    console.print(table)

    if summary.status == AlignmentStatus.FULLY_ALIGNED:
        console.print("[bold green][OK] Tokenizer alignment verified: Speculative decoding is fully compatible with zero token mapping overhead.[/bold green]")
    else:
        console.print(f"[bold {status_color}][!] Warning: Vocabulary mismatch detected. Alignment status is {summary.status.value}.[/bold {status_color}]")

    if export_remap:
        console.print(f"[bold green]Saved vocabulary remapping table to {export_remap}[/bold green]")


def profile_cmd(
    alpha: float = typer.Option(0.75, "--alpha", "-a", help="Initial token acceptance rate (0.0 - 1.0)"),
    depth: int = typer.Option(4, "--depth", "-k", help="Speculative lookahead depth K"),
    draft_latency: float = typer.Option(3.5, "--draft-latency", "-dl", help="Draft latency in ms"),
    target_latency: float = typer.Option(28.0, "--target-latency", "-tl", help="Target latency in ms"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Hardware preset (e.g. rtx-4090, rtx-4060-8gb)"),
    domain: DomainCategory = typer.Option(DomainCategory.GENERAL, "--domain", help="Corpus domain"),
    steps: int = typer.Option(60, "--steps", help="Number of profiling cycles"),
) -> None:
    """Profile speculative decoding acceptance rates and empirical speedup."""
    if preset and preset in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[preset]
        target_latency = hw.typical_target_latency_ms
        draft_latency = hw.typical_draft_latency_ms
        console.print(f"[dim]Loaded hardware preset: {hw.name} (Target: {target_latency}ms, Draft: {draft_latency}ms)[/dim]")

    config = ProfilerConfig(
        draft_latency_ms=draft_latency,
        target_latency_ms=target_latency,
        max_lookahead=depth,
        domain=domain,
    )
    profiler = SpeculativeProfiler(config)

    console.print(f"[bold cyan]Running Speculative Profiling Sweep ({domain.value.upper()})...[/bold cyan]")
    metrics = profiler.profile_simulated_stream(
        base_acceptance_rate=alpha,
        lookahead_depth=depth,
        num_steps=steps,
        vocab_size=32000,
    )

    table = Table(title="Speculative Decoding Performance Profile", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Domain Category", domain.value.upper())
    table.add_row("Lookahead Depth (K)", str(metrics.lookahead_depth))
    table.add_row("Empirical Acceptance Rate (alpha)", f"{metrics.alpha_acceptance_rate * 100:.2f}%")
    table.add_row("Expected Accepted Tokens / Step", f"{metrics.expected_accepted_tokens:.2f}")
    table.add_row("Target Latency per Token", f"{metrics.target_latency_ms:.1f} ms")
    table.add_row("Draft Latency per Token", f"{metrics.draft_latency_ms:.1f} ms")
    table.add_row("Latency Ratio (c = t_draft / t_target)", f"{metrics.latency_ratio:.3f}")
    table.add_row("Baseline Throughput", f"{metrics.tokens_per_second_baseline:.1f} tokens/s")
    table.add_row("Speculative Throughput", f"{metrics.tokens_per_second_speculative:.1f} tokens/s")
    table.add_row("Empirical Wallclock Speedup", f"[bold yellow]{metrics.wallclock_speedup:.2f}x[/bold yellow]")

    console.print(table)

    pos_table = Table(title="Positional Acceptance Degradation Curve", border_style="dim")
    for i in range(len(metrics.acceptance_by_position)):
        pos_table.add_column(f"Pos {i+1}", justify="center")
    pos_table.add_row(*[f"{p*100:.1f}%" for p in metrics.acceptance_by_position])
    console.print(pos_table)


def matrix_cmd(
    domain: DomainCategory = typer.Option(DomainCategory.GENERAL, "--domain", help="Evaluation domain"),
    alpha: float = typer.Option(0.75, "--alpha", "-a", help="Base acceptance rate"),
    draft_latency: float = typer.Option(3.5, "--draft-latency", "-dl", help="Draft latency in ms"),
    target_latency: float = typer.Option(28.0, "--target-latency", "-tl", help="Target latency in ms"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Hardware preset (e.g. rtx-4090, rtx-4060-8gb)"),
    min_depth: int = typer.Option(1, "--min-depth", help="Minimum lookahead depth"),
    max_depth: int = typer.Option(8, "--max-depth", help="Maximum lookahead depth"),
) -> None:
    """Generate speculative decoding matrix across depths, widths, and speedups."""
    if preset and preset in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[preset]
        target_latency = hw.typical_target_latency_ms
        draft_latency = hw.typical_draft_latency_ms
        console.print(f"[dim]Loaded hardware preset: {hw.name} (Target: {target_latency}ms, Draft: {draft_latency}ms)[/dim]")

    config = ProfilerConfig(
        min_lookahead=min_depth,
        max_lookahead=max_depth,
        draft_latency_ms=draft_latency,
        target_latency_ms=target_latency,
        tree_widths=[1, 2, 4],
        domain=domain,
    )
    evaluator = MatrixEvaluator(config)
    result = evaluator.evaluate_matrix(
        domain=domain,
        alpha=alpha,
        draft_latency_ms=draft_latency,
        target_latency_ms=target_latency,
    )

    table = Table(title=f"Speculative Decoding Matrix ({domain.value.upper()})", border_style="cyan")
    table.add_column("Depth (K)", justify="center", style="bold")
    table.add_column("Width (W)", justify="center")
    table.add_column("Exp. Tokens", justify="right")
    table.add_column("Speedup", justify="right", style="bold")
    table.add_column("Pareto Optimal", justify="center")

    for pt in result.points:
        sp_style = "bold yellow" if pt.speedup == result.peak_speedup else "green" if pt.speedup >= 1.5 else "white"
        pareto_badge = "[bold cyan]* OPTIMAL[/bold cyan]" if pt.is_pareto_optimal else "-"
        table.add_row(
            str(pt.lookahead_depth),
            f"W={pt.tree_width} {'(Chain)' if pt.tree_width == 1 else '(Tree)'}",
            f"{pt.expected_tokens:.2f}",
            f"[{sp_style}]{pt.speedup:.2f}x[/{sp_style}]",
            pareto_badge,
        )

    console.print(table)

    opt = result.optimal_point
    recommendation_panel = Panel(
        f"[bold green]Pareto Optimal Configuration Found[/bold green]\n\n"
        f"- Recommended Lookahead Depth (K*): [bold white]{opt.lookahead_depth}[/bold white]\n"
        f"- Recommended Tree Width (W*): [bold white]{opt.tree_width}[/bold white]\n"
        f"- Peak Acceleration: [bold yellow]{opt.speedup:.2f}x speedup[/bold yellow]\n"
        f"- Expected Tokens / Step: [bold white]{opt.expected_tokens:.2f}[/bold white]",
        border_style="green",
    )
    console.print(recommendation_panel)


def vram_cmd(
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Hardware preset (e.g. rtx-4090, rtx-4060-8gb, rtx-3060-12gb)"),
    params: float = typer.Option(8.0, "--params", help="Target model parameters in billions"),
    precision: PrecisionType = typer.Option(PrecisionType.Q4_K_M, "--precision", help="Target quantization"),
    context_len: int = typer.Option(4096, "--context", help="Context length in tokens"),
    depth: int = typer.Option(4, "--depth", "-k", help="Lookahead depth K"),
    tree_width: int = typer.Option(1, "--width", "-w", help="Tree width (1 for chain)"),
    vram_capacity: Optional[float] = typer.Option(None, "--vram", help="Physical GPU VRAM capacity in GB"),
) -> None:
    """Calculate VRAM memory footprint and evaluate GPU budget headroom."""
    cap = vram_capacity or 12.0
    hw_name = "Custom GPU"
    if preset and preset in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[preset]
        cap = hw.vram_gb
        hw_name = hw.name

    console.print(Panel(
        f"[bold cyan]DraftCraft VRAM & KV Cache Budget Analyzer[/bold cyan]\n"
        f"Hardware: {hw_name} ({cap:.1f} GB VRAM) | Base Model: {params:.1f}B ({precision.value.upper()}) | Context: {context_len:,} tok",
        border_style="cyan",
    ))

    breakdown = VRAMCalculator.calculate_memory(
        target_param_count_billions=params,
        target_precision=precision,
        context_len=context_len,
        lookahead_k=depth,
        tree_width=tree_width,
        gpu_capacity_gb=cap,
    )

    fit_color = "green" if breakdown.fits_in_gpu else "red"

    table = Table(title="VRAM Memory Allocation Breakdown", border_style="white")
    table.add_column("Component", style="bold white")
    table.add_column("Allocation (GB)", justify="right")

    table.add_row("Target Model Weights", f"{breakdown.target_weights_gb:.2f} GB")
    table.add_row("EAGLE Draft Head Weights", f"{breakdown.draft_head_weights_gb:.2f} GB")
    table.add_row("Base Context KV Cache", f"{breakdown.kv_cache_baseline_gb:.2f} GB")
    table.add_row("Speculative Extra KV Cache", f"{breakdown.kv_cache_speculative_overhead_gb:.2f} GB")
    table.add_row("Activations & CUDA Overhead", f"{breakdown.activations_and_cuda_overhead_gb:.2f} GB")
    table.add_row("[bold]Total VRAM Required[/bold]", f"[{fit_color} bold]{breakdown.total_vram_gb:.2f} GB[/{fit_color} bold]")
    table.add_row("Physical GPU Capacity", f"{breakdown.gpu_vram_capacity_gb:.2f} GB")
    table.add_row("Remaining Headroom", f"[{fit_color}]{breakdown.headroom_gb:.2f} GB[/{fit_color}]")

    console.print(table)

    status_title = "[bold green]BUDGET VERIFIED[/bold green]" if breakdown.fits_in_gpu else "[bold red]VRAM EXCEEDED[/bold red]"
    rec_panel = Panel(
        f"{status_title}\n\n"
        f"{breakdown.recommendation}\n"
        f"Theoretical Maximum Context Supported: [bold white]{breakdown.max_context_length_tokens:,}[/bold white] tokens",
        border_style=fit_color,
    )
    console.print(rec_panel)


def temp_profile_cmd(
    alpha: float = typer.Option(0.85, "--alpha", "-a", help="Base acceptance rate at T=0.0 (greedy)"),
    depth: int = typer.Option(4, "--depth", "-k", help="Lookahead depth K"),
    draft_latency: float = typer.Option(3.5, "--draft-latency", "-dl", help="Draft latency in ms"),
    target_latency: float = typer.Option(28.0, "--target-latency", "-tl", help="Target latency in ms"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Hardware preset"),
) -> None:
    """Evaluate acceptance rate degradation across sampling temperatures (T=0.0 to 1.0)."""
    if preset and preset in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[preset]
        target_latency = hw.typical_target_latency_ms
        draft_latency = hw.typical_draft_latency_ms

    console.print(Panel(
        f"[bold cyan]Temperature Sensitivity Sweep (T = 0.0 to 1.0)[/bold cyan]\n"
        f"Greedy Alpha: {alpha*100:.1f}% | Depth K: {depth} | Target: {target_latency}ms | Draft: {draft_latency}ms",
        border_style="cyan",
    ))

    res = TemperatureProfiler.profile_temperature_curve(
        base_greedy_alpha=alpha,
        lookahead_depth=depth,
        draft_latency_ms=draft_latency,
        target_latency_ms=target_latency,
    )

    table = Table(title="Temperature vs Acceptance & Acceleration", border_style="white")
    table.add_column("Temperature (T)", justify="center", style="bold")
    table.add_column("Acceptance Rate (alpha)", justify="right")
    table.add_column("Exp. Tokens", justify="right")
    table.add_column("Speedup", justify="right", style="bold")
    table.add_column("Viability", justify="center")

    for pt in res.points:
        sp_style = "bold yellow" if pt.speedup == res.peak_speedup else "green" if pt.speedup >= 1.5 else "white"
        viab = "[bold green]ACCELERATING[/bold green]" if pt.is_viable else "[bold red]DEGRADED[/bold red]"
        table.add_row(
            f"T = {pt.temperature:.2f}",
            f"{pt.acceptance_rate*100:.1f}%",
            f"{pt.expected_tokens:.2f}",
            f"[{sp_style}]{pt.speedup:.2f}x[/{sp_style}]",
            viab,
        )

    console.print(table)
    console.print(Panel(
        f"[bold green]Temperature Guidance[/bold green]\n\n"
        f"- Peak Acceleration: [bold yellow]{res.peak_speedup:.2f}x[/bold yellow] at T={res.peak_temperature:.1f}\n"
        f"- {res.recommendation}",
        border_style="green",
    ))


def export_cmd(
    output: Path = typer.Option(Path("draftcraft_report.md"), "--output", "-o", help="Output path (.md, .json, .csv)"),
    domain: DomainCategory = typer.Option(DomainCategory.GENERAL, "--domain", help="Corpus domain"),
    alpha: float = typer.Option(0.75, "--alpha", "-a", help="Base acceptance rate"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Hardware preset"),
) -> None:
    """Export benchmark report or production vLLM configuration to disk."""
    d_lat = 3.5
    t_lat = 28.0
    cap = 12.0
    if preset and preset in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[preset]
        d_lat = hw.typical_draft_latency_ms
        t_lat = hw.typical_target_latency_ms
        cap = hw.vram_gb

    evaluator = MatrixEvaluator(ProfilerConfig(draft_latency_ms=d_lat, target_latency_ms=t_lat, domain=domain))
    matrix_res = evaluator.evaluate_matrix(domain=domain, alpha=alpha, draft_latency_ms=d_lat, target_latency_ms=t_lat)

    vram_breakdown = VRAMCalculator.calculate_memory(gpu_capacity_gb=cap)
    temp_res = TemperatureProfiler.profile_temperature_curve(draft_latency_ms=d_lat, target_latency_ms=t_lat)

    BenchmarkExporter.save_report(
        output_path=output,
        matrix_result=matrix_res,
        vram_breakdown=vram_breakdown,
        temp_result=temp_res,
    )
    console.print(f"[bold green]Successfully exported benchmark report to {output}[/bold green]")


def train_cmd(
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    hidden_dim: int = typer.Option(256, "--hidden-dim", help="Draft head hidden dimension"),
    target_dim: int = typer.Option(512, "--target-dim", help="Target LLM hidden dimension"),
    vocab_size: int = typer.Option(2000, "--vocab-size", help="Vocabulary size"),
    num_samples: int = typer.Option(64, "--samples", help="Number of training samples"),
    checkpoint_dir: str = typer.Option("checkpoints", "--out-dir", "-o", help="Checkpoint output folder"),
) -> None:
    """Train an EAGLE speculative draft head on local hardware."""
    console.print(Panel(
        f"[bold cyan]DraftCraft EAGLE Head Training[/bold cyan]\n"
        f"Hidden: {hidden_dim} | Target Dim: {target_dim} | Vocab: {vocab_size} | Epochs: {epochs}",
        border_style="cyan",
    ))

    head_config = EagleDraftHeadConfig(
        hidden_dim=hidden_dim,
        target_hidden_dim=target_dim,
        num_layers=2,
        num_heads=4,
        intermediate_dim=hidden_dim * 2,
        vocab_size=vocab_size,
    )
    model = EagleDraftHead(head_config)

    console.print("[dim]Generating synthetic teacher feature distillation dataset...[/dim]")
    train_dataset = SyntheticFeatureGenerator.generate_synthetic_dataset(
        num_samples=num_samples,
        seq_len=24,
        hidden_dim=target_dim,
        vocab_size=vocab_size,
        seed=42,
    )
    eval_dataset = SyntheticFeatureGenerator.generate_synthetic_dataset(
        num_samples=max(16, num_samples // 4),
        seq_len=24,
        hidden_dim=target_dim,
        vocab_size=vocab_size,
        seed=1337,
    )

    train_config = EagleTrainConfig(
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        eval_interval=max(1, (num_samples // batch_size)),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Training EAGLE Head...", total=epochs * (len(train_dataset) // batch_size))

        def on_step(info: dict):
            loss = info["total_loss"]
            acc = info["top1_accuracy"] * 100
            progress.update(task_id, advance=1, description=f"[cyan]Epoch {info['epoch']} | Loss: {loss:.4f} | Top1 Acc: {acc:.1f}%")

        trainer = EagleTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_config=train_config,
            step_callback=on_step,
        )

        result = trainer.train()

    console.print(Panel(
        f"[bold green]Training Completed Successfully[/bold green]\n\n"
        f"- Total Steps: {result['total_steps']}\n"
        f"- Final Loss: {result['final_loss']:.4f}\n"
        f"- Checkpoint Saved: [bold white]{result['final_checkpoint']}[/bold white]",
        border_style="green",
    ))


def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Launch the FastAPI speculative decoding web dashboard and REST API."""
    import uvicorn
    console.print(Panel(
        f"[bold cyan]Starting DraftCraft Web Dashboard & API[/bold cyan]\n"
        f"Listening on [bold white]http://{host}:{port}[/bold white]\n"
        f"Interactive UI: [bold white]http://{host}:{port}/[/bold white]\n"
        f"API Docs: [bold white]http://{host}:{port}/docs[/bold white]",
        border_style="cyan",
    ))
    uvicorn.run("draftcraft.web.app:app", host=host, port=port, reload=reload)


WIZARD_MODEL_PROFILES = {
    "llama3-8b": {
        "title": "Meta-Llama-3-8B-Instruct",
        "params_b": 8.0,
        "layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 128256,
        "acceptance": 0.74,
    },
    "mistral-7b": {
        "title": "Mistral-7B-Instruct-v0.3",
        "params_b": 7.24,
        "layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 32768,
        "acceptance": 0.72,
    },
    "qwen2.5-7b": {
        "title": "Qwen2.5-7B-Instruct",
        "params_b": 7.61,
        "layers": 28,
        "hidden_dim": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "acceptance": 0.73,
    },
    "llama3-70b": {
        "title": "Meta-Llama-3-70B-Instruct",
        "params_b": 70.0,
        "layers": 80,
        "hidden_dim": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "vocab_size": 128256,
        "acceptance": 0.68,
    },
}


def wizard_cmd(
    model_choice: Optional[str] = typer.Option(None, "--model", "-m", help="Target model key (llama3-8b, mistral-7b, qwen2.5-7b, llama3-70b)"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Hardware preset (rtx-4090, rtx-4080, rtx-4070-ti, rtx-4060-8gb, rtx-3060-12gb, apple-m3-max)"),
    precision: str = typer.Option("fp16", "--precision", help="Target precision (fp16, int8, int4, q4_k_m)"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Serving concurrency / batch size"),
    context_length: int = typer.Option(4096, "--context-len", help="Context length tokens"),
    export_dir: Optional[Path] = typer.Option(None, "--export-dir", "-o", help="Optional directory to save plan and configs"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Bypass interactive prompts and use defaults"),
) -> None:
    """Interactive wizard to configure, budget, and export speculative decoding setups."""
    from rich.prompt import Prompt

    console.print(Panel(
        "[bold cyan]DraftCraft Speculative Decoding Deployment Wizard[/bold cyan]\n"
        "[dim]Design and budget an optimal speculative decoding setup in 3 simple steps.[/dim]",
        border_style="cyan",
    ))

    # Step 1: Select Model
    m_keys = list(WIZARD_MODEL_PROFILES.keys())
    if not model_choice or model_choice.lower() not in WIZARD_MODEL_PROFILES:
        if non_interactive:
            model_choice = "llama3-8b"
        else:
            console.print("\n[bold yellow]Step 1: Choose your Target LLM Architecture[/bold yellow]")
            for idx, k in enumerate(m_keys, 1):
                m_info = WIZARD_MODEL_PROFILES[k]
                console.print(f"  [{idx}] [bold white]{m_info['title']}[/bold white] ({m_info['params_b']}B params, {m_info['hidden_dim']} hidden dim)")
            sel = Prompt.ask("Select Model", choices=[str(i) for i in range(1, len(m_keys) + 1)], default="1")
            model_choice = m_keys[int(sel) - 1]

    model_info = WIZARD_MODEL_PROFILES[model_choice.lower()]

    # Step 2: Select Hardware Preset
    h_keys = list(HARDWARE_PRESETS.keys())
    if not preset or preset.lower() not in HARDWARE_PRESETS:
        if non_interactive:
            preset = "rtx-4090"
        else:
            console.print("\n[bold yellow]Step 2: Choose your GPU Hardware Setup[/bold yellow]")
            for idx, k in enumerate(h_keys, 1):
                h_info = HARDWARE_PRESETS[k]
                console.print(f"  [{idx}] [bold white]{h_info.name}[/bold white] ({h_info.vram_gb} GB VRAM, {h_info.bandwidth_gb_s} GB/s)")
            h_sel = Prompt.ask("Select Hardware", choices=[str(i) for i in range(1, len(h_keys) + 1)], default="1")
            preset = h_keys[int(h_sel) - 1]

    # Step 3: Precision validation
    try:
        prec = PrecisionType(precision.lower())
    except ValueError:
        prec = PrecisionType.FP16

    hw = HARDWARE_PRESETS[preset.lower()]

    # Run VRAM budget calculation
    budget = VRAMCalculator.calculate_memory(
        target_param_count_billions=model_info["params_b"],
        target_precision=prec,
        target_layers=model_info["layers"],
        target_hidden_dim=model_info["hidden_dim"],
        target_heads=model_info["num_heads"],
        head_hidden_dim=1024,
        head_layers=2,
        vocab_size=model_info["vocab_size"],
        context_len=context_length,
        batch_size=batch_size,
        lookahead_k=4,
        tree_width=1,
        gpu_capacity_gb=hw.vram_gb,
    )

    # Acceptance and Speedup calculation
    acc_rate = model_info["acceptance"]
    lat_target = hw.typical_target_latency_ms
    lat_draft = hw.typical_draft_latency_ms
    opt_k, max_speedup = AcceptanceMath.find_optimal_lookahead(
        alpha=acc_rate,
        draft_latency_ms=lat_draft,
        target_latency_ms=lat_target,
        max_lookahead=8,
    )

    # Present Results Table
    feasibility_color = "green" if budget.fits_in_gpu else "red"
    feasibility_text = "[OK] FEASIBLE (Fits VRAM)" if budget.fits_in_gpu else "[!] EXCEEDS VRAM (Out of Memory)"

    summary_table = Table(title="Speculative Blueprint & Hardware Budget", border_style="cyan")
    summary_table.add_column("Parameter", style="bold white")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Target Model", model_info["title"])
    summary_table.add_row("Hardware Device", hw.name)
    summary_table.add_row("Total GPU VRAM", f"{hw.vram_gb:.1f} GB")
    summary_table.add_row("Target Precision", prec.value.upper())
    summary_table.add_row("Target Weights", f"{budget.target_weights_gb:.2f} GB")
    summary_table.add_row("EAGLE Draft Head", f"{budget.draft_head_weights_gb:.2f} GB")
    summary_table.add_row("KV Cache", f"{budget.total_kv_cache_gb:.2f} GB")
    summary_table.add_row("Total Required Memory", f"{budget.total_vram_gb:.2f} GB")
    summary_table.add_row("Available VRAM Headroom", f"{budget.headroom_gb:.2f} GB")
    summary_table.add_row("Deployment Feasibility", f"[{feasibility_color}]{feasibility_text}[/{feasibility_color}]")
    summary_table.add_row("Estimated Acceptance Rate", f"{acc_rate * 100:.1f}%")
    summary_table.add_row("Optimal Speculative K*", f"{opt_k} tokens")
    summary_table.add_row("Expected Wall-Clock Speedup", f"{max_speedup:.2f}x")

    console.print(summary_table)

    vllm_cmd = (
        f"python -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {model_choice} \\\n"
        f"  --speculative-model draftcraft/{model_choice}-eagle-head \\\n"
        f"  --num-speculative-tokens {opt_k} \\\n"
        f"  --gpu-memory-utilization 0.90 \\\n"
        f"  --max-model-len {context_length}"
    )

    console.print(Panel(
        f"[bold white]Recommended vLLM Serving Command:[/bold white]\n\n"
        f"[green]{vllm_cmd}[/green]",
        title="vLLM Production Command",
        border_style="green",
    ))

    # Export if requested
    if export_dir is not None:
        export_p = Path(export_dir)
        export_p.mkdir(parents=True, exist_ok=True)
        vllm_cfg = {
            "model": model_choice,
            "speculative_model": f"draftcraft/{model_choice}-eagle-head",
            "num_speculative_tokens": opt_k,
            "expected_speedup": round(max_speedup, 2),
            "max_model_len": context_length,
            "gpu_memory_utilization": 0.90,
            "vram_budget_gb": budget.model_dump(),
        }
        (export_p / "vllm_speculative_config.json").write_text(json.dumps(vllm_cfg, indent=2), encoding="utf-8")
        console.print(f"[dim]Exported speculative blueprint to {export_p / 'vllm_speculative_config.json'}[/dim]")
