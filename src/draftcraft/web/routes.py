"""FastAPI REST API routes and endpoints for DraftCraft."""

import asyncio
import sys
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import torch

from draftcraft import __version__
from draftcraft.alignment.tokenizer_verifier import TokenizerAlignmentVerifier
from draftcraft.core.config import EagleDraftHeadConfig, EagleTrainConfig, ProfilerConfig
from draftcraft.core.types import AlignmentStatus, DomainCategory, MatrixEvaluationResult, TokenVerificationSummary
from draftcraft.models.draft_head import EagleDraftHead
from draftcraft.profiler.matrix_evaluator import MatrixEvaluator
from draftcraft.profiler.speculative_profiler import SpeculativeProfiler
from draftcraft.trainer.dataset import SyntheticFeatureGenerator
from draftcraft.trainer.eagle_trainer import EagleTrainer

router = APIRouter(prefix="/api")

# In-memory training job state registry
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


class TokenizerVerifyRequest(BaseModel):
    target_vocab_size: int = 128256
    draft_vocab_size: int = 128256
    target_hidden_dim: int = 4096
    head_hidden_dim: int = 1024
    is_eagle: bool = True


class ProfileRequest(BaseModel):
    alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    lookahead_depth: int = Field(default=4, ge=1, le=16)
    draft_latency_ms: float = Field(default=3.5, gt=0.0)
    target_latency_ms: float = Field(default=28.0, gt=0.0)
    domain: DomainCategory = DomainCategory.GENERAL
    num_steps: int = Field(default=60, ge=10, le=200)


class MatrixEvalRequest(BaseModel):
    domain: DomainCategory = DomainCategory.GENERAL
    alpha: Optional[float] = None
    draft_latency_ms: float = 3.5
    target_latency_ms: float = 28.0
    min_lookahead: int = 1
    max_lookahead: int = 8


class TrainJobRequest(BaseModel):
    epochs: int = Field(default=2, ge=1, le=10)
    batch_size: int = Field(default=8, ge=2, le=64)
    lr: float = Field(default=1e-3, gt=0.0)
    hidden_dim: int = Field(default=128, ge=32)
    target_dim: int = Field(default=256, ge=32)
    vocab_size: int = Field(default=1000, ge=100)
    num_samples: int = Field(default=32, ge=8)


@router.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "version": __version__}


@router.get("/info")
def system_info() -> Dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    return {
        "version": __version__,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "threads": torch.get_num_threads(),
    }


@router.post("/verify-tokenizer", response_model=TokenVerificationSummary)
def verify_tokenizer_endpoint(req: TokenizerVerifyRequest) -> TokenVerificationSummary:
    if req.is_eagle:
        return TokenizerAlignmentVerifier.verify_eagle_head(
            target_vocab_size=req.target_vocab_size,
            head_vocab_size=req.draft_vocab_size,
            target_hidden_dim=req.target_hidden_dim,
            head_hidden_dim=req.head_hidden_dim,
        )

    target_vocab = {f"tok_{i}": i for i in range(req.target_vocab_size)}
    draft_vocab = {f"tok_{i}": i for i in range(req.draft_vocab_size)}
    verifier = TokenizerAlignmentVerifier(target_vocab, draft_vocab)
    return verifier.verify_vocabularies()


@router.post("/profile")
def profile_endpoint(req: ProfileRequest) -> Dict[str, Any]:
    config = ProfilerConfig(
        draft_latency_ms=req.draft_latency_ms,
        target_latency_ms=req.target_latency_ms,
        max_lookahead=req.lookahead_depth,
        domain=req.domain,
    )
    profiler = SpeculativeProfiler(config)
    metrics = profiler.profile_simulated_stream(
        base_acceptance_rate=req.alpha,
        lookahead_depth=req.lookahead_depth,
        num_steps=req.num_steps,
    )
    return metrics.model_dump()


@router.post("/matrix-eval", response_model=MatrixEvaluationResult)
def matrix_eval_endpoint(req: MatrixEvalRequest) -> MatrixEvaluationResult:
    config = ProfilerConfig(
        min_lookahead=req.min_lookahead,
        max_lookahead=req.max_lookahead,
        draft_latency_ms=req.draft_latency_ms,
        target_latency_ms=req.target_latency_ms,
        tree_widths=[1, 2, 4],
        domain=req.domain,
    )
    evaluator = MatrixEvaluator(config)
    return evaluator.evaluate_matrix(
        domain=req.domain,
        alpha=req.alpha,
        draft_latency_ms=req.draft_latency_ms,
        target_latency_ms=req.target_latency_ms,
    )


def _run_training_job_sync(job_id: str, req: TrainJobRequest) -> None:
    job = _TRAINING_JOBS[job_id]
    try:
        head_cfg = EagleDraftHeadConfig(
            hidden_dim=req.hidden_dim,
            target_hidden_dim=req.target_dim,
            num_layers=1,
            num_heads=2,
            intermediate_dim=req.hidden_dim * 2,
            vocab_size=req.vocab_size,
        )
        model = EagleDraftHead(head_cfg)

        train_ds = SyntheticFeatureGenerator.generate_synthetic_dataset(
            num_samples=req.num_samples,
            seq_len=16,
            hidden_dim=req.target_dim,
            vocab_size=req.vocab_size,
        )

        train_cfg = EagleTrainConfig(
            learning_rate=req.lr,
            batch_size=req.batch_size,
            epochs=req.epochs,
            checkpoint_dir="checkpoints",
        )

        def step_hook(info: Dict[str, Any]) -> None:
            job["current_epoch"] = info["epoch"]
            job["current_step"] = info["step"]
            job["current_loss"] = info["total_loss"]
            job["current_acc"] = info["top1_accuracy"]
            job["history"].append(info)

        trainer = EagleTrainer(
            model=model,
            train_dataset=train_ds,
            train_config=train_cfg,
            step_callback=step_hook,
        )

        result = trainer.train()
        job["status"] = "completed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@router.post("/train/start")
def start_train_job(req: TrainJobRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())[:8]
    _TRAINING_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "current_epoch": 0,
        "current_step": 0,
        "current_loss": 0.0,
        "current_acc": 0.0,
        "history": [],
    }
    background_tasks.add_task(_run_training_job_sync, job_id, req)
    return {"job_id": job_id, "status": "running"}


@router.get("/train/status/{job_id}")
def get_train_status(job_id: str) -> Dict[str, Any]:
    if job_id not in _TRAINING_JOBS:
        raise HTTPException(status_code=404, detail="Training job not found")
    return _TRAINING_JOBS[job_id]
