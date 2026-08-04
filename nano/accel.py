"""Device and distributed handling, via Accelerate.

    python -m nano.models.qwen_nano                              # CPU / MPS / one GPU
    accelerate launch -m nano.models.qwen_nano --size small      # many GPUs, FSDP, DeepSpeed
    torchrun --nproc_per_node=8 -m nano.models.qwen_nano         # still works

One path, whatever you launch with. Accelerate figures out the device, the
process group and the mixed precision, so the training loops no longer carry
`dist.init_process_group`, a DDP wrapper, a DistributedSampler, an autocast
context, or a hand-rolled device search. That is roughly forty lines of
boilerplate per model file replaced by `accelerator.prepare(...)`.

It also makes the distributed path *testable without GPUs*:

    accelerate launch --cpu --num_processes 2 -m nano.models.qwen_next_nano

which is how the auxiliary-loss gradient sync actually got checked — MTP's loss
has to be computed inside the wrapped forward or its gradients never leave rank 0.

`accelerate` is a training dependency; the architectures and every self-check
still run on torch alone.
"""

_ACCELERATOR = None


def accelerator(mixed_precision=None, gradient_accumulation_steps=1):
    """The process-wide Accelerator, created once.

    A singleton because Accelerate keeps global state — building a second one
    with different settings silently ignores the new settings.
    """
    global _ACCELERATOR
    if _ACCELERATOR is None:
        try:
            from accelerate import Accelerator
        except ImportError as e:
            raise SystemExit(f"Training needs `accelerate`: pip install accelerate ({e})") from None
        _ACCELERATOR = Accelerator(
            mixed_precision=mixed_precision or "no",
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    return _ACCELERATOR


def is_main_process():
    return _ACCELERATOR.is_main_process if _ACCELERATOR else True


def get_world_size():
    return _ACCELERATOR.num_processes if _ACCELERATOR else 1


def get_rank():
    return _ACCELERATOR.process_index if _ACCELERATOR else 0


def is_distributed():
    return get_world_size() > 1


def log(msg):
    """Print on rank 0 only."""
    if is_main_process():
        print(msg)


def wait():
    if _ACCELERATOR is not None:
        _ACCELERATOR.wait_for_everyone()


def unwrap(model):
    """The bare module behind any DDP/FSDP wrapper."""
    return _ACCELERATOR.unwrap_model(model) if _ACCELERATOR else model


def precision_for(use_amp, device_type=None):
    """Accelerate's mixed-precision name for this repo's `use_amp` setting.

    bf16 is what the loops used before; on CPU Accelerate accepts it but it is
    slower than fp32 there, so single-process CPU runs stay in full precision.
    """
    if not use_amp:
        return "no"
    if device_type == "cpu":
        return "no"
    return "bf16"
