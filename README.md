# Background Remover for Videos

Remove backgrounds from MP4 videos using AI (U²-Net), with GPU acceleration.

## Installation

```bash
cd bg-remover
pip install -e .
```

## Usage

**Remove background (Green Screen output):**
```bash
bg-remover input.mp4 output.mp4
```

**Output transparent PNG frames:**
```bash
bg-remover input.mp4 output_dir --frames
```

### Options

| Flag | Description |
|------|-------------|
| `--no-gpu` | Force CPU usage |
| `--model <name>` | Select model (default: `u2net`) |

**Available Models:** `u2net` (best), `u2netp` (fast), `u2net_human_seg` (human), `u2net_cloth_seg` (cloth), `silueta`.

## License

MIT License
