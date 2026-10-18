# Training records

`dev-run/` is the development training run behind the numbers in the top-level README
(4 CPU cores, Ironclad, Ascension 8):

| file | what it is |
|---|---|
| `learning_curve.md` | evaluation after each training iteration (64 selection seeds) |
| `metrics.jsonl` | the full per-iteration record: self-play, training loss, evaluation |
| `train.log` | the trainer's console output |
| `config.json` | the command-line settings it ran with |
| `final-eval/` | held-out evaluation on 128 unseen seeds (`*_s48` / `*_s128` = combat simulations per move), per-run results included |
| `shards/` | the self-play data itself (every recorded state and how its run ended) |

The checkpoint it produced is `models/best.pt`. To continue that exact run:

```bash
mkdir -p runs/main && cp -r results/dev-run/shards runs/main/ && cp results/dev-run/metrics.jsonl runs/main/
cp models/best.pt runs/main/latest.pt && cp models/best.pt runs/main/best.pt
python -m sts2rl.train --name main --workers $(nproc)
```
