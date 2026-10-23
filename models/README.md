# Trained checkpoints

`best.pt` is the value network from the development training run (`runs/main`, iteration
8), fine-tuned for one iteration (`runs/v2`, 400 self-play runs) on the simulator with
patches 0001–0015. It replaced the `runs/main` network in the paired test on validation
seeds `EVAL000000–000127` and then on held-out test seeds `EVAL001000+`: Act 1 cleared on
89 of 128 test seeds to 72, at 48 sims and 2 samples (see `best.json`; with 3 samples, 79
to 72). `best.json` → `previous_checkpoint` keeps the old network's numbers. It is a
starting point, not a finished agent: continue training from it with

```bash
mkdir -p runs/main && cp models/best.pt runs/main/latest.pt
python -m sts2rl.train --name main --workers $(nproc)
```

The run takes this checkpoint as its first champion, so a fine-tune replaces it only by
beating it in the paired test on the validation seeds.

or use it directly:

```bash
python -m sts2rl.watch    --model models/best.pt --seed ANYSEED
python -m sts2rl.evaluate --model models/best.pt --runs 200 --split test
python -m sts2rl.bridge.cli_eval --cli third_party/sts2-cli --model models/best.pt --runs 32
python -m sts2rl.bridge.live --model models/best.pt --attach
```

A checkpoint is tied to the simulator's data tables (card/relic/encounter ids); after
updating the simulator to a new game patch, `ValueNet.load` refuses a checkpoint whose
encoding no longer matches, and you retrain.
