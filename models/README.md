# Trained checkpoints

`best.pt` is the value network from the development training run (`runs/main`), selected
by evaluation score on seeds `EVAL000000–000063` and then re-measured on held-out seeds
`EVAL001000+` (see `best.json`). It is a starting point, not a finished agent: continue
training from it with

```bash
mkdir -p runs/main && cp models/best.pt runs/main/latest.pt
python -m sts2rl.train --name main --workers $(nproc)
```

or use it directly:

```bash
python -m sts2rl.watch    --model models/best.pt --seed ANYSEED
python -m sts2rl.evaluate --model models/best.pt --runs 200 --offset 1000
python -m sts2rl.bridge.live --model models/best.pt --attach
```

A checkpoint is tied to the simulator's data tables (card/relic/encounter ids); after
updating the simulator to a new game patch, `ValueNet.load` refuses a checkpoint whose
encoding no longer matches, and you retrain.
