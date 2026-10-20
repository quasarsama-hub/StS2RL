# Trained checkpoints

`best.pt` is the value network from the development training run (`runs/main`), selected
by evaluation score on validation seeds `EVAL000000–000063` and then measured on held-out
test seeds `EVAL001000+` (see `best.json`). It was trained before simulator patch 0006
gave treasure chests their relic, and while the agent could still walk away from events
(which the game does not allow); `best.json` → `results_patch0006` re-measures it with
both fixed. It is a starting point, not a finished agent: continue training from it with

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
