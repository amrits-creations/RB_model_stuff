# RB_model_stuff

An implementation of the Rao & Ballard predictive coding model, applied to live
hand-tracking data. A predictive coding layer learns from a hand during a short
training phase, then drives an "avatar" hand via active inference.

The model equations are written out in [`docs/model_equations.md`](docs/model_equations.md).

## Usage

    pip install -r requirements.txt
    python hand_avatar.py

Wave your hand during training, then watch the avatar. Press `q` to quit.

## Reference

> Rao, R. P. N., & Ballard, D. H. (1999). *Predictive coding in the visual cortex:
> a functional interpretation of some extra-classical receptive-field effects.*
> Nature Neuroscience, 2(1), 79–87. https://doi.org/10.1038/4580

An independent re-implementation; equations are from the paper, the code is mine.
