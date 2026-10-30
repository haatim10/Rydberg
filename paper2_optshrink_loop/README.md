# Paper 2: in-loop OptShrink check

Read `PROTOCOL.md` for the frozen comparison and its limits. From the repository
root, use the existing `.venv-paper2` Python environment:

```sh
../../.venv-paper2/bin/python -m unittest paper2_optshrink_loop.test_estimator -v
../../.venv-paper2/bin/python -u -m paper2_optshrink_loop.run
../../.venv-paper2/bin/python -m paper2_optshrink_loop.analyze
```

Run the second command again only to resume an interrupted comparison. A process
lock prevents concurrent runs. Sources, configuration and checkpoints must still
match their frozen hashes. Complete case files are reused, not overwritten.

Results and live progress are in `results/paper2_optshrink_loop/`. The final
summary recalculates errors from saved estimates and checks selected in-loop
and final-filter outputs using an independent scalar-weight implementation.

All historical packages, training weights and manuscripts remain unchanged.
The classical comparator is an adaptation of OptShrink to Rician EM, not a
reproduction of LRHD. There is no new neural training in this check.
