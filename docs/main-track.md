# Riding vLLM main: the weekly rebase track

Porting to a release tag is expensive because everything arrives at once: months of upstream change meet the
whole patch series in one sitting, and the retirements, the moved hunks and the renamed enums all land on the
same afternoon. The 0.27.1 to 0.28.0 move took six weeks. The main track pays that cost in weekly pieces, so
that when the next tag arrives the port is a rebase that has already been done. It is a smoke, not a product:
production stays on the tag branches, and nothing is decided on the track.

## What it is

- **A branch on the vLLM fork**, `qwen38/main-track`, which is the patch series (one commit per patch, the
  same commits the tag branch is exported from) rebased onto a recent commit of `vllm-project/vllm` `main`.
- **The base commit is the nightly wheel's.** `pip download vllm --pre --index-url
  https://wheels.vllm.ai/nightly/cu130` returns a version such as `0.29.1rc1.dev5+ge52be1a62`; the suffix is the
  main commit the wheel was built from, and the fork tree must sit on exactly that commit, because the wheel is
  what supplies the compiled libraries.
- **An image for it**, built the way the release image is: the nightly wheel with its own dependency set
  (transformers and tokenizers pinned to the release image's, so a difference is vLLM's alone), then the fork
  tree installed over it. `verify.sh --install` is recorded, not gating, because retired patches are absent by
  design.

## The cadence

Weekly, or sooner when an upstream PR the ledger names merges.

1. Read the new base off the nightly index, and rebase: `git rebase --onto <new base> <old base>
   qwen38/main-track`.
2. Every stop gets its cause before it gets a resolution: `git log <old base>..<new base> -- <file>` names the
   upstream commit that moved the context. The ledger line for the stop says which upstream PR made the hunk
   redundant, if one did; a hunk that upstream has absorbed is retired, not re-cut.
3. Two numbers are recorded every time: wall clock, and the number of stops. The first rebase of the 0.29 series
   (v0.29.0 to e52be1a62, 601 upstream commits) took 6 minutes and stopped 8 times; 2 topics retired outright and
   3 more became retirement candidates pending the smoke.
4. Count commits before and after. A conflict resolved by taking upstream's side can leave a commit empty, and
   `git rebase` drops it silently; the count is the only thing that notices. Confirm the branch ref points at the
   rebased work before pushing anything.
5. Build the image and run the smoke: `bench/acceptance/acceptance.sh` with the fast profile and the quality
   battery, the int4 profile for the block-promotion lines and the mq3d oracle, and the fixed-prompt acceptance
   counters. The smoke answers two questions: does the series still boot and hold quality on main, and which
   retirement candidates can go.

## What comes out of it

- **Retirements, early.** Upstream absorbing one of our fixes shows up as a rebase stop weeks before a tag
  carries it, with the upstream commit named.
- **The next port, mostly paid.** When a tag arrives, the port is the track's last rebase re-targeted at the tag,
  plus whatever the tag branch carries that the track did not.
- **A radar.** A CI job can attempt the rebase onto `upstream/main` on a schedule, skipping and recording each
  conflicting topic; a red run is the list of topics the next real rebase will stop at, a week before anyone sits
  down to do it.

## The rule

Nothing is decided on the track. A retirement, a fix or a pin move happens on the tag branch under the two-box
bar (both boxes, same session controls, the acceptance table). The track only says earlier what that work will
contain.
