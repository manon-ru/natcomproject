# Proposal Patch: Path-Encoded GA

## Edit 1: Replace the GA parameters paragraph in Section 4.1

**OLD** (around lines 71-73 of proposal.txt):
> **GA parameters (Shrestha et al. [10.5120/ijca2024923402]).**
> Population 50, roulette wheel selection, two-point crossover with split 0.5,
> flip-bit mutation at rate 0.3. Chromosome encoding is determined by our
> grid-based implementation and differs from Shrestha's 36-bit waypoint encoding.

**NEW**:
> **GA parameters.**
> Population 50, roulette wheel selection, two-point crossover at common cells
> with rate 0.5, per-chromosome mutation at rate 0.3 (truncate-and-regrow).
> Chromosome encoding is a variable-length sequence of cells visited from start,
> following Lamini et al. [LAMINI] and Tu & Yang [TU-YANG]. This differs from
> Shrestha et al. [10.5120/ijca2024923402], whose 36-bit waypoint encoding
> suits continuous 2D environments with random waypoints scattered through
> free space; in our grid-based formulation with cell-by-cell maze navigation,
> no continuous space exists between cells and the path itself is the
> chromosome's natural representation. The proposal-spec operators carry over
> directly: roulette wheel selection (identical to Shrestha), two-point
> crossover (adapted to splice at common cells between parents — a standard
> adaptation for variable-length path chromosomes), and mutation at rate 0.3
> (adapted from per-bit flip to per-chromosome truncate-regrow, since per-cell
> mutation has no well-defined analogue when adjacency must be preserved).

## Edit 2: Add to the bibliography

```
[LAMINI] C. Lamini, S. Benhlima, and A. Elbekri. "Genetic algorithm based
approach for autonomous mobile robot path planning." Procedia Computer
Science, 127:180-189, 2018.

[TU-YANG] J. Tu and S. Yang. "Genetic algorithm based path planning for a
mobile robot." In Proceedings of IEEE International Conference on Robotics
and Automation, vol. 1, pp. 1221-1226, 2003.
```

## Edit 3: Add to Section 6 (Limitations) — methodological note

> **Implementation note on GA encoding.** We initially implemented a
> direction-string encoding (chromosome = fixed-length sequence of U/D/L/R
> moves of length 2(W+H)) as a literal interpretation of "flip-bit mutation
> at rate 0.3." This produced 0% success across all 9 experimental cells
> (3 mazes × 3 population sizes; 100 trials each). We verified that this
> failure was not parameter-tuning-induced via a 4×2 mutation-rate × chromosome-
> length sweep (40 trials; 0% success across all cells) and not iteration-bound
> via an iteration-budget sweep up to 50,000 generations × 50 population =
> 2.5M chromosome evaluations per trial (16 trials; 0% success across all
> budgets; mean best-found distance to goal plateaued at ~23 cells regardless
> of compute). The failure mode is structural: direction-string crossover does
> not preserve "valid path" as an invariant property of recombination, so
> splicing two random direction strings produces another random direction
> string. Path-encoded GAs are robust to this issue because two-point crossover
> at common cells always produces a structurally valid hybrid path. We retain
> this as evidence that operator descriptions from waypoint-encoded GAs
> (Shrestha et al.) do not transfer to direction-encoded grid GAs without
> structural-preservation considerations.