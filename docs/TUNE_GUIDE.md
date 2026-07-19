# Hyperparameter Tuning Guide

The `tune` package provides fast, parallel hyperparameter search with automatic early stopping via the Asynchronous Successive Halving Algorithm (ASHA).

## Quick Start

```go
import "neugo/tune"

// 1. Define your search space
space := tune.NewSpace().
    LogFloat("lr", 1e-4, 0.5).
    Int("hidden", 4, 64).
    Choice("act", "relu", "tanh")

// 2. Define an objective function
objective := func(trial *tune.Trial) (float64, error) {
    lr := trial.Params.Float("lr")
    hidden := trial.Params.Int("hidden")
    activation := trial.Params.Choice("act")
    
    // Build and train model
    loss := trainModel(lr, hidden, activation)
    return loss, nil
}

// 3. Run the search
results, err := tune.Run(ctx, space, objective, tune.Config{
    Trials:  60,
    Workers: runtime.NumCPU(),
    ASHA: &tune.ASHAConfig{
        MinResource:     2,
        MaxResource:     32,
        ReductionFactor: 4,
    },
})

// 4. Inspect results
best := results.Best()
fmt.Printf("Best loss: %.4f\n", best.Value)
```

## API Reference

### Search Space

A `Space` defines which hyperparameters to search and their ranges.

```go
space := tune.NewSpace()
```

#### LogFloat (for learning rates and other exponential ranges)
```go
space.LogFloat("lr", 1e-4, 0.5)  // log-uniform: samples exponent uniformly in [log(1e-4), log(0.5)]
```
Prefer `LogFloat` for learning rates and other parameters that work best on a log scale.

#### Float (uniform)
```go
space.Float("momentum", 0.8, 0.99)  // uniform
```

#### Int (discrete)
```go
space.Int("hidden", 4, 64)  // uniform over integers [4, 64] inclusive
```

#### Choice (categorical)
```go
space.Choice("act", "relu", "tanh", "sigmoid")  // picks one
```

Space methods return `*Space` for chaining.

### Objective Function

Your objective function receives a `Trial` and returns `(score, error)`.

```go
type Trial struct {
    ID     int           // trial number (0 to Trials-1)
    Params Params        // parameter values for this trial
    Seed   int64         // deterministic seed derived from config.Seed + ID
}

type Objective func(t *Trial) (float64, error)
```

Lower scores are better by default (minimize); set `Config.Maximize = true` to maximize.

#### Extracting Hyperparameters

```go
func objective(trial *tune.Trial) (float64, error) {
    lr := trial.Params.Float("lr")           // panics if missing or wrong type
    hidden := trial.Params.Int("hidden")
    activation := trial.Params.Choice("act")
    
    // Train and evaluate
    loss := trainModel(lr, hidden, activation)
    return loss, nil
}
```

#### Per-Epoch Reporting (ASHA)

If you enable ASHA pruning, report intermediate metrics to allow early stopping:

```go
for epoch := 1; epoch <= maxEpochs; epoch++ {
    loss := trainOneEpoch()
    
    // Report validation loss at this resource level (epoch count)
    trial.Report(epoch, loss)
    
    // Check if ASHA wants to prune this trial
    if trial.ShouldPrune() {
        return loss, nil  // stop training, return best loss so far
    }
}
```

- `Trial.Report(resource, value)` — Records an intermediate metric. `resource` is typically epoch count; `value` is your metric (e.g., validation loss).
- `Trial.ShouldPrune()` — Returns `true` if ASHA has marked this trial for early stopping based on its performance relative to other trials at the same resource level.

When ASHA is not configured (nil), `Report` is a no-op and `ShouldPrune()` always returns false.

### Running the Search

```go
results, err := tune.Run(ctx, space, objective, cfg)
```

#### Config

```go
type Config struct {
    Trials   int         // number of trials to run (default none)
    Workers  int         // max parallel trials (<=0 → runtime.NumCPU())
    Seed     int64       // random seed for parameter sampling
    Maximize bool        // true to maximize, false to minimize
    ASHA     *ASHAConfig // pruning config; nil = no pruning
}
```

#### ASHA Config (Early Stopping)

```go
type ASHAConfig struct {
    MinResource     int  // Starting resource level (e.g., epochs); must be > 0
    MaxResource     int  // Maximum resource level; must be >= MinResource
    ReductionFactor int  // Reduction factor η (default 3 if <= 1)
}
```

**Example: ASHA for 60 trials, max 32 epochs per trial**

```go
ASHA: &tune.ASHAConfig{
    MinResource:     2,
    MaxResource:     32,
    ReductionFactor: 4,
}
```

This creates 3 "rungs" of successive halving:
- **Rung 0** (2 epochs): All 60 trials run 2 epochs; bottom 75% (45 trials) pruned.
- **Rung 1** (8 epochs): Remaining 15 trials run 6 more epochs (8 total); bottom 75% (11 trials) pruned.
- **Rung 2** (32 epochs): Remaining 4 trials run to 32 epochs (24 more).

At each rung, the bottom `1 - 1/ReductionFactor` fraction is eliminated, and survivors run longer.

See the [ASHA paper](https://arxiv.org/abs/1810.05934) for details.

### Results

```go
results, err := tune.Run(ctx, space, objective, cfg)
if err != nil {
    // Handle error (may include context cancellation)
}

best := results.Best()
fmt.Printf("Best loss: %.4f\n", best.Value)

top := &tune.Results{Trials: results.Top(10)}
fmt.Println(top.String())  // Pretty-printed table of top 10
```

#### TrialResult

```go
type TrialResult struct {
    ID       int            // trial ID
    Params   Params         // sampled hyperparameters
    Value    float64        // returned score
    Err      error          // if any error occurred
    Pruned   bool           // true if ASHA pruned this trial
    Duration time.Duration  // wall time spent in objective
}
```

- **Best()** — Returns the trial result with the best score (highest if Maximize, lowest if minimize). Always ranked first.
- **Top(k)** — Returns the top k trials (sorted best-first).
- **String()** — Returns a formatted table of all trials sorted by rank.

Results are pre-sorted best-first; successful non-pruned trials come first, then errored/pruned trials.

## Determinism

Hyperparameter sampling is deterministic: **same Seed → same parameter sets regardless of scheduling or worker count.**

Each trial is seeded as `config.Seed + trial_ID`. The space's parameter sampling uses `math/rand` and therefore is fully deterministic given a seed:

```go
// These two runs will sample identical parameters:
cfg1 := tune.Config{Trials: 60, Seed: 42, Workers: 1}
cfg2 := tune.Config{Trials: 60, Seed: 42, Workers: runtime.NumCPU()}
// Same 60 parameter sets, just run in different order.
```

**Note:** Trial execution order may differ (parallelism), but parameter values are the same. Model training randomness is controlled by `trial.Seed`.

## Efficiency: One Process, All Cores, No Cluster

The `tune` package is designed for single-machine tuning with **minimal overhead**:

- **One process** — No subprocess or IPC overhead; objective runs in goroutines.
- **All cores** — Workers default to `runtime.NumCPU()` for automatic parallelism.
- **No cluster** — Results are sorted in-memory; no network round-trips or external database.

Typical wall time for 60 trials (32 epochs max, ASHA enabled):
- **Without ASHA:** ~120 worker-seconds (60 trials × 32 epochs) / Workers
- **With ASHA:** ~30–50 worker-seconds (aggressive early stopping)

On a 4-core machine with 4 workers, ASHA-enabled tuning of 60 trials typically completes in **30–60 seconds**, not hours.

## Example: Wine Quality Tuning

See `examples/tune_wine/main.go` for a complete example:

1. **Load data** — CSV with normalization.
2. **Create space** — LogFloat for learning rate, Int for hidden size, Choice for activation.
3. **Define objective** — Build Sequential model, train epoch-by-epoch, report validation loss.
4. **Run search** — 60 trials, ASHA pruning with 3 rungs.
5. **Display results** — Best parameters and ASHA efficiency stats.

```bash
cd examples/tune_wine
go run .
```

Output:
```
Best validation loss: 0.2139
  Learning rate: 0.275105
  Hidden size: 46
  Activation: tanh

=== ASHA Efficiency ===
Wall time: 3.6s
Total epochs executed: 534 (vs. 1920 possible)
ASHA pruning efficiency: 25.3% (lower = more aggressive)
```

## Tips

1. **Use LogFloat for continuous hyperparameters that span orders of magnitude** (learning rate, weight decay, dropout).
2. **Report at regular intervals** in the objective (e.g., every epoch) to let ASHA prune inefficient trials early.
3. **Set MinResource to a small value** (e.g., 2–5 epochs) so ASHA can quickly identify bad configurations.
4. **Tune learning rate first**, then other hyperparameters — learning rate often has the largest impact.
5. **Determinism is powerful:** Use the same Seed for reproducible searches; vary Seed across runs for exploration.

## Common Errors

**"Params.Float/Int/Choice: parameter not found"** — Accessing a parameter by the wrong name. Check your space definition.

**"LogFloat: min must be > 0"** — LogFloat requires positive bounds (for log scale). Use `Float` for negative ranges.

**"ShouldPrune always false"** — ASHA is not configured. Set `Config.ASHA` if you want early stopping.

**"Trial execution is slower than expected"** — Reduce Workers or check if your objective is I/O-bound. Tune package parallelizes via goroutines, not processes, so CPU contention may reduce per-worker speed.
