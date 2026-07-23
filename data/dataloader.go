// data/dataloader.go
package data

import "math/rand"

// DataLoader yields shuffled batches of *indices* into a fixed-size
// dataset — the streaming counterpart to this package's usual "load
// everything into memory upfront" model (see doc.go). It stays agnostic
// to whatever slice type actually backs the dataset ([]*Image,
// [][]float32, a lazy on-disk source, ...) by never touching the samples
// themselves: callers pull batches of indices via Next and use them to
// index into (or decode from) their own data, then convert that batch
// into whatever tensor type they need (train.Trainer.FitStream is built
// for the common case of converting into *nn.Tensor, kept out of this
// package per the data<->nn boundary).
type DataLoader struct {
	n         int
	batchSize int
	rng       *rand.Rand
	shuffle   bool

	indices []int
	pos     int
}

// NewDataLoader creates a loader over n samples, yielding indices
// batchSize at a time (the last batch of an epoch may be smaller). rng is
// used only when shuffle is true.
func NewDataLoader(n, batchSize int, rng *rand.Rand, shuffle bool) *DataLoader {
	return &DataLoader{n: n, batchSize: batchSize, rng: rng, shuffle: shuffle}
}

// Reset starts a new epoch: re-shuffles (if enabled) and rewinds to the
// first batch. Next calls Reset automatically before the very first
// batch, so an explicit Reset is only needed to start a *subsequent*
// epoch.
func (d *DataLoader) Reset() {
	d.indices = make([]int, d.n)
	for i := range d.indices {
		d.indices[i] = i
	}
	if d.shuffle {
		d.rng.Shuffle(d.n, func(i, j int) { d.indices[i], d.indices[j] = d.indices[j], d.indices[i] })
	}
	d.pos = 0
}

// Next returns the next batch of indices, or ok=false once the current
// epoch is exhausted — call Reset to start another epoch.
func (d *DataLoader) Next() (batch []int, ok bool) {
	if d.indices == nil {
		d.Reset()
	}
	if d.pos >= d.n {
		return nil, false
	}
	end := d.pos + d.batchSize
	if end > d.n {
		end = d.n
	}
	batch = d.indices[d.pos:end]
	d.pos = end
	return batch, true
}

// NumBatches returns how many batches one epoch contains (the last one
// possibly smaller than the configured batch size).
func (d *DataLoader) NumBatches() int {
	if d.batchSize <= 0 {
		return 0
	}
	return (d.n + d.batchSize - 1) / d.batchSize
}
