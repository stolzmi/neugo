package nn

import "math/rand"

func NewRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
