package nn

import "testing"

func TestNewRNGDeterministic(t *testing.T) {
	a := NewRNG(42)
	b := NewRNG(42)
	for i := 0; i < 5; i++ {
		if a.Float32() != b.Float32() {
			t.Fatal("NewRNG(42) produced different sequences across two instances")
		}
	}
}
