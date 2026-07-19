package nn

import (
	"math"
	"testing"
)

func TestZerosInit(t *testing.T) {
	rng := NewRNG(1)
	tn := ZerosInit()(rng, []int{4, 4})
	for _, v := range tn.Data {
		if v != 0 {
			t.Fatalf("ZerosInit produced non-zero value %v", v)
		}
	}
}

func TestUniformInitBounds(t *testing.T) {
	rng := NewRNG(1)
	tn := UniformInit(-0.5, 0.5)(rng, []int{1000})
	for _, v := range tn.Data {
		if v < -0.5 || v >= 0.5 {
			t.Fatalf("UniformInit(-0.5,0.5) produced out-of-range value %v", v)
		}
	}
}

func TestHeInitMomentBounds(t *testing.T) {
	rng := NewRNG(1)
	fanIn := 256
	tn := HeInit()(rng, []int{fanIn, 64})
	var sumSq float64
	for _, v := range tn.Data {
		sumSq += float64(v) * float64(v)
	}
	variance := sumSq / float64(len(tn.Data))
	wantVariance := 2.0 / float64(fanIn)
	if math.Abs(variance-wantVariance)/wantVariance > 0.25 {
		t.Fatalf("He-initialized variance = %v, want close to %v", variance, wantVariance)
	}
}

func TestXavierInitConv4DShape(t *testing.T) {
	rng := NewRNG(1)
	// Conv weight convention: [outC, inC, kh, kw]
	tn := XavierInit()(rng, []int{8, 3, 3, 3})
	if tn.Size() != 8*3*3*3 {
		t.Fatalf("Size() = %d, want %d", tn.Size(), 8*3*3*3)
	}
}
