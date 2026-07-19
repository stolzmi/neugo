package nn

import (
	"bytes"
	"math"
	"testing"
)

func testModel(t *testing.T) *SequentialModel {
	t.Helper()
	rng := NewRNG(7)
	m, err := Sequential([]int{1, 3},
		Linear(rng, 3, 4, nil), ReLU(),
		Linear(rng, 4, 2, nil), Sigmoid(),
	)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func forward1(t *testing.T, m *SequentialModel, in []float32) []float32 {
	t.Helper()
	x, err := NewTensorFromData(in, []int{1, len(in)})
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatal(err)
	}
	return out.Data
}

func TestMarshalUnmarshalRoundTrip(t *testing.T) {
	m := testModel(t)
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{0.5, -1, 2}
	a, b := forward1(t, m, in), forward1(t, m2, in)
	if !bytes.Equal(float32Bytes(a), float32Bytes(b)) {
		t.Fatalf("round-trip changed outputs: %v vs %v", a, b)
	}
}

func TestCloneIsDeepAndIndependent(t *testing.T) {
	m := testModel(t)
	c, err := Clone(m)
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{1, 1, 1}
	before := forward1(t, m, in)
	// Mutate every clone param; original outputs must not move.
	for _, p := range c.Params() {
		for i := range p.Value.Data {
			p.Value.Data[i] += 100
		}
	}
	after := forward1(t, m, in)
	if !bytes.Equal(float32Bytes(before), float32Bytes(after)) {
		t.Fatalf("mutating clone changed original: %v vs %v", before, after)
	}
}

func TestUnmarshalRejectsNonSequentialRoot(t *testing.T) {
	if _, err := Unmarshal([]byte(`{"type":"linear"}`)); err == nil {
		t.Fatal("want error for non-sequential root")
	}
}

func float32Bytes(v []float32) []byte {
	b := make([]byte, 0, len(v)*4)
	for _, x := range v {
		u := math.Float32bits(x)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	return b
}
