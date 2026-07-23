package nn

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadDenseModelRoundTrip(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 3},
		Linear(rng, 3, 4, XavierInit()),
		ReLU(),
		BatchNorm(4),
		Dropout(0.2),
		Linear(rng, 4, 2, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// Run one Train-mode forward so BatchNorm accumulates non-trivial running stats.
	ctx := &Context{Mode: Train, RNG: NewRNG(2)}
	x := NewTensor([]int{2, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	if _, err := model.Forward(ctx, x); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	infCtx := &Context{Mode: Inference}
	want, err := model.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward original: %v", err)
	}
	got, err := loaded.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v (loaded model diverges from original)", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConvModelRoundTrip(t *testing.T) {
	rng := NewRNG(3)
	model, err := Sequential([]int{1, 6, 6, 1},
		Conv2D(rng, 1, 2, 3, HeInit()),
		ReLU(),
		MaxPool2D(2, 2),
		Flatten(),
		Linear(rng, 0, 1, XavierInit()),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 6, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "cnn.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConvStridePreserved(t *testing.T) {
	rng := NewRNG(9)
	model, err := Sequential([]int{1, 7, 7, 1},
		Conv2DStrided(rng, 1, 2, 3, 2, 1, HeInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 7, 7, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(want.Shape) != 4 || want.Shape[1] != 4 || want.Shape[2] != 4 {
		t.Fatalf("expected stride-2 output spatial dims 4x4, got %v", want.Shape)
	}

	path := filepath.Join(t.TempDir(), "strided.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Fatalf("loaded output shape %v, want %v — stride not preserved through Save/Load", got.Shape, want.Shape)
		}
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadResidualBlockRoundTrip(t *testing.T) {
	rng := NewRNG(10)
	model, err := Sequential([]int{1, 6, 6, 2},
		Residual(
			Conv2DStrided(rng, 2, 4, 1, 2, 0, HeInit()),
			Conv2DStrided(rng, 2, 4, 3, 2, 1, HeInit()),
			ReLU(),
			Conv2DSame(rng, 4, 4, 3, HeInit()),
		),
		Flatten(),
		Linear(rng, 0, 2, XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 6, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "residual.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadGroupNormRoundTrip(t *testing.T) {
	g := GroupNorm(2, 4)
	x := NewTensor([]int{3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7) * 0.15
	}
	ctx := &Context{Mode: Train}
	want, err := g.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{3, 4}, g)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "groupnorm.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadLayerNormRoundTrip(t *testing.T) {
	l := LayerNorm(4)
	x := NewTensor([]int{2, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7) * 0.15
	}
	ctx := &Context{Mode: Train}
	want, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{2, 3, 4}, l)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "layernorm.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadEmbeddingRoundTrip(t *testing.T) {
	rng := NewRNG(11)
	e := Embedding(rng, 6, 3, NormalInit(0, 0.5))
	x, err := NewTensorFromData([]float32{5, 0, 2}, []int{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &Context{Mode: Inference}
	want, err := e.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{1, 3}, e)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "embedding.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConv1DRoundTrip(t *testing.T) {
	rng := NewRNG(12)
	model, err := Sequential([]int{1, 6, 1},
		Conv1DStrided(rng, 1, 2, 3, 2, 1, HeInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "conv1d.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Fatalf("loaded output shape %v, want %v", got.Shape, want.Shape)
		}
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConvTranspose2DRoundTrip(t *testing.T) {
	rng := NewRNG(13)
	model, err := Sequential([]int{1, 3, 3, 2},
		ConvTranspose2D(rng, 2, 3, 3, 2, 1, HeInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "convtranspose.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Fatalf("loaded output shape %v, want %v", got.Shape, want.Shape)
		}
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadFrozenRoundTrip(t *testing.T) {
	rng := NewRNG(14)
	model, err := Sequential([]int{1, 3},
		Frozen(Linear(rng, 3, 2, XavierInit())),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "frozen.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := len(loaded.Params()); got != 0 {
		t.Fatalf("loaded model has %d params, want 0 (frozen layer's params must stay excluded after Load)", got)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadWithMetadataRoundTrip(t *testing.T) {
	rng := NewRNG(15)
	model, err := Sequential([]int{1, 3}, Linear(rng, 3, 2, XavierInit()))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	meta := Metadata{
		InputShape: []int{1, 3},
		ClassNames: []string{"cat", "dog"},
		Normalization: &NormalizationStats{
			Mean: []float32{0.1, 0.2, 0.3},
			Std:  []float32{0.9, 1.1, 1.0},
		},
		Manifest: map[string]string{
			"go_version": "go1.22.0",
			"git_commit": "abc1234",
			"seed":       "42",
		},
	}
	path := filepath.Join(t.TempDir(), "with_meta.json")
	if err := SaveWithMetadata(model, path, meta); err != nil {
		t.Fatalf("SaveWithMetadata: %v", err)
	}
	loaded, loadedMeta, err := LoadWithMetadata(path)
	if err != nil {
		t.Fatalf("LoadWithMetadata: %v", err)
	}

	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}

	if len(loadedMeta.ClassNames) != 2 || loadedMeta.ClassNames[0] != "cat" || loadedMeta.ClassNames[1] != "dog" {
		t.Errorf("ClassNames = %v, want [cat dog]", loadedMeta.ClassNames)
	}
	if len(loadedMeta.InputShape) != 2 || loadedMeta.InputShape[0] != 1 || loadedMeta.InputShape[1] != 3 {
		t.Errorf("InputShape = %v, want [1 3]", loadedMeta.InputShape)
	}
	if loadedMeta.Normalization == nil {
		t.Fatal("Normalization is nil, want the saved stats")
	}
	for i, want := range []float32{0.1, 0.2, 0.3} {
		if loadedMeta.Normalization.Mean[i] != want {
			t.Errorf("Normalization.Mean[%d] = %v, want %v", i, loadedMeta.Normalization.Mean[i], want)
		}
	}
	for k, want := range map[string]string{"go_version": "go1.22.0", "git_commit": "abc1234", "seed": "42"} {
		if got := loadedMeta.Manifest[k]; got != want {
			t.Errorf("Manifest[%q] = %q, want %q", k, got, want)
		}
	}
}

func TestMetadataManifestOmittedWhenNil(t *testing.T) {
	rng := NewRNG(17)
	model, err := Sequential([]int{1, 2}, Linear(rng, 2, 1, XavierInit()))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "no_manifest.json")
	if err := SaveWithMetadata(model, path, Metadata{ClassNames: []string{"a"}}); err != nil {
		t.Fatalf("SaveWithMetadata: %v", err)
	}
	_, loadedMeta, err := LoadWithMetadata(path)
	if err != nil {
		t.Fatalf("LoadWithMetadata: %v", err)
	}
	if loadedMeta.Manifest != nil {
		t.Errorf("Manifest = %v, want nil when never set", loadedMeta.Manifest)
	}
}

func TestPlainLoadRejectsMetadataFileAndViceVersa(t *testing.T) {
	rng := NewRNG(16)
	model, err := Sequential([]int{1, 2}, Linear(rng, 2, 1, XavierInit()))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	plainPath := filepath.Join(t.TempDir(), "plain.json")
	if err := Save(model, plainPath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, _, err := LoadWithMetadata(plainPath); err == nil {
		t.Error("LoadWithMetadata on a plain Save file returned nil error, want an error (formats are not cross-compatible)")
	}

	metaPath := filepath.Join(t.TempDir(), "meta.json")
	if err := SaveWithMetadata(model, metaPath, Metadata{}); err != nil {
		t.Fatalf("SaveWithMetadata: %v", err)
	}
	if _, err := Load(metaPath); err == nil {
		t.Error("Load on a SaveWithMetadata file returned nil error, want an error (formats are not cross-compatible)")
	}
}

// TestSaveLoadCoversRemainingModuleTypes exercises the module types not
// touched by the other round-trip tests: Tanh, LeakyReLU with a non-zero
// alpha, GELU, AvgPool2D, and a nested Sequential-within-Sequential model.
func TestSaveLoadCoversRemainingModuleTypes(t *testing.T) {
	rng := NewRNG(4)
	inner, err := Sequential([]int{1, 4}, Tanh(), LeakyReLU(0.15), GELU())
	if err != nil {
		t.Fatalf("Sequential (inner): %v", err)
	}
	model, err := Sequential([]int{1, 4, 4, 1},
		AvgPool2D(2, 2),
		Flatten(),
		inner,
		Linear(rng, 4, 2, XavierInit()),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential (outer): %v", err)
	}

	x := NewTensor([]int{1, 4, 4, 1})
	for i := range x.Data {
		x.Data[i] = float32(i)*0.1 - 0.5 // mix of negative/positive values to exercise LeakyReLU's alpha branch
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "remaining_types.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadNewActivationsRoundTrip(t *testing.T) {
	model, err := Sequential([]int{1, 4}, ELU(0.7), SELU(), SiLU(), Softplus(), Mish(), Hardswish())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 4})
	for i := range x.Data {
		x.Data[i] = float32(i)*0.3 - 0.5
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "new_activations.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}

	// The ELU alpha must survive the round trip (not just default to 1).
	loadedELU, ok := loaded.Modules()[0].(*ActivationModule)
	if !ok || loadedELU.Alpha() != 0.7 {
		t.Errorf("loaded ELU alpha = %v, want 0.7", loadedELU.Alpha())
	}
}

func TestSaveLoadPReLURoundTrip(t *testing.T) {
	p := PReLU(4)
	for i := range p.Alpha.Value.Data {
		p.Alpha.Value.Data[i] = float32(i) * 0.1
	}
	x := NewTensor([]int{3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.2 - 0.6
	}
	ctx := &Context{Mode: Inference}
	want, err := p.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{3, 4}, p)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "prelu.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadRMSNormRoundTrip(t *testing.T) {
	r := RMSNorm(4)
	for i := range r.Gamma.Value.Data {
		r.Gamma.Value.Data[i] = 1 + float32(i)*0.2
	}
	x := NewTensor([]int{3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.15 - 0.4
	}
	ctx := &Context{Mode: Inference}
	want, err := r.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{3, 4}, r)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "rmsnorm.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadInstanceNormRoundTrip(t *testing.T) {
	in := InstanceNorm(3)
	x := NewTensor([]int{2, 2, 2, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.11 - 0.3
	}
	ctx := &Context{Mode: Inference}
	want, err := in.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{2, 2, 2, 3}, in)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "instancenorm.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadPoolingAdditionsRoundTrip(t *testing.T) {
	model, err := Sequential([]int{1, 4, 4, 2}, AdaptiveAvgPool2D(2, 2), GlobalMaxPool2D())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.05
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "pooling_additions.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadRNNRoundTrip(t *testing.T) {
	rng := NewRNG(2)
	r := RNN(rng, 3, 4, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%11)*0.08 - 0.4
	}
	ctx := &Context{Mode: Inference}
	want, err := r.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{2, 3, 3}, r)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "rnn.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadLSTMRoundTrip(t *testing.T) {
	rng := NewRNG(3)
	l := LSTM(rng, 3, 4, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.09 - 0.4
	}
	ctx := &Context{Mode: Inference}
	want, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{2, 3, 3}, l)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "lstm.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadGRURoundTrip(t *testing.T) {
	rng := NewRNG(4)
	g := GRU(rng, 3, 4, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%13)*0.07 - 0.35
	}
	ctx := &Context{Mode: Inference}
	want, err := g.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{2, 3, 3}, g)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "gru.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadRNNThenLastTimestepRoundTrip(t *testing.T) {
	rng := NewRNG(6)
	model, err := Sequential([]int{2, 3, 3}, LSTM(rng, 3, 4, XavierInit()), LastTimestep())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%10) * 0.1
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "rnn_last_timestep.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadRotaryMultiHeadAttentionRoundTrip(t *testing.T) {
	rng := NewRNG(9)
	m := RotaryMultiHeadAttention(rng, 4, 2, true, XavierInit())
	x := NewTensor([]int{1, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.1 - 0.3
	}
	ctx := &Context{Mode: Inference}
	want, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	model, err := Sequential([]int{1, 3, 4}, m)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	path := filepath.Join(t.TempDir(), "rope.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

// TestLoadRejectsTruncatedParamData is the Finding-1 regression test: a
// param whose data array doesn't match the shape the module actually
// expects must fail Load with an error, never silently copy a short slice.
func TestLoadRejectsTruncatedParamData(t *testing.T) {
	// A "linear" module configured for 2x2 W (4 values) but the saved W
	// data only has 3 — truncated/corrupt file.
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "linear",
				"config": {"in_features": 2, "out_features": 2},
				"params": {
					"W": {"shape": [2, 2], "data": [1, 2, 3]},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "truncated.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a truncated param array, got nil")
	}
}

func TestLoadMissingFileReturnsError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json")); err == nil {
		t.Fatal("expected error loading a missing file, got nil")
	}
}

func TestLoadRejectsMalformedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("not json"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading malformed JSON, got nil")
	}
}

// TestLoadRejectsNonPositiveLinearInFeatures is a regression test: a
// "linear" module with in_features<=0 (e.g. omitted or explicitly 0) used
// to reach Linear's lazy-build path, which leaves W/B nil, and then
// decodeModule's unconditional access to l.W.Value.Data would panic with a
// nil-pointer dereference. Load must reject it with a clean error instead.
func TestLoadRejectsNonPositiveLinearInFeatures(t *testing.T) {
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "linear",
				"config": {"in_features": 0, "out_features": 2},
				"params": {
					"W": {"shape": [0, 2], "data": []},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "zero_in_features.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a linear module with in_features=0, got nil")
	}
}

// TestLoadRejectsNegativeConv2DChannels is a regression test: a "conv2d"
// module with a negative in_channels/out_channels/kernel_size used to reach
// newConv2D's unconditional tensor allocation, which panics with
// "makeslice: len out of range" for negative sizes. Load must reject it
// with a clean error instead.
func TestLoadRejectsNegativeConv2DChannels(t *testing.T) {
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "conv2d",
				"config": {"in_channels": -1, "out_channels": 2, "kernel_size": 3, "padding": 0},
				"params": {
					"W": {"shape": [2, -1, 3, 3], "data": []},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "negative_channels.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a conv2d module with in_channels=-1, got nil")
	}
}

// TestSaveNilModelReturnsError is a regression test: Save(nil, path) used to
// panic inside encodeModule's *SequentialModel case when it dereferenced
// v.modules on a nil model pointer. Save must reject it with a clean error
// instead.
func TestSaveNilModelReturnsError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nil_model.json")
	if err := Save(nil, path); err == nil {
		t.Fatal("expected error saving a nil model, got nil")
	}
}

func TestSaveLoadPositionalEmbeddingRoundTrip(t *testing.T) {
	rng := NewRNG(18)
	p := PositionalEmbedding(rng, 5, 3, NormalInit(0, 0.1))
	model, err := Sequential([]int{1, 3, 3}, p)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%4) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "positional.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadTransformerBlockRoundTrip(t *testing.T) {
	rng := NewRNG(19)
	block, err := TransformerBlock(rng, 4, 2, 8, true, XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock: %v", err)
	}
	model, err := Sequential([]int{1, 3, 4}, block)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "transformerblock.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadMultiHeadAttentionRoundTrip(t *testing.T) {
	rng := NewRNG(17)
	m := MultiHeadAttention(rng, 4, 2, true, XavierInit())
	model, err := Sequential([]int{1, 3, 4}, m)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "attention.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}
