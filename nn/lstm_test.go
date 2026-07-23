package nn

import "testing"

func TestLSTMOutputShape(t *testing.T) {
	l := LSTM(NewRNG(1), 3, 5, nil)
	out, err := l.OutputShape([]int{2, 4, 3})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 4, 5}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestLSTMRejectsWrongFeatureCount(t *testing.T) {
	l := LSTM(NewRNG(1), 3, 5, nil)
	if _, err := l.OutputShape([]int{2, 4, 7}); err == nil {
		t.Fatal("expected error for feature mismatch, got nil")
	}
}

func TestLSTMGateWeightShapes(t *testing.T) {
	l := LSTM(NewRNG(1), 3, 5, nil)
	if len(l.Wx.Value.Data) != 3*4*5 {
		t.Errorf("Wx has %d values, want %d (features*4*hidden)", len(l.Wx.Value.Data), 3*4*5)
	}
	if len(l.Wh.Value.Data) != 5*4*5 {
		t.Errorf("Wh has %d values, want %d (hidden*4*hidden)", len(l.Wh.Value.Data), 5*4*5)
	}
	if len(l.B.Value.Data) != 4*5 {
		t.Errorf("B has %d values, want %d (4*hidden)", len(l.B.Value.Data), 4*5)
	}
}

func TestLSTMGradients(t *testing.T) {
	rng := NewRNG(11)
	l := LSTM(rng, 3, 4, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.09 - 0.4
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, l, ctx, x)
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return l.Backward(ctx, grad) }
	for _, p := range l.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// FuzzLSTMGradient fuzzes over shape dimensions (batch, seqLen, features,
// hidden — all kept small since BPTT cost grows with seqLen), looking for
// any combination where the analytic and numeric gradients disagree.
func FuzzLSTMGradient(f *testing.F) {
	f.Add(2, 3, 3, 4)
	f.Add(1, 1, 1, 1)
	f.Fuzz(func(t *testing.T, batch, seqLen, features, hidden int) {
		batch = clampDim(batch, 1, 3)
		seqLen = clampDim(seqLen, 1, 4)
		features = clampDim(features, 1, 4)
		hidden = clampDim(hidden, 1, 4)

		rng := NewRNG(1)
		l := LSTM(rng, features, hidden, XavierInit())
		x := NewTensor([]int{batch, seqLen, features})
		for i := range x.Data {
			x.Data[i] = float32((i*7+seqLen+hidden)%11)*0.05 - 0.25
		}
		ctx := &Context{Mode: Train}
		checkInputGradient(t, l, ctx, x)
		forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
		backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
		for _, p := range l.Params() {
			checkParamGradient(t, forward, backward, p)
		}
	})
}

func TestLSTMSingleTimestepMatchesGateEquations(t *testing.T) {
	// batch=1, seqLen=1, features=1, hidden=1 — hand-checkable.
	l := LSTM(NewRNG(1), 1, 1, ZerosInit())
	// Gate order in Wx/B is (i, f, g, o); with all zero weights and a
	// nonzero bias for each gate we can predict the output exactly.
	l.B.Value.Data[0] = 0 // input gate bias -> sigmoid(0) = 0.5
	l.B.Value.Data[1] = 0 // forget gate bias -> sigmoid(0) = 0.5 (irrelevant, c_prev=0)
	l.B.Value.Data[2] = 0 // cell candidate bias -> tanh(0) = 0
	l.B.Value.Data[3] = 0 // output gate bias -> sigmoid(0) = 0.5
	x, _ := NewTensorFromData([]float32{0}, []int{1, 1, 1})
	out, err := l.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// c_0 = f*c_prev + i*g = 0.5*0 + 0.5*0 = 0; h_0 = o*tanh(c_0) = 0.5*0 = 0
	if out.Data[0] != 0 {
		t.Fatalf("h_0 = %v, want 0", out.Data[0])
	}
}
