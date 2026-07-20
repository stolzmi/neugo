// nn/transformer.go
package nn

import "math/rand"

// TransformerBlock returns a standard post-norm Transformer encoder
// block: self-attention with a residual connection and LayerNorm,
// followed by a two-layer feed-forward sub-layer with its own residual
// connection and LayerNorm. Since *SequentialModel already implements
// Module, the result nests directly inside an outer Sequential(...) call
// — stacking blocks is just listing several TransformerBlock(...) calls
// among the outer modules.
//
// Decoder blocks with cross-attention are not covered here: cross-
// attention needs two distinct input sequences, which can't be expressed
// as a single-input Sequential chain — compose CrossAttentionLayer
// directly in hand-written Forward code instead.
func TransformerBlock(rng *rand.Rand, dModel, numHeads, ffHidden int, causal bool, init Initializer) (*SequentialModel, error) {
	return Sequential([]int{1, 2, dModel}, // placeholder seqLen=2, construction-time shape validation only
		Residual(nil, MultiHeadAttention(rng, dModel, numHeads, causal, init)),
		LayerNorm(dModel),
		Residual(nil,
			Linear(rng, dModel, ffHidden, init),
			GELU(),
			Linear(rng, ffHidden, dModel, init),
		),
		LayerNorm(dModel),
	)
}
