package main

import (
	"fmt"
	"neugo/Network"
)

// Custom MLP module (Flax NNX style)
type MLP struct {
	Linear1 *Network.LinearModule
	Dropout *Network.Dropout
	BN      *Network.BatchNorm
	Linear2 *Network.LinearModule
}

// NewMLP creates a new MLP (similar to __init__)
func NewMLP(din, dmid, dout int, dropoutRate float32) *MLP {
	return &MLP{
		Linear1: Network.NewLinear(din, dmid, Network.Linear),
		Dropout: Network.NewDropout(dropoutRate),
		BN:      Network.NewBatchNorm(dmid),
		Linear2: Network.NewLinear(dmid, dout, Network.Linear),
	}
}

// Forward performs forward pass (similar to __call__)
func (mlp *MLP) Forward(x []float32) []float32 {
	// x = gelu(dropout(bn(linear1(x))))
	x = mlp.Linear1.Forward(x)
	x = mlp.BN.Forward(x)
	x = mlp.Dropout.Forward(x)
	x = Network.GELUFunc(x)

	// return linear2(x)
	return mlp.Linear2.Forward(x)
}

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                                ║")
	fmt.Println("║          🎯  NEUGO NNX-STYLE API DEMONSTRATION  🎯             ║")
	fmt.Println("║            Flax NNX-Inspired Module System                     ║")
	fmt.Println("║                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// ═══════════════════════════════════════════════════════════════════
	// STYLE 1: Custom Module (Like Flax NNX)
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n📦 STYLE 1: Custom Module (Flax NNX Style)")
	fmt.Println("────────────────────────────────────────────────────────────────")

	fmt.Println("\n// Define custom module:")
	fmt.Println("type MLP struct {")
	fmt.Println("    Linear1 *Network.LinearModule")
	fmt.Println("    Dropout *Network.Dropout")
	fmt.Println("    BN      *Network.BatchNorm")
	fmt.Println("    Linear2 *Network.LinearModule")
	fmt.Println("}")
	fmt.Println()
	fmt.Println("func NewMLP(din, dmid, dout int, dropoutRate float32) *MLP {")
	fmt.Println("    return &MLP{")
	fmt.Println("        Linear1: Network.NewLinear(din, dmid, Network.Linear),")
	fmt.Println("        Dropout: Network.NewDropout(dropoutRate),")
	fmt.Println("        BN:      Network.NewBatchNorm(dmid),")
	fmt.Println("        Linear2: Network.NewLinear(dmid, dout, Network.Linear),")
	fmt.Println("    }")
	fmt.Println("}")
	fmt.Println()
	fmt.Println("func (mlp *MLP) Forward(x []float32) []float32 {")
	fmt.Println("    x = mlp.Linear1.Forward(x)")
	fmt.Println("    x = mlp.BN.Forward(x)")
	fmt.Println("    x = mlp.Dropout.Forward(x)")
	fmt.Println("    x = Network.GELU(x)")
	fmt.Println("    return mlp.Linear2.Forward(x)")
	fmt.Println("}")

	// Create instance
	fmt.Println("\n// Create and use:")
	fmt.Println("model := NewMLP(10, 64, 1, 0.1)")

	model := NewMLP(10, 64, 1, 0.1)

	// Test forward pass
	input := make([]float32, 10)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output := model.Forward(input)
	fmt.Printf("\n✓ Forward pass successful!")
	fmt.Printf("\n✓ Input size: %d", len(input))
	fmt.Printf("\n✓ Output size: %d", len(output))
	fmt.Printf("\n✓ Output value: %.6f", output[0])

	// ═══════════════════════════════════════════════════════════════════
	// STYLE 2: Built-in MLP Module
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n📦 STYLE 2: Built-in MLP Module")
	fmt.Println("────────────────────────────────────────────────────────────────")

	mlp := Network.NewMLPModule(10, 64, 1, 0.1)

	fmt.Println("\nmlp := Network.NewMLP(din, dmid, dout, dropoutRate)")
	fmt.Println("output := mlp.Forward(input)")

	output2 := mlp.Forward(input)
	fmt.Printf("\n✓ Forward pass successful!")
	fmt.Printf("\n✓ Parameters: %d", mlp.Parameters())
	fmt.Printf("\n✓ Output: %.6f", output2[0])

	// ═══════════════════════════════════════════════════════════════════
	// STYLE 3: Sequential Module
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n📦 STYLE 3: Sequential Module (Container)")
	fmt.Println("────────────────────────────────────────────────────────────────")

	sequential := Network.NewSequentialModule(
		Network.NewLinear(10, 64, Network.ReLU),
		Network.NewBatchNorm(64),
		Network.NewDropout(0.2),
		Network.NewLinear(64, 32, Network.ReLU),
		Network.NewLinear(32, 1, Network.Sigmoid),
	)

	fmt.Println("\nsequential := Network.NewSequentialModule(")
	fmt.Println("    Network.NewLinear(10, 64, Network.ReLU),")
	fmt.Println("    Network.NewBatchNorm(64),")
	fmt.Println("    Network.NewDropout(0.2),")
	fmt.Println("    Network.NewLinear(64, 32, Network.ReLU),")
	fmt.Println("    Network.NewLinear(32, 1, Network.Sigmoid),")
	fmt.Println(")")

	output3 := sequential.Forward(input)
	fmt.Printf("\n✓ Forward pass successful!")
	fmt.Printf("\n✓ Total parameters: %d", sequential.Parameters())
	fmt.Printf("\n✓ Output: %.6f", output3[0])

	// ═══════════════════════════════════════════════════════════════════
	// STYLE 4: High-Level Builders
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n📦 STYLE 4: High-Level Module Builders")
	fmt.Println("────────────────────────────────────────────────────────────────")

	// Simple MLP
	simple := Network.SimpleMLP(10, 64, 1)
	fmt.Println("\n// Simple MLP:")
	fmt.Println("model := Network.SimpleMLP(din, dmid, dout)")
	output4 := simple.Forward(input)
	fmt.Printf("✓ Output: %.6f, Parameters: %d\n", output4[0], simple.Parameters())

	// MLP Classifier
	classifier := Network.MLPClassifier(10, 64, 3, 0.1, true)
	fmt.Println("\n// MLP Classifier:")
	fmt.Println("model := Network.MLPClassifier(din, dmid, dout, dropout, useNorm)")
	output5 := classifier.Forward(input)
	fmt.Printf("✓ Output size: %d, Parameters: %d\n", len(output5), classifier.Parameters())

	// Deep MLP
	deep := Network.DeepMLP(10, []int{128, 64, 32}, 1, 0.2)
	fmt.Println("\n// Deep MLP:")
	fmt.Println("model := Network.DeepMLP(din, []int{128, 64, 32}, dout, dropout)")
	output6 := deep.Forward(input)
	fmt.Printf("✓ Output: %.6f, Parameters: %d\n", output6[0], deep.Parameters())

	// ═══════════════════════════════════════════════════════════════════
	// DEMONSTRATION: Module Composition
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n📦 DEMONSTRATION: Module Composition")
	fmt.Println("────────────────────────────────────────────────────────────────")

	// Build a custom architecture by composing modules
	encoder := Network.NewSequentialModule(
		Network.NewLinear(10, 128, Network.ReLU),
		Network.NewBatchNorm(128),
		Network.NewDropout(0.1),
		Network.NewLinear(128, 64, Network.ReLU),
	)

	decoder := Network.NewSequentialModule(
		Network.NewLinear(64, 32, Network.ReLU),
		Network.NewLinear(32, 1, Network.Sigmoid),
	)

	fmt.Println("\n// Compose modules:")
	fmt.Println("encoder := NewSequentialModule(...)")
	fmt.Println("decoder := NewSequentialModule(...)")
	fmt.Println()
	fmt.Println("// Use in sequence:")
	fmt.Println("encoded := encoder.Forward(input)")
	fmt.Println("output := decoder.Forward(encoded)")

	encoded := encoder.Forward(input)
	finalOutput := decoder.Forward(encoded)

	fmt.Printf("\n✓ Encoded size: %d", len(encoded))
	fmt.Printf("\n✓ Final output: %.6f", finalOutput[0])
	fmt.Printf("\n✓ Total parameters: %d", encoder.Parameters()+decoder.Parameters())

	// ═══════════════════════════════════════════════════════════════════
	// COMPARISON: NNX vs Traditional
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n📦 COMPARISON: NNX Style vs Traditional")
	fmt.Println("────────────────────────────────────────────────────────────────")

	fmt.Println("\n// Traditional (verbose):")
	fmt.Println("layer1 := Network.NewLayerWithActivation(10, Network.Linear)")
	fmt.Println("layer2 := Network.NewLayerWithActivation(64, Network.ReLU)")
	fmt.Println("layer3 := Network.NewLayerWithActivation(1, Network.Sigmoid)")
	fmt.Println("layers := []Network.Layer{layer1, layer2, layer3}")
	fmt.Println("network := Network.NewNetwork(layers)")

	fmt.Println("\n// NNX Style (clean):")
	fmt.Println("model := Network.SimpleMLP(10, 64, 1)")
	fmt.Println("output := model.Forward(input)")

	// ═══════════════════════════════════════════════════════════════════
	// Summary
	// ═══════════════════════════════════════════════════════════════════
	fmt.Println("\n\n╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        ✨ SUMMARY ✨                           ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("✅ NNX-Style Features:")
	fmt.Println("   • Custom modules with explicit structure")
	fmt.Println("   • Forward() method (like __call__)")
	fmt.Println("   • Module composition")
	fmt.Println("   • Clean, readable code")
	fmt.Println("   • Explicit layer definitions")
	fmt.Println()
	fmt.Println("🏗️  Module Types:")
	fmt.Println("   • Linear - fully connected layer")
	fmt.Println("   • Dropout - regularization")
	fmt.Println("   • BatchNorm - normalization")
	fmt.Println("   • Sequential - container for modules")
	fmt.Println()
	fmt.Println("🎯 Activation Functions:")
	fmt.Println("   • GELU, ReLU, Sigmoid, Tanh")
	fmt.Println("   • Element-wise operations")
	fmt.Println()
	fmt.Println("📚 High-Level Builders:")
	fmt.Println("   • SimpleMLP - basic MLP")
	fmt.Println("   • MLPClassifier - classification MLP")
	fmt.Println("   • DeepMLP - deep architecture")
	fmt.Println("   • Custom modules - full flexibility")
	fmt.Println()
	fmt.Println("💡 Benefits:")
	fmt.Println("   • Similar to Flax NNX (familiar to Python devs)")
	fmt.Println("   • Explicit and readable")
	fmt.Println("   • Composable modules")
	fmt.Println("   • Easy to extend with custom modules")
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
}
