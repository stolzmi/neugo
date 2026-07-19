package Network

import (
	"math"
	"math/rand"
	"neugo/tensor"
)

type Conv2D struct {
	InputChannels  int
	OutputChannels int
	KernelSize     int
	Stride         int
	Padding        int

	Kernels      [][][][]float32
	Biases       []float32
	Activation   ActivationFunction
	Input        *tensor.Tensor3D
	Output       *tensor.Tensor3D
	KernelGrads  [][][][]float32
	BiasGrads    []float32
	InputGrad    *tensor.Tensor3D
	OutputHeight int
	OutputWidth  int
}

func NewConv2D(inputChannels, outputChannels, kernelSize, stride, padding int, activation ActivationType) *Conv2D {
	kernels := make([][][][]float32, outputChannels)
	kernelGrads := make([][][][]float32, outputChannels)

	scale := float32(math.Sqrt(2.0 / float64(inputChannels*kernelSize*kernelSize)))

	for oc := 0; oc < outputChannels; oc++ {
		kernels[oc] = make([][][]float32, inputChannels)
		kernelGrads[oc] = make([][][]float32, inputChannels)
		for ic := 0; ic < inputChannels; ic++ {
			kernels[oc][ic] = make([][]float32, kernelSize)
			kernelGrads[oc][ic] = make([][]float32, kernelSize)
			for kh := 0; kh < kernelSize; kh++ {
				kernels[oc][ic][kh] = make([]float32, kernelSize)
				kernelGrads[oc][ic][kh] = make([]float32, kernelSize)
				for kw := 0; kw < kernelSize; kw++ {
					kernels[oc][ic][kh][kw] = (rand.Float32()*2 - 1) * scale
				}
			}
		}
	}

	biases := make([]float32, outputChannels)
	biasGrads := make([]float32, outputChannels)

	return &Conv2D{
		InputChannels:  inputChannels,
		OutputChannels: outputChannels,
		KernelSize:     kernelSize,
		Stride:         stride,
		Padding:        padding,
		Kernels:        kernels,
		Biases:         biases,
		KernelGrads:    kernelGrads,
		BiasGrads:      biasGrads,
		Activation:     GetActivationFunction(activation),
	}
}

func (c *Conv2D) Forward(input *tensor.Tensor3D) *tensor.Tensor3D {
	c.Input = input

	paddedInput := c.addPadding(input)

	outHeight := (paddedInput.Height-c.KernelSize)/c.Stride + 1
	outWidth := (paddedInput.Width-c.KernelSize)/c.Stride + 1

	c.OutputHeight = outHeight
	c.OutputWidth = outWidth

	output := tensor.NewTensor3D(outHeight, outWidth, c.OutputChannels)

	for oc := 0; oc < c.OutputChannels; oc++ {
		for oh := 0; oh < outHeight; oh++ {
			for ow := 0; ow < outWidth; ow++ {
				sum := float32(0.0)

				for ic := 0; ic < c.InputChannels; ic++ {
					for kh := 0; kh < c.KernelSize; kh++ {
						for kw := 0; kw < c.KernelSize; kw++ {
							ih := oh*c.Stride + kh
							iw := ow*c.Stride + kw
							sum += paddedInput.Data[ic][ih][iw] * c.Kernels[oc][ic][kh][kw]
						}
					}
				}

				sum += c.Biases[oc]
				output.Data[oc][oh][ow] = c.Activation.Apply(sum)
			}
		}
	}

	c.Output = output
	return output
}

func (c *Conv2D) Backward(outputGrad *tensor.Tensor3D, learningRate float32) *tensor.Tensor3D {
	paddedInput := c.addPadding(c.Input)
	inputGrad := tensor.NewTensor3D(paddedInput.Height, paddedInput.Width, c.InputChannels)

	for oc := 0; oc < c.OutputChannels; oc++ {
		for ic := 0; ic < c.InputChannels; ic++ {
			for kh := 0; kh < c.KernelSize; kh++ {
				for kw := 0; kw < c.KernelSize; kw++ {
					c.KernelGrads[oc][ic][kh][kw] = 0
				}
			}
		}
	}

	for oc := 0; oc < c.OutputChannels; oc++ {
		c.BiasGrads[oc] = 0
	}

	for oc := 0; oc < c.OutputChannels; oc++ {
		for oh := 0; oh < c.OutputHeight; oh++ {
			for ow := 0; ow < c.OutputWidth; ow++ {
				activationDeriv := c.Activation.Derivative(c.Output.Data[oc][oh][ow])
				grad := outputGrad.Data[oc][oh][ow] * activationDeriv

				c.BiasGrads[oc] += grad

				for ic := 0; ic < c.InputChannels; ic++ {
					for kh := 0; kh < c.KernelSize; kh++ {
						for kw := 0; kw < c.KernelSize; kw++ {
							ih := oh*c.Stride + kh
							iw := ow*c.Stride + kw

							c.KernelGrads[oc][ic][kh][kw] += grad * paddedInput.Data[ic][ih][iw]
							inputGrad.Data[ic][ih][iw] += grad * c.Kernels[oc][ic][kh][kw]
						}
					}
				}
			}
		}
	}

	for oc := 0; oc < c.OutputChannels; oc++ {
		c.Biases[oc] -= learningRate * c.BiasGrads[oc]
		for ic := 0; ic < c.InputChannels; ic++ {
			for kh := 0; kh < c.KernelSize; kh++ {
				for kw := 0; kw < c.KernelSize; kw++ {
					c.Kernels[oc][ic][kh][kw] -= learningRate * c.KernelGrads[oc][ic][kh][kw]
				}
			}
		}
	}

	c.InputGrad = c.removePadding(inputGrad)
	return c.InputGrad
}

func (c *Conv2D) addPadding(input *tensor.Tensor3D) *tensor.Tensor3D {
	if c.Padding == 0 {
		return input
	}

	paddedHeight := input.Height + 2*c.Padding
	paddedWidth := input.Width + 2*c.Padding
	padded := tensor.NewTensor3D(paddedHeight, paddedWidth, input.Channels)

	for ch := 0; ch < input.Channels; ch++ {
		for h := 0; h < input.Height; h++ {
			for w := 0; w < input.Width; w++ {
				padded.Data[ch][h+c.Padding][w+c.Padding] = input.Data[ch][h][w]
			}
		}
	}

	return padded
}

func (c *Conv2D) removePadding(input *tensor.Tensor3D) *tensor.Tensor3D {
	if c.Padding == 0 {
		return input
	}

	unpadded := tensor.NewTensor3D(
		input.Height-2*c.Padding,
		input.Width-2*c.Padding,
		input.Channels,
	)

	for ch := 0; ch < input.Channels; ch++ {
		for h := 0; h < unpadded.Height; h++ {
			for w := 0; w < unpadded.Width; w++ {
				unpadded.Data[ch][h][w] = input.Data[ch][h+c.Padding][w+c.Padding]
			}
		}
	}

	return unpadded
}
