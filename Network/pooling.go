package Network

import "neugo/tensor"

type MaxPool2D struct {
	PoolSize     int
	Stride       int
	Input        *tensor.Tensor3D
	Output       *tensor.Tensor3D
	MaxIndices   [][][]int
	OutputHeight int
	OutputWidth  int
}

func NewMaxPool2D(poolSize, stride int) *MaxPool2D {
	return &MaxPool2D{
		PoolSize: poolSize,
		Stride:   stride,
	}
}

func (m *MaxPool2D) Forward(input *tensor.Tensor3D) *tensor.Tensor3D {
	m.Input = input

	outHeight := (input.Height-m.PoolSize)/m.Stride + 1
	outWidth := (input.Width-m.PoolSize)/m.Stride + 1

	m.OutputHeight = outHeight
	m.OutputWidth = outWidth

	output := tensor.NewTensor3D(outHeight, outWidth, input.Channels)
	m.MaxIndices = make([][][]int, input.Channels)

	for c := 0; c < input.Channels; c++ {
		m.MaxIndices[c] = make([][]int, outHeight*outWidth)

		for oh := 0; oh < outHeight; oh++ {
			for ow := 0; ow < outWidth; ow++ {
				maxVal := float32(-1e9)
				maxH := 0
				maxW := 0

				for ph := 0; ph < m.PoolSize; ph++ {
					for pw := 0; pw < m.PoolSize; pw++ {
						ih := oh*m.Stride + ph
						iw := ow*m.Stride + pw

						if input.Data[c][ih][iw] > maxVal {
							maxVal = input.Data[c][ih][iw]
							maxH = ih
							maxW = iw
						}
					}
				}

				output.Data[c][oh][ow] = maxVal
				idx := oh*outWidth + ow
				m.MaxIndices[c][idx] = []int{maxH, maxW}
			}
		}
	}

	m.Output = output
	return output
}

func (m *MaxPool2D) Backward(outputGrad *tensor.Tensor3D) *tensor.Tensor3D {
	inputGrad := tensor.NewTensor3D(m.Input.Height, m.Input.Width, m.Input.Channels)

	for c := 0; c < m.Input.Channels; c++ {
		for oh := 0; oh < m.OutputHeight; oh++ {
			for ow := 0; ow < m.OutputWidth; ow++ {
				idx := oh*m.OutputWidth + ow
				maxH := m.MaxIndices[c][idx][0]
				maxW := m.MaxIndices[c][idx][1]

				inputGrad.Data[c][maxH][maxW] += outputGrad.Data[c][oh][ow]
			}
		}
	}

	return inputGrad
}

type AvgPool2D struct {
	PoolSize     int
	Stride       int
	Input        *tensor.Tensor3D
	Output       *tensor.Tensor3D
	OutputHeight int
	OutputWidth  int
}

func NewAvgPool2D(poolSize, stride int) *AvgPool2D {
	return &AvgPool2D{
		PoolSize: poolSize,
		Stride:   stride,
	}
}

func (a *AvgPool2D) Forward(input *tensor.Tensor3D) *tensor.Tensor3D {
	a.Input = input

	outHeight := (input.Height-a.PoolSize)/a.Stride + 1
	outWidth := (input.Width-a.PoolSize)/a.Stride + 1

	a.OutputHeight = outHeight
	a.OutputWidth = outWidth

	output := tensor.NewTensor3D(outHeight, outWidth, input.Channels)

	poolArea := float32(a.PoolSize * a.PoolSize)

	for c := 0; c < input.Channels; c++ {
		for oh := 0; oh < outHeight; oh++ {
			for ow := 0; ow < outWidth; ow++ {
				sum := float32(0.0)

				for ph := 0; ph < a.PoolSize; ph++ {
					for pw := 0; pw < a.PoolSize; pw++ {
						ih := oh*a.Stride + ph
						iw := ow*a.Stride + pw
						sum += input.Data[c][ih][iw]
					}
				}

				output.Data[c][oh][ow] = sum / poolArea
			}
		}
	}

	a.Output = output
	return output
}

func (a *AvgPool2D) Backward(outputGrad *tensor.Tensor3D) *tensor.Tensor3D {
	inputGrad := tensor.NewTensor3D(a.Input.Height, a.Input.Width, a.Input.Channels)

	poolArea := float32(a.PoolSize * a.PoolSize)

	for c := 0; c < a.Input.Channels; c++ {
		for oh := 0; oh < a.OutputHeight; oh++ {
			for ow := 0; ow < a.OutputWidth; ow++ {
				grad := outputGrad.Data[c][oh][ow] / poolArea

				for ph := 0; ph < a.PoolSize; ph++ {
					for pw := 0; pw < a.PoolSize; pw++ {
						ih := oh*a.Stride + ph
						iw := ow*a.Stride + pw
						inputGrad.Data[c][ih][iw] += grad
					}
				}
			}
		}
	}

	return inputGrad
}
