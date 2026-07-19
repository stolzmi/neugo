package Network

import (
	"math/rand"
	"neugo/tensor"
)

type CNNLayer interface {
	IsConv() bool
}

type ConvLayer struct {
	Conv *Conv2D
}

func (c ConvLayer) IsConv() bool { return true }

type PoolLayer struct {
	Pool *MaxPool2D
}

func (p PoolLayer) IsConv() bool { return true }

type FlattenLayer struct {
	Flatten *Flatten
}

func (f FlattenLayer) IsConv() bool { return false }

type CNN struct {
	ConvLayers      []interface{}
	DenseNetwork    *Network
	FlattenLayer    *Flatten
	InputHeight     int
	InputWidth      int
	InputChannels   int
	CurrentChannels int
	CurrentHeight   int
	CurrentWidth    int
	Loss            LossFunction
}

func NewCNN(inputHeight, inputWidth, inputChannels int, lossType LossType) *CNN {
	return &CNN{
		ConvLayers:      make([]interface{}, 0),
		InputHeight:     inputHeight,
		InputWidth:      inputWidth,
		InputChannels:   inputChannels,
		CurrentChannels: inputChannels,
		CurrentHeight:   inputHeight,
		CurrentWidth:    inputWidth,
		Loss:            GetLossFunction(lossType),
	}
}

func (cnn *CNN) AddConv2D(outputChannels, kernelSize, stride, padding int, activation ActivationType) {
	conv := NewConv2D(cnn.CurrentChannels, outputChannels, kernelSize, stride, padding, activation)
	cnn.ConvLayers = append(cnn.ConvLayers, conv)

	paddedHeight := cnn.CurrentHeight + 2*padding
	paddedWidth := cnn.CurrentWidth + 2*padding
	cnn.CurrentHeight = (paddedHeight-kernelSize)/stride + 1
	cnn.CurrentWidth = (paddedWidth-kernelSize)/stride + 1
	cnn.CurrentChannels = outputChannels
}

func (cnn *CNN) AddMaxPool2D(poolSize, stride int) {
	pool := NewMaxPool2D(poolSize, stride)
	cnn.ConvLayers = append(cnn.ConvLayers, pool)

	cnn.CurrentHeight = (cnn.CurrentHeight-poolSize)/stride + 1
	cnn.CurrentWidth = (cnn.CurrentWidth-poolSize)/stride + 1
}

func (cnn *CNN) AddAvgPool2D(poolSize, stride int) {
	pool := NewAvgPool2D(poolSize, stride)
	cnn.ConvLayers = append(cnn.ConvLayers, pool)

	cnn.CurrentHeight = (cnn.CurrentHeight-poolSize)/stride + 1
	cnn.CurrentWidth = (cnn.CurrentWidth-poolSize)/stride + 1
}

func (cnn *CNN) AddFlatten() {
	cnn.FlattenLayer = NewFlatten()
}

func (cnn *CNN) SetDenseNetwork(layers []Layer) {
	cnn.DenseNetwork = &Network{
		layers:  layers,
		weights: nil,
		loss:    cnn.Loss,
	}

	weights := make([][][]float32, len(layers)-1)
	for i := 0; i < len(layers)-1; i++ {
		weights[i] = make([][]float32, layers[i].Size())
		for j := 0; j < layers[i].Size(); j++ {
			weights[i][j] = make([]float32, layers[i+1].Size())
			scale := float32(1.0) / float32(layers[i].Size())
			for k := 0; k < layers[i+1].Size(); k++ {
				weights[i][j][k] = (rand.Float32()*2 - 1) * scale
			}
		}
	}
	cnn.DenseNetwork.weights = weights
}

func (cnn *CNN) ForwardPass(input *tensor.Tensor3D) {
	current := input

	for _, layer := range cnn.ConvLayers {
		switch l := layer.(type) {
		case *Conv2D:
			current = l.Forward(current)
		case *MaxPool2D:
			current = l.Forward(current)
		case *AvgPool2D:
			current = l.Forward(current)
		}
	}

	flattened := cnn.FlattenLayer.Forward(current)

	cnn.DenseNetwork.ForwardPass(flattened)
}

func (cnn *CNN) BackPropagation(input *tensor.Tensor3D, labels []float32, learningRate float32) {
	cnn.ForwardPass(input)

	cnn.DenseNetwork.BackPropagation(labels, learningRate)

	denseInputGrad := make([]float32, len(cnn.DenseNetwork.layers[0].neurons))
	for i := range cnn.DenseNetwork.layers[0].neurons {
		denseInputGrad[i] = cnn.DenseNetwork.layers[0].neurons[i].Gradient()
	}

	convGrad := cnn.FlattenLayer.Backward(denseInputGrad)

	for i := len(cnn.ConvLayers) - 1; i >= 0; i-- {
		layer := cnn.ConvLayers[i]
		switch l := layer.(type) {
		case *Conv2D:
			convGrad = l.Backward(convGrad, learningRate)
		case *MaxPool2D:
			convGrad = l.Backward(convGrad)
		case *AvgPool2D:
			convGrad = l.Backward(convGrad)
		}
	}
}

func (cnn *CNN) Train(inputs []*tensor.Tensor3D, labels [][]float32, epochs int, learningRate float32) []float32 {
	losses := make([]float32, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0.0)

		for i := 0; i < len(inputs); i++ {
			cnn.ForwardPass(inputs[i])

			loss := cnn.Loss.Calculate(
				[]float32{cnn.DenseNetwork.GetOutput()[0].Activation()},
				labels[i],
			)
			epochLoss += loss

			cnn.BackPropagation(inputs[i], labels[i], learningRate)
		}

		losses[epoch] = epochLoss / float32(len(inputs))

		if (epoch+1)%10 == 0 {
			println("Epoch", epoch+1, "Loss:", losses[epoch])
		}
	}

	return losses
}

func (cnn *CNN) Predict(input *tensor.Tensor3D) float32 {
	cnn.ForwardPass(input)
	return cnn.DenseNetwork.GetOutput()[0].Activation()
}

func (cnn *CNN) Evaluate(inputs []*tensor.Tensor3D, labels [][]float32, threshold float32) Metrics {
	predictions := make([]float32, len(inputs))

	for i := 0; i < len(inputs); i++ {
		predictions[i] = cnn.Predict(inputs[i])
	}

	tp := 0
	tn := 0
	fp := 0
	fn := 0
	totalLoss := float32(0.0)

	for i := 0; i < len(predictions); i++ {
		pred := predictions[i]
		actual := labels[i][0]

		totalLoss += cnn.Loss.Calculate([]float32{pred}, labels[i])

		if pred >= threshold && actual == 1 {
			tp++
		} else if pred >= threshold && actual == 0 {
			fp++
		} else if pred < threshold && actual == 0 {
			tn++
		} else {
			fn++
		}
	}

	accuracy := float32(tp+tn) / float32(len(predictions)) * 100

	precision := float32(0)
	if tp+fp > 0 {
		precision = float32(tp) / float32(tp+fp)
	}

	recall := float32(0)
	if tp+fn > 0 {
		recall = float32(tp) / float32(tp+fn)
	}

	f1 := float32(0)
	if precision+recall > 0 {
		f1 = 2 * (precision * recall) / (precision + recall)
	}

	confusionMatrix := [][]int{
		{tn, fp},
		{fn, tp},
	}

	return Metrics{
		Accuracy:        accuracy,
		Precision:       precision,
		Recall:          recall,
		F1Score:         f1,
		Loss:            totalLoss / float32(len(predictions)),
		ConfusionMatrix: confusionMatrix,
	}
}
