package tensor

type Tensor3D struct {
	Data     [][][]float32
	Height   int
	Width    int
	Channels int
}

func NewTensor3D(height, width, channels int) *Tensor3D {
	data := make([][][]float32, channels)
	for c := 0; c < channels; c++ {
		data[c] = make([][]float32, height)
		for h := 0; h < height; h++ {
			data[c][h] = make([]float32, width)
		}
	}
	return &Tensor3D{
		Data:     data,
		Height:   height,
		Width:    width,
		Channels: channels,
	}
}

func FromData(data [][][]float32) *Tensor3D {
	if len(data) == 0 || len(data[0]) == 0 || len(data[0][0]) == 0 {
		return &Tensor3D{Data: data}
	}
	return &Tensor3D{
		Data:     data,
		Channels: len(data),
		Height:   len(data[0]),
		Width:    len(data[0][0]),
	}
}

func (t *Tensor3D) Get(channel, row, col int) float32 {
	return t.Data[channel][row][col]
}

func (t *Tensor3D) Set(channel, row, col int, value float32) {
	t.Data[channel][row][col] = value
}

func (t *Tensor3D) Flatten() []float32 {
	size := t.Height * t.Width * t.Channels
	result := make([]float32, size)
	idx := 0
	for c := 0; c < t.Channels; c++ {
		for h := 0; h < t.Height; h++ {
			for w := 0; w < t.Width; w++ {
				result[idx] = t.Data[c][h][w]
				idx++
			}
		}
	}
	return result
}

func Unflatten(data []float32, height, width, channels int) *Tensor3D {
	tensor := NewTensor3D(height, width, channels)
	idx := 0
	for c := 0; c < channels; c++ {
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				tensor.Data[c][h][w] = data[idx]
				idx++
			}
		}
	}
	return tensor
}

func (t *Tensor3D) Clone() *Tensor3D {
	clone := NewTensor3D(t.Height, t.Width, t.Channels)
	for c := 0; c < t.Channels; c++ {
		for h := 0; h < t.Height; h++ {
			for w := 0; w < t.Width; w++ {
				clone.Data[c][h][w] = t.Data[c][h][w]
			}
		}
	}
	return clone
}

func (t *Tensor3D) Add(other *Tensor3D) {
	for c := 0; c < t.Channels; c++ {
		for h := 0; h < t.Height; h++ {
			for w := 0; w < t.Width; w++ {
				t.Data[c][h][w] += other.Data[c][h][w]
			}
		}
	}
}

func (t *Tensor3D) Scale(factor float32) {
	for c := 0; c < t.Channels; c++ {
		for h := 0; h < t.Height; h++ {
			for w := 0; w < t.Width; w++ {
				t.Data[c][h][w] *= factor
			}
		}
	}
}
