package Network

import (
	"encoding/json"
	"os"
)

// ModelConfig represents the serializable model configuration
type ModelConfig struct {
	Layers      []LayerConfig `json:"layers"`
	Weights     [][][]float32 `json:"weights"`
	LossType    LossType      `json:"loss_type"`
	Version     string        `json:"version"`
}

// LayerConfig represents a serializable layer configuration
type LayerConfig struct {
	Size           int            `json:"size"`
	ActivationType ActivationType `json:"activation_type"`
}

// SaveToFile saves the network to a JSON file
func (network *Network) SaveToFile(filename string) error {
	config := network.ToConfig()

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(config)
}

// LoadFromFile loads a network from a JSON file
func LoadFromFile(filename string) (Network, error) {
	file, err := os.Open(filename)
	if err != nil {
		return Network{}, err
	}
	defer file.Close()

	var config ModelConfig
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&config); err != nil {
		return Network{}, err
	}

	return NetworkFromConfig(config), nil
}

// ToConfig converts the network to a serializable configuration
func (network *Network) ToConfig() ModelConfig {
	// Extract layer configurations
	// Save the regular size (without bias node) for reconstruction
	layerConfigs := make([]LayerConfig, len(network.layers))
	for i, layer := range network.layers {
		layerConfigs[i] = LayerConfig{
			Size:           layer.RegularSize(), // Save regular size, not including bias
			ActivationType: layer.activation.Type,
		}
	}

	// Weights already include bias connections, so no separate bias needed
	return ModelConfig{
		Layers:   layerConfigs,
		Weights:  network.weights,
		LossType: network.loss.Type,
		Version:  "2.0", // Updated version for new bias format
	}
}

// NetworkFromConfig creates a network from a configuration
func NetworkFromConfig(config ModelConfig) Network {
	// Reconstruct layers - NewNetworkWithLoss will add bias nodes automatically
	layers := make([]Layer, len(config.Layers))
	for i, layerConfig := range config.Layers {
		layers[i] = NewLayerWithActivation(layerConfig.Size, layerConfig.ActivationType)
	}

	// Create network with NewNetworkWithLoss to properly set up bias nodes
	network := NewNetworkWithLoss(layers, config.LossType)

	// Replace weights with loaded weights
	network.weights = config.Weights

	return network
}

// SaveToJSON saves the network to a JSON string
func (network *Network) SaveToJSON() (string, error) {
	config := network.ToConfig()
	bytes, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// LoadFromJSON loads a network from a JSON string
func LoadFromJSON(jsonStr string) (Network, error) {
	var config ModelConfig
	if err := json.Unmarshal([]byte(jsonStr), &config); err != nil {
		return Network{}, err
	}
	return NetworkFromConfig(config), nil
}
