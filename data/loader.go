package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// CSVConfig holds configuration for CSV loading
type CSVConfig struct {
	Delimiter        rune   // CSV delimiter (e.g., ',' or ';')
	HasHeader        bool   // Whether CSV has header row
	SkipColumns      []int  // Column indices to skip
	LabelColumn      int    // Index of label column (-1 for last column)
	LabelType        string // "binary", "multiclass", "regression"
	BinaryThreshold  float64 // For binary classification (value > threshold = 1)
	FeatureColumns   []int  // Specific feature columns (empty = all except label)
}

// Dataset holds loaded data
type Dataset struct {
	Features      [][]float32
	Labels        [][]float32
	FeatureNames  []string
	LabelName     string
	NumSamples    int
	NumFeatures   int
	RawLabels     []float32 // Original label values before transformation
}

// LoadCSV loads a CSV file with flexible configuration
func LoadCSV(filepath string, config CSVConfig) (*Dataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = config.Delimiter
	reader.LazyQuotes = true
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %v", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("CSV file is empty")
	}

	dataset := &Dataset{}
	startRow := 0

	// Handle header
	if config.HasHeader {
		dataset.FeatureNames = records[0]
		startRow = 1
	}

	// Determine label column
	labelCol := config.LabelColumn
	if labelCol == -1 {
		labelCol = len(records[0]) - 1
	}

	// Parse records
	records = records[startRow:]
	dataset.NumSamples = len(records)
	dataset.Features = make([][]float32, dataset.NumSamples)
	dataset.Labels = make([][]float32, dataset.NumSamples)
	dataset.RawLabels = make([]float32, dataset.NumSamples)

	skipMap := make(map[int]bool)
	for _, col := range config.SkipColumns {
		skipMap[col] = true
	}
	skipMap[labelCol] = true

	// Determine feature columns
	numCols := len(records[0])
	featureCols := make([]int, 0)
	if len(config.FeatureColumns) > 0 {
		featureCols = config.FeatureColumns
	} else {
		for i := 0; i < numCols; i++ {
			if !skipMap[i] {
				featureCols = append(featureCols, i)
			}
		}
	}

	dataset.NumFeatures = len(featureCols)

	// Parse data
	for i, record := range records {
		// Parse features
		dataset.Features[i] = make([]float32, dataset.NumFeatures)
		for j, colIdx := range featureCols {
			val, err := strconv.ParseFloat(record[colIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("failed to parse feature at row %d, col %d: %v", i+startRow, colIdx, err)
			}
			dataset.Features[i][j] = float32(val)
		}

		// Parse label
		labelVal, err := strconv.ParseFloat(record[labelCol], 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse label at row %d: %v", i+startRow, err)
		}
		dataset.RawLabels[i] = float32(labelVal)

		// Transform label based on type
		switch config.LabelType {
		case "binary":
			if labelVal > config.BinaryThreshold {
				dataset.Labels[i] = []float32{1.0}
			} else {
				dataset.Labels[i] = []float32{0.0}
			}
		case "regression":
			dataset.Labels[i] = []float32{float32(labelVal)}
		case "multiclass":
			// One-hot encoding would go here
			// For now, just store as single value
			dataset.Labels[i] = []float32{float32(labelVal)}
		default:
			dataset.Labels[i] = []float32{float32(labelVal)}
		}
	}

	return dataset, nil
}

// SaveCSV saves predictions to CSV file
func SaveCSV(filepath string, predictions []float32, threshold float32, includeClass bool) error {
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if includeClass {
		writer.Write([]string{"SampleID", "Probability", "PredictedClass"})
	} else {
		writer.Write([]string{"SampleID", "Prediction"})
	}

	// Write predictions
	for i, pred := range predictions {
		if includeClass {
			class := "0"
			if pred >= threshold {
				class = "1"
			}
			writer.Write([]string{
				fmt.Sprintf("%d", i),
				fmt.Sprintf("%.6f", pred),
				class,
			})
		} else {
			writer.Write([]string{
				fmt.Sprintf("%d", i),
				fmt.Sprintf("%.6f", pred),
			})
		}
	}

	return nil
}

// QuickLoad provides a simple interface for common CSV formats
func QuickLoadBinaryCSV(filepath string, delimiter rune, labelThreshold float64) (*Dataset, error) {
	config := CSVConfig{
		Delimiter:       delimiter,
		HasHeader:       true,
		LabelColumn:     -1, // Last column
		LabelType:       "binary",
		BinaryThreshold: labelThreshold,
	}
	return LoadCSV(filepath, config)
}

// QuickLoadRegressionCSV loads a regression dataset
func QuickLoadRegressionCSV(filepath string, delimiter rune) (*Dataset, error) {
	config := CSVConfig{
		Delimiter:   delimiter,
		HasHeader:   true,
		LabelColumn: -1,
		LabelType:   "regression",
	}
	return LoadCSV(filepath, config)
}
