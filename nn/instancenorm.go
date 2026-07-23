// nn/instancenorm.go
package nn

// InstanceNormLayer normalizes each (sample, channel) pair independently
// over spatial positions only — exactly GroupNorm with one group per
// channel (Ulyanov, Vedaldi & Lempitsky, 2016), kept as its own named
// type/constructor for a clearer API and a dedicated serialization tag
// rather than requiring callers to know the GroupNorm(channels, channels)
// equivalence.
type InstanceNormLayer struct {
	*GroupNormLayer
}

func InstanceNorm(channels int) *InstanceNormLayer {
	return &InstanceNormLayer{GroupNormLayer: GroupNorm(channels, channels)}
}
