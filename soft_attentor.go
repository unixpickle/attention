package attention

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var s SoftAttentor
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeSoftAttentor)
}

// A SoftAttentor is a feed-forward network that takes a
// "query vector" and an encoded vector as inputs and
// produces a weighting (in the log domain) as output.
//
// Typically, the query vector will be taken from the
// output of a decoder RNN.
//
// For efficiency, the first layer is split up into a
// query transformation and an encoder transformation.
// The vectors from these two transformations are added
// and then fed to OutTrans.
type SoftAttentor struct {
	QueryTrans anynet.Layer
	EncTrans   anynet.Layer
	OutTrans   anynet.Layer
}

// DeserializeSoftAttentor deserializes a SoftAttentor.
func DeserializeSoftAttentor(d []byte) (*SoftAttentor, error) {
	var s SoftAttentor
	err := serializer.DeserializeAny(d, &s.QueryTrans, &s.EncTrans, &s.OutTrans)
	if err != nil {
		return nil, essentials.AddCtx("deserialize SoftAttentor", err)
	}
	return &s, nil
}

// Parameters returns the network's parameters.
func (s *SoftAttentor) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, l := range []anynet.Layer{s.QueryTrans, s.EncTrans, s.OutTrans} {
		if p, ok := l.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a SoftAttentor with the serializer package.
func (s *SoftAttentor) SerializerType() string {
	return "github.com/unixpickle/attention.SoftAttentor"
}

// Serialize serializes a SoftAttentor.
func (s *SoftAttentor) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		s.QueryTrans,
		s.EncTrans,
		s.OutTrans,
	)
}
