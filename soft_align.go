package attention

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/serializer"
)

func init() {
	var s SoftAlign
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeSoftAlign)
}

// SoftAlign uses an attention mechanism similar to the
// one described in https://arxiv.org/abs/1409.0473.
// In this mechanism, an encoder generates a sequence of
// vectors from the input sequence.
// Then, a decoder RNN produces an output sequence by
// focusing on different parts of the encoded input
// sequence.
//
// At every timestep, the decoder block produces queries
// which are used to focus on parts of the encoded
// sequence for the next timestep.
type SoftAlign struct {
	// Encoder encodes the input sequences.
	Encoder func(in anyseq.Seq) anyseq.Seq

	// Attentor takes queries and encoded vectors and
	// produces weights (in the log domain) for the
	// importance of that encoded vector for the given
	// query.
	//
	// The first input is the query; the second input is the
	// encoded vector.
	Attentor *Combiner

	// Decoder is the block which takes values from
	// InCombiner and produces queries.
	Decoder anyrnn.Block

	// InCombiner combines the results of queries (i.e.
	// averaged encoded vectors) with inputs at the current
	// decoding timestep to produce an input for Decoder.
	//
	// The first input is the query result; the second input
	// is the block input.
	InCombiner *Combiner

	// InitQuery is query for the first decoding timestep.
	InitQuery *anydiff.Var
}

// DeserializeSoftAlign deserializes a SoftAlign.
//
// The resulting instance will have a nil Encoder field.
func DeserializeSoftAlign(d []byte) (*SoftAlign, error) {
	var res SoftAlign
	err := serializer.DeserializeAny(d, &res.Attentor, &res.Decoder, &res.InCombiner,
		&res.InitQuery)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// Block creates a block which uses the decoder in
// conjunction with the batch of inputs.
//
// The resultant block must only be used with a starting
// batch size equal to the batch size of in.
//
// There must be at least one input sequence, and all
// input sequences must be non-empty.
func (s *SoftAlign) Block(in anyseq.Seq) anyrnn.Block {
	if len(in.Output()) == 0 || in.Output()[0].NumPresent() == 0 {
		panic("cannot have no input sequences")
	}
	if len(in.Output()[0].Present) != in.Output()[0].NumPresent() {
		panic("cannot have empty input sequence")
	}
	encoded := s.Encoder(in)
	return &softBlock{
		Internal:   s.Decoder,
		Encoded:    encoded,
		Attentor:   s.Attentor,
		InitQuery:  s.InitQuery,
		ToInternal: s.InCombiner,
	}
}

// Parameters collects the parameters of every component
// of the SoftAlign except for s.Encoder.
func (s *SoftAlign) Parameters() []*anydiff.Var {
	res := []*anydiff.Var{s.InitQuery}
	for _, x := range []interface{}{s.Attentor, s.Decoder, s.InCombiner} {
		if p, ok := x.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a SoftAlign with the serializer package.
func (s *SoftAlign) SerializerType() string {
	return "github.com/unixpickle/attention.SoftAlign"
}

// Serialize serializes the SoftAlign.
//
// s.Encoder is ignored since it cannot be serialized.
func (s *SoftAlign) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		s.Attentor,
		s.Decoder,
		s.InCombiner,
		s.InitQuery,
	)
}
