package attention

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// A TimeStepper implements some sort of model which can
// produce a sequence of vectors, one at a time, given the
// next input.
type TimeStepper interface {
	StepTime(in linalg.Vector) linalg.Vector
}

// SoftAlign uses an attention mechanism similar to the
// one described in https://arxiv.org/abs/1409.0473.
// In this mechanism, an encoder generates a sequence of
// vectors from the input sequence, allowing a decoder
// to produce an output sequence by focusing on different
// parts of the encoded input sequence.
type SoftAlign struct {
	// Encoder encodes input sequences so that the attention
	// mechanism has meaningful features to look at.
	Encoder seqfunc.RFunc

	// Attentor decides how important a vector from the
	// encoder is for a certain stage in the decoder.
	//
	// Inputs to Attentor are of the form <query, encvec>,
	// representing the query from the Decoder concatenated
	// with a feature vector from the Encoder.
	//
	// The Attentor outputs an "energy", which is transformed
	// into a probability using a softmax with the other
	// energies for the query.
	Attentor neuralnet.Network

	// Decoder produces outputs by focusing on the encoded
	// input sequence.
	//
	// The Decoder's input at each timestep is the encoded
	// feature vector produced after the previous timestep.
	//
	// The first QuerySize components of the block's output
	// are fed directly to the Attentor in order to produce
	// the next feature vector.
	Decoder rnn.Block

	// BatchSize is the batch size for applying the Attentor.
	// If this is 0, inputs will not be batched.
	BatchSize int

	// StartQuery is the initial attention query, used to
	// focus for the first timestep.
	StartQuery *autofunc.Variable
}

// Apply applies the RNN to some input sequences.
//
// The in parameter specifies the inputs to be fed to the
// encoder.
//
// The decIn parameter specifies the additional inputs to
// be fed to the decoder at each decoding timestep.
// The returned sequences will be the same lengths as the
// ones in decIn.
func (s *SoftAlign) Apply(in seqfunc.Result, decIn seqfunc.Result) seqfunc.Result {
	encoded := s.Encoder.ApplySeqs(in)
	n := len(encoded.OutputSeqs())
	return seqfunc.Pool(encoded, func(encoded seqfunc.Result) seqfunc.Result {
		tempBlock := interfacerBlock{
			Resources:  make([]autofunc.RFunc, n),
			Block:      s.Decoder,
			StartQuery: s.StartQuery,
		}
		attentor := s.Attentor.BatchLearner()
		for i := range tempBlock.Resources {
			tempBlock.Resources[i] = &focusFunc{
				Encoded:   encoded,
				Attentor:  attentor,
				BatchSize: s.BatchSize,
				Lane:      i,
			}
		}
		tempSeqFunc := rnn.BlockSeqFunc{B: &tempBlock}
		return tempSeqFunc.ApplySeqs(decIn)
	})
}

// ApplyR is like Apply, but with R-operator support.
func (s *SoftAlign) ApplyR(rv autofunc.RVector, in seqfunc.RResult,
	decIn seqfunc.RResult) seqfunc.RResult {
	encoded := s.Encoder.ApplySeqsR(rv, in)
	n := len(encoded.OutputSeqs())
	return seqfunc.PoolR(encoded, func(encoded seqfunc.RResult) seqfunc.RResult {
		tempBlock := interfacerBlock{
			Resources:  make([]autofunc.RFunc, n),
			Block:      s.Decoder,
			StartQuery: s.StartQuery,
		}
		attentor := s.Attentor.BatchLearner()
		for i := range tempBlock.Resources {
			tempBlock.Resources[i] = &focusFunc{
				EncodedR:  encoded,
				Attentor:  attentor,
				BatchSize: s.BatchSize,
				Lane:      i,
			}
		}
		tempSeqFunc := rnn.BlockSeqFunc{B: &tempBlock}
		return tempSeqFunc.ApplySeqsR(rv, decIn)
	})
}

// TimeStepper generates a TimeStepper for the input
// sequence.
func (s *SoftAlign) Generate(in []linalg.Vector) TimeStepper {
	inSeq := seqfunc.ConstResult([][]linalg.Vector{in})
	encoded := s.Encoder.ApplySeqs(inSeq)
	attentor := s.Attentor.BatchLearner()
	tempBlock := interfacerBlock{
		Resources: []autofunc.RFunc{
			&focusFunc{
				Encoded:   encoded,
				Attentor:  attentor,
				BatchSize: s.BatchSize,
				Lane:      0,
			},
		},
		Block:      s.Decoder,
		StartQuery: s.StartQuery,
	}
	return &rnn.Runner{Block: &tempBlock}
}

type focusFunc struct {
	Encoded   seqfunc.Result
	EncodedR  seqfunc.RResult
	Attentor  autofunc.RBatcher
	BatchSize int
	Lane      int
}

func (f *focusFunc) Apply(query autofunc.Result) autofunc.Result {
	if f.Encoded == nil {
		panic("no relevant encoded sequence")
	}
	return autofunc.Pool(query, func(query autofunc.Result) autofunc.Result {
		enc := seqfunc.SliceList(f.Encoded, f.Lane, f.Lane+1)
		augmented := seqfunc.Map(enc, func(encVec autofunc.Result) autofunc.Result {
			return autofunc.Concat(query, encVec)
		})
		mapper := seqfunc.FixedMapBatcher{
			B:         f.Attentor,
			BatchSize: f.BatchSize,
		}
		if mapper.BatchSize == 0 {
			mapper.BatchSize = 1
		}
		energies := mapper.ApplySeqs(augmented)
		exps := seqfunc.Map(energies, autofunc.Exp{}.Apply)
		normalizer := autofunc.Inverse(seqfunc.AddAll(exps))
		probs := seqfunc.Map(exps, func(in autofunc.Result) autofunc.Result {
			return autofunc.Mul(in, normalizer)
		})
		masked := seqfunc.MapN(func(ins ...autofunc.Result) autofunc.Result {
			return autofunc.ScaleFirst(ins[0], ins[1])
		}, enc, probs)
		return seqfunc.AddAll(masked)
	})
}

func (f *focusFunc) ApplyR(rv autofunc.RVector, query autofunc.RResult) autofunc.RResult {
	if f.EncodedR == nil {
		panic("no relevant encoded sequence")
	}
	return autofunc.PoolR(query, func(query autofunc.RResult) autofunc.RResult {
		enc := seqfunc.SliceListR(f.EncodedR, f.Lane, f.Lane+1)
		augmented := seqfunc.MapR(enc, func(encVec autofunc.RResult) autofunc.RResult {
			return autofunc.ConcatR(query, encVec)
		})
		mapper := seqfunc.FixedMapRBatcher{
			B:         f.Attentor,
			BatchSize: f.BatchSize,
		}
		if mapper.BatchSize == 0 {
			mapper.BatchSize = 1
		}
		energies := mapper.ApplySeqsR(rv, augmented)
		exps := seqfunc.MapR(energies, func(in autofunc.RResult) autofunc.RResult {
			return autofunc.Exp{}.ApplyR(rv, in)
		})
		normalizer := autofunc.InverseR(seqfunc.AddAllR(exps))
		probs := seqfunc.MapR(exps, func(in autofunc.RResult) autofunc.RResult {
			return autofunc.MulR(in, normalizer)
		})
		masked := seqfunc.MapNR(func(ins ...autofunc.RResult) autofunc.RResult {
			return autofunc.ScaleFirstR(ins[0], ins[1])
		}, enc, probs)
		return seqfunc.AddAllR(masked)
	})
}
