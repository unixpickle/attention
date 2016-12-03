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
	//
	// The first timestep of the Decoder is used solely to
	// produce a query for the next timestep; the output is
	// discarded from the final result.
	Decoder rnn.Block

	// BatchSize is the batch size for applying the Attentor.
	// If this is 0, inputs will not be batched.
	BatchSize int

	// QuerySize is the size of the vectors produced by the
	// Decoder which serve as queries to the Attentor.
	QuerySize int
}

// Apply applies the RNN to some input sequences.
//
// The in parameter specifies the inputs to be fed to the
// encoder.
//
// The decIn parameter specifies the additional inputs to
// be fed to the decoder at each decoding timestep.
// The decIn sequences should include the initial input to
// be fed into the decoder for the first timestep, even
// though the decoder's first output is thrown away.
//
// The returned sequences will be one timestep shorter than
// the corresponding sequences in decIn, since the first
// timestep is used only to generate an initial query.
func (s *SoftAlign) Apply(in seqfunc.Result, decIn seqfunc.Result) seqfunc.Result {
	encoded := s.Encoder.ApplySeqs(in)
	n := len(encoded.OutputSeqs())
	return seqfunc.Pool(encoded, func(encoded seqfunc.Result) seqfunc.Result {
		tempBlock := interfacerBlock{
			Resources: make([]autofunc.RFunc, n),
			Block:     s.Decoder,
			ResInSize: s.QuerySize,
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
		outSeqs := tempSeqFunc.ApplySeqs(decIn)
		return newSkipFirstResult(outSeqs)
	})
}

// ApplyR is like Apply, but with R-operator support.
func (s *SoftAlign) ApplyR(rv autofunc.RVector, in seqfunc.RResult,
	decIn seqfunc.RResult) seqfunc.RResult {
	encoded := s.Encoder.ApplySeqsR(rv, in)
	n := len(encoded.OutputSeqs())
	return seqfunc.PoolR(encoded, func(encoded seqfunc.RResult) seqfunc.RResult {
		tempBlock := interfacerBlock{
			Resources: make([]autofunc.RFunc, n),
			Block:     s.Decoder,
			ResInSize: s.QuerySize,
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
		outSeqs := tempSeqFunc.ApplySeqsR(rv, decIn)
		return newSkipFirstRResult(outSeqs)
	})
}

// TimeStepper generates a TimeStepper for the input
// sequence.
// The first timestep's output should not be used, since
// the purpose of the first timestep is to generate an
// initial attention query.
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
		Block:     s.Decoder,
		ResInSize: s.QuerySize,
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

type skipFirstResult struct {
	Res seqfunc.Result

	Skipped [][]linalg.Vector
}

func newSkipFirstResult(r seqfunc.Result) *skipFirstResult {
	outs := r.OutputSeqs()
	s := make([][]linalg.Vector, len(outs))
	for i, x := range outs {
		s[i] = x[1:]
	}
	return &skipFirstResult{Res: r, Skipped: s}
}

func (s *skipFirstResult) OutputSeqs() [][]linalg.Vector {
	return s.Skipped
}

func (s *skipFirstResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	newU := make([][]linalg.Vector, len(u))
	for i, x := range u {
		zeroVec := make(linalg.Vector, len(s.Res.OutputSeqs()[i][0]))
		newU[i] = append([]linalg.Vector{zeroVec}, x...)
	}
	s.Res.PropagateGradient(newU, g)
}

type skipFirstRResult struct {
	Res seqfunc.RResult

	Skipped  [][]linalg.Vector
	RSkipped [][]linalg.Vector
}

func newSkipFirstRResult(r seqfunc.RResult) *skipFirstRResult {
	outs := r.OutputSeqs()
	outsR := r.ROutputSeqs()
	s := make([][]linalg.Vector, len(outs))
	sR := make([][]linalg.Vector, len(outs))
	for i, x := range outs {
		s[i] = x[1:]
		sR[i] = outsR[i][1:]
	}
	return &skipFirstRResult{Res: r, Skipped: s, RSkipped: sR}
}

func (s *skipFirstRResult) OutputSeqs() [][]linalg.Vector {
	return s.Skipped
}

func (s *skipFirstRResult) ROutputSeqs() [][]linalg.Vector {
	return s.RSkipped
}

func (s *skipFirstRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	newU := make([][]linalg.Vector, len(u))
	newUR := make([][]linalg.Vector, len(u))
	for i, x := range u {
		zeroVec := make(linalg.Vector, len(s.Res.OutputSeqs()[i][0]))
		newU[i] = append([]linalg.Vector{zeroVec}, x...)
		newUR[i] = append([]linalg.Vector{zeroVec}, uR[i]...)
	}
	s.Res.PropagateRGradient(newU, newUR, rg, g)
}
