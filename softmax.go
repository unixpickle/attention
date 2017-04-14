package attention

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// seqSoftmax applies the softmax by treating each
// sequence as a vector with each timestep as a component.
func seqSoftmax(s anyseq.Seq) anyseq.Seq {
	return anyseq.Pool(s, func(s anyseq.Seq) anyseq.Seq {
		maxes := maxPerSeq(s)
		exps := subAndExp(s, maxes)
		return anyseq.Pool(exps, func(exps anyseq.Seq) anyseq.Seq {
			sum := anyseq.SumEach(exps)
			c := sum.Output().Creator()
			ones := c.MakeVector(sum.Output().Len())
			ones.AddScalar(c.MakeNumeric(1))
			scalers := anydiff.Div(anydiff.NewConst(ones), sum)
			return applyScalers(exps, scalers)
		})
	})
}

// maxPerSeq computes the maximum element for each
// sequence in the list.
func maxPerSeq(rawOuts anyseq.Seq) anyvec.Vector {
	maxBlock := &anyrnn.FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			elemMax := in.Output().Copy()
			anyvec.ElemMax(elemMax, state.Output())
			return nil, anydiff.NewConst(elemMax)
		},
		MakeStart: func(n int) anydiff.Res {
			c := rawOuts.Creator()
			outs := c.MakeVector(n)
			// TODO: look into using -inf here.
			outs.AddScalar(c.MakeNumeric(-10000))
			return anydiff.NewConst(outs)
		},
	}
	return anyseq.Tail(anyrnn.Map(rawOuts, maxBlock)).Output()
}

// subAndExp subtracts the values from maxPerSeq and then
// exponentiates.
func subAndExp(rawOuts anyseq.Seq, maxes anyvec.Vector) anyseq.Seq {
	expBlock := &anyrnn.FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			return anydiff.Exp(anydiff.Sub(in, state)), state
		},
		MakeStart: func(n int) anydiff.Res {
			if n != maxes.Len() {
				panic("bad state size")
			}
			return anydiff.NewConst(maxes)
		},
	}
	return anyrnn.Map(rawOuts, expBlock)
}

// applyScalers multiplies a different scaler to each
// sequence.
func applyScalers(s anyseq.Seq, scalers anydiff.Res) anyseq.Seq {
	scaleBlock := &anyrnn.FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			return anydiff.Mul(in, state), state
		},
		MakeStart: func(n int) anydiff.Res {
			if n != scalers.Output().Len() {
				panic("bad state size")
			}
			return scalers
		},
	}
	return anyrnn.Map(s, scaleBlock)
}
