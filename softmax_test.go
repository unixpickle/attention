package attention

import (
	"testing"

	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestSeqSoftmax(t *testing.T) {
	c := anyvec32.DefaultCreator{}
	inSeq := anyseq.ConstSeqList(c, [][]anyvec.Vector{
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1})),
			c.MakeVectorData(c.MakeNumericList([]float64{-2})),
			c.MakeVectorData(c.MakeNumericList([]float64{2})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1})),
			c.MakeVectorData(c.MakeNumericList([]float64{0})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1000})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1001})),
			c.MakeVectorData(c.MakeNumericList([]float64{-1001})),
			c.MakeVectorData(c.MakeNumericList([]float64{1000.5})),
			c.MakeVectorData(c.MakeNumericList([]float64{100})),
		},
	})
	actual := seqSoftmax(inSeq)
	expected := anyseq.ConstSeqList(c, [][]anyvec.Vector{
		{
			c.MakeVectorData(c.MakeNumericList([]float64{0.2653879287722419})),
			c.MakeVectorData(c.MakeNumericList([]float64{0.0132128869537894})),
			c.MakeVectorData(c.MakeNumericList([]float64{0.7213991842739688})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{0.731058578630005})),
			c.MakeVectorData(c.MakeNumericList([]float64{0.268941421369995})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1})),
		},
		{
			c.MakeVectorData(c.MakeNumericList([]float64{0.622459331201854})),
			c.MakeVectorData(c.MakeNumericList([]float64{0})),
			c.MakeVectorData(c.MakeNumericList([]float64{0.377540668798145})),
			c.MakeVectorData(c.MakeNumericList([]float64{0})),
		},
	})
	if !anydifftest.SeqsClose(actual, expected, 1e-3) {
		t.Error("bad output")
	}
}
