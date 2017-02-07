package attention

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var c Combiner
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeCombiner)
}

// A Combiner is a feed-forward network that takes two
// vectors as inputs and produces one output vector
//
// For efficiency, the first layer is split up into two
// separate transformations.
// The vectors from these two transformations are added
// and then fed to OutTrans.
type Combiner struct {
	InTrans  [2]anynet.Layer
	OutTrans anynet.Layer
}

// DeserializeCombiner deserializes a Combiner.
func DeserializeCombiner(d []byte) (*Combiner, error) {
	var c Combiner
	err := serializer.DeserializeAny(d, &c.InTrans[0], &c.InTrans[1], &c.OutTrans)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Combiner", err)
	}
	return &c, nil
}

// NewCombiner creates a simple feed-forward combiner
// network with the given input sizes, hidden size, and
// output count.
func NewCombiner(c anyvec.Creator, in1, in2, hidden, out int) *Combiner {
	return &Combiner{
		InTrans: [2]anynet.Layer{
			anynet.NewFC(c, in1, hidden),
			anynet.NewFC(c, in2, hidden),
		},
		OutTrans: anynet.Net{
			anynet.Tanh,
			anynet.NewFC(c, hidden, out),
		},
	}
}

// Apply applies the Combiner.
func (c *Combiner) Apply(in1, in2 anydiff.Res, n int) anydiff.Res {
	return c.OutTrans.Apply(anydiff.Add(
		c.InTrans[0].Apply(in1, n),
		c.InTrans[1].Apply(in2, n),
	), n)
}

// Parameters returns the parameters of the Combiner's
// sub-components.
func (c *Combiner) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, l := range []anynet.Layer{c.InTrans[0], c.InTrans[1], c.OutTrans} {
		if p, ok := l.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Combiner with the serializer package.
func (c *Combiner) SerializerType() string {
	return "github.com/unixpickle/attention.Combiner"
}

// Serialize serializes a Combiner.
func (c *Combiner) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		c.InTrans[0],
		c.InTrans[1],
		c.OutTrans,
	)
}
