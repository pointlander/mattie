// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txt

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"

	"github.com/pointlander/matrix"
	"github.com/pointlander/matrix/vector"

	"github.com/alixaxel/pagerank"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Size is the link size
	Size = 4
	// Input is the network input size
	Input = Symbols + 2*Size
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// SetSize is the size of a symbol set
	SetSize = 4
	// SampleSets is the number of samples per set
	SampleSets = 100
	// Samples is the number of samplee
	Samples = SampleSets * SetSize
)

var (
	// To converts to a code
	To = make(map[byte]int, Symbols)
	// From converts from a code
	From = make(map[int]byte, Symbols)
)

func init() {
	index := 0
	for i := 'a'; i <= 'z'; i++ {
		To[byte(i)] = index
		From[index] = byte(i)
		index++
	}
	for i := 'A'; i <= 'Z'; i++ {
		To[byte(i)] = index
		From[index] = byte(i)
		index++
	}
	i := '.'
	To[byte(i)] = index
	From[index] = byte(i)
	index++
	i = ','
	To[byte(i)] = index
	From[index] = byte(i)
	index++
	i = ' '
	To[byte(i)] = index
	From[index] = byte(i)
	index++
}

// PageRank computes the page rank of Q, K
func PageRank(Q, K matrix.Matrix) float64 {
	graph := pagerank.NewGraph()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			d := float64(vector.Dot(K, Q))
			graph.Link(uint32(i), uint32(j), d*d)
		}
	}
	result := 0.0
	graph.Rank(0.85, 1e-6, func(node uint32, rank float64) {
		if node == uint32(K.Rows-1) {
			result = rank
		}
	})
	return result
}

// Set is a set of examples
type Set struct {
	Text []byte
}

// Sets is many sets
type Sets []Set

// Load load text data
func Load() Sets {
	sets := make(Sets, 6)
	input, err := os.Open("1513.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	decoded := bzip2.NewReader(input)
	data, err := io.ReadAll(decoded)
	if err != nil {
		panic(err)
	}
	sets[0].Text = data

	sets[1].Text = []byte("abcdabcdabcdabcda")
	sets[2].Text = []byte("abcdabcdabcdabcdab")
	sets[3].Text = []byte("abcdabcdabcdabcdabc")
	sets[4].Text = []byte("abcdabcdabcdabcdabcd")
	sets[5].Text = []byte("abcddcbaabcddcbaabcd")
	return sets
}

// Problem is an optimization problem
type Problem struct {
	Opt    matrix.Matrix
	Input  []byte
	Output []byte
	Count  int
}

// Size is the size of the input
func (p Problem) Size() int {
	return p.Count
}

// GetSingleTrainingData gets the training data
func (sets Sets) GetSingleTrainingData(s int) Problem {
	if s == 0 {
		set := sets[s]
		txt := make([]byte, len(set.Text[2048:2048+512]))
		copy(txt, set.Text[2048:2048+512])
		problem := Problem{
			Input:  txt,
			Output: txt[511 : 511+1],
			Count:  512 + 1,
		}
		problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
		index := 0
		for i := 0; i < len(txt); i++ {
			problem.Opt.Data[index+To[txt[i]]] = 1
			index += Input
		}
		return problem
	}
	set := sets[s]
	txt := make([]byte, len(set.Text))
	copy(txt, set.Text)
	problem := Problem{
		Input:  txt[:len(txt)-1],
		Output: txt[len(txt)-1:],
		Count:  len(txt) + 1,
	}
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	index := 0
	for i := 0; i < len(txt)-1; i++ {
		problem.Opt.Data[index+To[txt[i]]] = 1
		index += Input
	}
	return problem
}

// Model model is the random matrix model
type Model struct {
	Query matrix.CompressedRandomMatrix
	Key   matrix.CompressedRandomMatrix
	Order matrix.CompressedRandomMatrix
}

// Sample is a sample
type Sample struct {
	Query matrix.CompressedGenerator
	Key   matrix.CompressedGenerator
	Order matrix.CompressedGenerator
	S     int
	Cost  float64
}

// Search searches for a symbol
func Search(sets Sets, s int, seed uint32) []Sample {
	opt := sets.GetSingleTrainingData(s)
	model := Model{
		Query: matrix.NewCompressedRandomMatrix(Input, Input),
		Key:   matrix.NewCompressedRandomMatrix(Input, Input),
		Order: matrix.NewCompressedRandomMatrix(Size, opt.Size()),
	}
	samples := make([]Sample, Samples)
	rng := matrix.Rand(seed)
	for i := 0; i < SampleSets; i++ {
		for j := 0; j < SetSize; j++ {
			query := model.Query.Sample(&rng)
			key := model.Key.Sample(&rng)
			order := model.Order.Sample(&rng)
			samples[i*SetSize+j].Query = query
			samples[i*SetSize+j].Key = key
			samples[i*SetSize+j].Order = order
			samples[i*SetSize+j].S = j
		}
	}
	done := make(chan bool, 8)
	process := func(sample *Sample) {
		opt := sets.GetSingleTrainingData(s)
		order := sample.Order.Sample()
		a, b := 0, 1
		for j := 0; j < opt.Opt.Rows; j++ {
			x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
			copy(opt.Opt.Data[j*Input+Symbols:j*Input+Symbols+Size], order.Data[x*Size:(x+1)*Size])
			copy(opt.Opt.Data[j*Input+Symbols+Size:j*Input+Symbols+2*Size], order.Data[(y)*Size:(y+1)*Size])
			a, b = b, a
		}
		params := opt.Opt.Data[Input*(opt.Size()-1):]
		params[sample.S] = 1
		query := sample.Query.Sample()
		key := sample.Key.Sample()
		q := query.MulT(opt.Opt)
		k := key.MulT(opt.Opt)
		sample.Cost = PageRank(q, k)
		done <- true
	}
	flight, index, cpus := 0, 0, runtime.NumCPU()
	for flight < cpus && index < len(samples) {
		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for index < len(samples) {
		<-done
		flight--

		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for i := 0; i < flight; i++ {
		<-done
	}

	return samples
}

// Text mode
func Text(full bool, s int) int {
	sets := Load()
	opt := sets.GetSingleTrainingData(s)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	samples := []Sample{}
	for i := 1; i < 64; i++ {
		result := Search(sets, s, uint32(i))
		samples = append(samples, result...)
	}
	avg := [SetSize]float64{}
	count := [SetSize]float64{}
	for sample := range samples {
		index := samples[sample].S
		avg[index] += samples[sample].Cost
		count[index]++
	}
	for i := range avg {
		avg[i] /= count[i]
	}
	stddev := [SetSize]float64{}
	for sample := range samples {
		index := samples[sample].S
		diff := avg[index] - samples[sample].Cost
		stddev[index] += diff * diff
	}
	for i, v := range stddev {
		stddev[i] = math.Sqrt(v)
	}
	fmt.Println(avg)
	fmt.Println(stddev)
	metric := [SetSize]float64{}
	for i, v := range avg {
		metric[i] = v / stddev[i]
	}
	fmt.Println(metric)
	max, sym := 0.0, 0
	for key, value := range avg {
		value /= stddev[key]
		if value > max {
			max, sym = value, key
		}
	}
	return sym + 1
}
