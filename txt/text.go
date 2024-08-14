// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txt

import (
	"compress/bzip2"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Input is the network input size
	Input = Symbols + 2*7
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
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

// Set is a set of examples
type Set struct {
	Text []byte
}

// Sets is many sets
type Sets []Set

// Load load text data
func Load() Sets {
	sets := make(Sets, 2)
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

	sets[1].Text = []byte("abcabcabcabc")
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
func (sets Sets) GetSingleTrainingData(tail, s, t int) Problem {
	if s == 0 {
		set := sets[s]
		problem := Problem{
			Input:  set.Text[2048 : 2048+512],
			Output: set.Text[2048+512 : 2048+512+1],
			Count:  512 + 1,
		}
		problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
		index := 0
		for i := 0; i < 512; i++ {
			problem.Opt.Data[index+To[set.Text[i+2048]]] = 1
			index += Input
		}
		return problem
	}
	set := sets[s]
	problem := Problem{
		Input:  set.Text[:len(set.Text)-1],
		Output: set.Text[len(set.Text)-1 : len(set.Text)],
		Count:  len(set.Text),
	}
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	index := 0
	for i := 0; i < len(set.Text)-1; i++ {
		problem.Opt.Data[index+To[set.Text[i]]] = 1
		index += Input
	}
	return problem

}

// Model model is the random matrix model
type Model struct {
	Query    matrix.CompressedRandomMatrix
	Key      matrix.CompressedRandomMatrix
	Value    matrix.CompressedRandomMatrix
	Solution matrix.CompressedRandomMatrix
	Order    matrix.CompressedRandomMatrix
}

// Sample is a sample
type Sample struct {
	Query matrix.CompressedGenerator
	Key   matrix.CompressedGenerator
	Value matrix.CompressedGenerator
	S     int
	Order matrix.CompressedGenerator
	Cost  float64
}

// Stat is a statistic
type Stat struct {
	Count      float64
	Sum        float64
	SumSquared float64
}

// Text mode
func Text() {
	sets := Load()
	const (
		SampleSets = 1000
		Samples    = SampleSets * 3
	)
	type Result struct {
		Context int
		Symbol  int
		Score   float64
	}
	var search func(context int, seed uint32, suffix []byte, depth int, results chan Result)
	search = func(context int, seed uint32, suffix []byte, depth int, results chan Result) {
		depth--
		opt := sets.GetSingleTrainingData(len(suffix), 1, 0)
		model := Model{
			Query: matrix.NewCompressedRandomMatrix(Input, Input),
			Key:   matrix.NewCompressedRandomMatrix(Input, Input),
			Value: matrix.NewCompressedRandomMatrix(Input, Input),
			Order: matrix.NewCompressedRandomMatrix(7, opt.Size()),
		}
		stats := make([]float64, Symbols)
		samples := make([]Sample, Samples)
		rng := matrix.Rand(seed)
		for i := 0; i < SampleSets; i++ {
			for j := 0; j < 3; j++ {
				query := model.Query.Sample(&rng)
				key := model.Key.Sample(&rng)
				value := model.Value.Sample(&rng)
				order := model.Order.Sample(&rng)
				samples[i*3+j].Query = query
				samples[i*3+j].Key = key
				samples[i*3+j].Value = value
				samples[i*3+j].Order = order
				samples[i*3+j].S = j
			}
		}
		done := make(chan bool, 8)
		process := func(sample *Sample) {
			opt := sets.GetSingleTrainingData(len(suffix), 1, 0)
			sum := 0.0
			order := sample.Order.Sample()
			a, b := 0, 1
			for j := 0; j < opt.Opt.Rows; j++ {
				x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
				copy(opt.Opt.Data[j*Input+Symbols:j*Input+Symbols+7], order.Data[x*7:(x+1)*7])
				copy(opt.Opt.Data[j*Input+Symbols+7:j*Input+Symbols+2*7], order.Data[(y)*7:(y+1)*7])
				a, b = b, a
			}
			index := 0
			for i := len(suffix) + 1; i > 1; i-- {
				opt.Opt.Data[Input*(opt.Size()-i)+To[byte(suffix[index])]] = 1
				index++
			}
			params := opt.Opt.Data[Input*(opt.Size()-1):]
			params[To[byte(sample.S)]] = 1
			/*out := matrix.SelfAttention(
			sample.Query.MulT(opt.Opt),
			sample.Key.MulT(opt.Opt),
			sample.Value.MulT(opt.Opt))*/
			query := sample.Query.Sample()
			key := sample.Key.Sample()
			value := sample.Value.Sample()
			entropy := matrix.SelfEntropy(
				query.MulT(opt.Opt),
				key.MulT(opt.Opt),
				value.MulT(opt.Opt))
			for _, value := range entropy {
				sum += float64(value)
			}
			/*for j := 0; j < out.Rows; j++ {
				for k := 0; k < out.Cols; k++ {
					diff := out.Data[j*out.Cols+k] - opt.Opt.Data[j*out.Cols+k]
					sum += float64(diff*diff)
				}
			}*/
			sample.Cost = sum
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

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})
		/*avg, vr := 0.0, 0.0
		for i := 0; i < len(samples); i++ {
			avg += samples[i].Cost
		}
		avg /= float64(len(samples))
		for i := 0; i < len(samples); i++ {
			diff := samples[i].Cost - avg
			vr += diff * diff
		}
		vr /= float64(len(samples))
		type Cut struct {
			Reduction float64
			Index     int
		}
		cuts := make(chan Cut, 8)
		mvr := func(i int) {
			avga, avgb := 0.0, 0.0
			vara, varb := 0.0, 0.0
			for j := 0; j < i; j++ {
				avga += samples[j].Cost
			}
			avga /= float64(i)
			for j := 0; j < i; j++ {
				diff := samples[j].Cost - avga
				vara += diff * diff
			}
			vara /= float64(i)
			for j := i; j < len(samples); j++ {
				avgb += samples[j].Cost
			}
			avgb /= float64(len(samples) - i)
			for j := i; j < len(samples); j++ {
				diff := samples[j].Cost - avgb
				varb += diff * diff
			}
			varb /= float64(len(samples) - i)
			reduction := vr - (vara + varb)
			cuts <- Cut{
				Reduction: reduction,
				Index:     i,
			}
		}
		maxReduction, cut := 0.0, 0
		flight, index, cpus = 0, 1, runtime.NumCPU()
		for flight < cpus && index < len(samples)-1 {
			go mvr(index)
			index++
			flight++
		}
		for index < len(samples)-1 {
			result := <-cuts
			if result.Reduction > maxReduction {
				maxReduction, cut = result.Reduction, result.Index
			}
			flight--

			go mvr(index)
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			result := <-cuts
			if result.Reduction > maxReduction {
				maxReduction, cut = result.Reduction, result.Index
			}
		}*/

		acc := [Symbols]int{}
		factor := [Symbols]int{}
		for sample := range samples {
			index := samples[sample].S
			acc[index]++
			scale := sample - factor[index]
			factor[index] = sample
			if scale == 0 {
				scale = 1
			}
			max, sym := 0.0, 0
			for key, value := range acc {
				if float64(value) > max {
					max, sym = float64(value), key
				}
			}
			stats[sym] += 1 / float64(scale)
		}
		/*vr = math.Sqrt(vr)
		for sample := range samples {
			x := samples[sample].Cost
			g := math.Exp(-(x-avg)*(x-avg)/(2*vr*vr)) / (vr * math.Sqrt(2*math.Pi))
			index := samples[sample].S
			stats[index] += 1 / g
		}*/
		fmt.Println(stats)

		max, index := 0.0, 0
		if depth > 0 {
			results := make(chan Result, Symbols)
			for i := range stats /*[:16]*/ {
				cp := make([]byte, len(suffix))
				copy(cp, suffix)
				s := append(cp, byte(i))
				seed := rng.Uint32() + 1
				if seed == 0 {
					seed = 1
				}
				go search(i, seed, s, depth, results)
			}
			count := 0
			for result := range results {
				if score := result.Score + stats[result.Context]; score > max {
					max, index = score, result.Context
				}
				count++
				if count == Symbols {
					break
				}
			}
		} else {
			for i, stat := range stats /*[:16]*/ {
				if stat > max {
					max, index = stat, i
				}
			}
		}
		results <- Result{
			Context: context,
			Symbol:  index,
			Score:   max,
		}
	}
	opt := sets.GetSingleTrainingData(0, 1, 0)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	results := make(chan Result, Symbols)
	search(0, 1, []byte{}, 1, results)
	result := <-results
	fmt.Printf("%c %f\n", From[result.Symbol], result.Score)
	symbols := []byte{byte(result.Symbol)}
	for i := 0; i < 100; i++ {
		search(0, uint32(i)+2, symbols, 1, results)
		result = <-results
		fmt.Printf("%c %f\n", From[result.Symbol], result.Score)
		symbols = append(symbols, byte(result.Symbol))
	}
}
