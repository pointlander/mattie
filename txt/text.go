// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txt

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
	"github.com/pointlander/matrix/vector"
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

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V matrix.Matrix) []float32 {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			/*if j < i {
				continue
			}*/
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = vector.Dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return results
}

// Set is a set of examples
type Set struct {
	Text []byte
}

// Sets is many sets
type Sets []Set

// Load load text data
func Load() Sets {
	sets := make(Sets, 5)
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

	sets[1].Text = []byte("abcdabcdabcda")
	sets[2].Text = []byte("abcdabcdabcdab")
	sets[3].Text = []byte("abcdabcdabcdabc")
	sets[4].Text = []byte("abcdabcdabcdabcdabcdabcd")
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
func (sets Sets) GetSingleTrainingData(tail []byte, s, t int) Problem {
	if s == 0 {
		set := sets[s]
		txt := make([]byte, len(set.Text[2048:2048+512]))
		copy(txt, set.Text[2048:2048+512])
		txt = append(txt, tail...)
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
	txt = append(txt, tail...)
	problem := Problem{
		Input:  txt[:len(txt)-1],
		Output: txt[len(txt)-1:],
		Count:  len(txt) + 1,
	}
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	index := 0
	for i := 0; i < len(txt)-1; i++ {
		problem.Opt.Data[index+To[txt[i]]] = 1
		/*if i-1 > 0 {
			problem.Opt.Data[index+Symbols+To[txt[(i-1)]]] = 1
		} else {
			problem.Opt.Data[index+Symbols+To[txt[(i)]]] = 1
		}*/
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

// Filtered is a filterd sample
type Filtered struct {
	Sample   *Sample
	Filtered complex128
}

// FilteredSet is a filtered sample set
type FilteredSet []Filtered

// Len is the length of the filtered set
func (f FilteredSet) Len() int {
	return len(f)
}

// Less determines if one member of the set is less than the other
func (f FilteredSet) Less(i, j int) bool {
	return cmplx.Abs(f[i].Filtered) < cmplx.Abs(f[j].Filtered)
}

// Swap swaps two set members
func (f FilteredSet) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

// Text mode
func Text(full bool, s int) int {
	os.Mkdir("output", 0755)
	sets := Load()
	const (
		SetSize    = 4
		SampleSets = 100
		Samples    = SampleSets * SetSize
	)
	type Result struct {
		Context int
		Symbol  int
		Score   float64
	}
	var search func(context int, seed uint32, suffix []byte, depth int, results chan Result)
	search = func(context int, seed uint32, suffix []byte, depth int, results chan Result) {
		depth--
		opt := sets.GetSingleTrainingData(suffix, s, 0)
		model := Model{
			Query: matrix.NewCompressedRandomMatrix(Input, Input),
			Key:   matrix.NewCompressedRandomMatrix(Input, Input),
			Value: matrix.NewCompressedRandomMatrix(Input, Input),
			Order: matrix.NewCompressedRandomMatrix(7, opt.Size()),
		}
		stats := make([]float64, SetSize)
		samples := make([]Sample, Samples)
		rng := matrix.Rand(seed)
		for i := 0; i < SampleSets; i++ {
			for j := 0; j < SetSize; j++ {
				query := model.Query.Sample(&rng)
				key := model.Key.Sample(&rng)
				value := model.Value.Sample(&rng)
				order := model.Order.Sample(&rng)
				samples[i*SetSize+j].Query = query
				samples[i*SetSize+j].Key = key
				samples[i*SetSize+j].Value = value
				samples[i*SetSize+j].Order = order
				samples[i*SetSize+j].S = j
			}
		}
		done := make(chan bool, 8)
		process := func(sample *Sample) {
			opt := sets.GetSingleTrainingData(suffix, s, 0)
			sum := 0.0
			order := sample.Order.Sample()
			a, b := 0, 1
			jj := opt.Opt.Rows //- 1
			for j := 0; j < jj; j++ {
				x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
				copy(opt.Opt.Data[j*Input+Symbols:j*Input+Symbols+7], order.Data[x*7:(x+1)*7])
				copy(opt.Opt.Data[j*Input+Symbols+7:j*Input+Symbols+2*7], order.Data[(y)*7:(y+1)*7])
				a, b = b, a
			}
			/*if x := jj + a; x < opt.Opt.Rows {
				copy(opt.Opt.Data[jj*Input+Symbols:jj*Input+Symbols+7], order.Data[x*7:(x+1)*7])
			}
			if y := jj + b; y < opt.Opt.Rows {
				copy(opt.Opt.Data[jj*Input+Symbols+7:jj*Input+Symbols+2*7], order.Data[(y)*7:(y+1)*7])
			}*/
			params := opt.Opt.Data[Input*(opt.Size()-1):]
			params[sample.S] = 1
			/*out := matrix.SelfAttention(
			sample.Query.MulT(opt.Opt),
			sample.Key.MulT(opt.Opt),
			sample.Value.MulT(opt.Opt))*/
			query := sample.Query.Sample()
			key := sample.Key.Sample()
			value := sample.Value.Sample()
			entropy := SelfEntropy(
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

		/*costs := make([]float64, len(samples))
		for i := range costs {
			costs[i] = samples[i].Cost
		}
		spectrum := fft.FFTReal(costs)
		cp := make([]complex128, len(spectrum))
		copy(cp, spectrum)
		filter := cp[:len(cp)/2]
		for i := range filter {
			filter[i] = 0
		}
		filtered := fft.IFFT(cp)
		fs := make(FilteredSet, len(filtered))
		for key, value := range filtered {
			fs[key].Sample = &samples[key]
			fs[key].Filtered = value
		}
		sort.Sort(fs)
		output, err := os.Create(fmt.Sprintf("output/spectrum_%d.txt", seed))
		if err != nil {
			panic(err)
		}
		defer output.Close()
		for _, value := range filtered {
			fmt.Fprintf(output, "%f\n", cmplx.Abs(value))
		}
		points := make(plotter.XYs, len(spectrum)-1)
		spectrum = spectrum[1:]
		for i := range spectrum {
			points[i] = plotter.XY{X: float64(i), Y: cmplx.Abs(spectrum[i])}
		}

		p := plot.New()
		p.Title.Text = "spectrum"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)
		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("output/spectrum_%d.png", seed))
		if err != nil {
			panic(err)
		}*/

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

		factor := [Symbols]float64{}
		for sample := range samples {
			index := samples[sample].S
			scale := samples[sample].Cost - factor[index]
			if scale < 0 {
				scale = -scale
			}
			factor[index] = samples[sample].Cost
			if scale == 0 {
				continue
			}
			stats[index] += 1 / float64(scale)
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
			for i := range stats[:4] {
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
			for i, stat := range stats[:4] {
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
	opt := sets.GetSingleTrainingData([]byte{}, s, 0)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	symbols := []byte{}
	results := make(chan Result, Symbols)
	set := make([]Result, 0, 8)
	histogram := make([]int, 4)
	for i := 1; i < 64; i++ {
		search(0, uint32(i), []byte{}, 1, results)
		result := <-results
		fmt.Printf("%d %c %f\n", i, From[result.Symbol], result.Score)
		symbols = []byte{byte(result.Symbol)}
		histogram[result.Symbol]++
		set = append(set, result)
	}
	sort.Slice(set, func(i, j int) bool {
		return set[i].Score < set[j].Score
	})
	fmt.Println("histogram", histogram)
	for _, value := range set {
		fmt.Printf("%c %f\n", From[value.Symbol], value.Score)
	}
	stats := make([]float64, SetSize)
	factor := [Symbols]float64{}
	for sample := range set {
		index := set[sample].Symbol
		scale := set[sample].Score - factor[index]
		if scale < 0 {
			scale = -scale
		}
		factor[index] = set[sample].Score
		if scale == 0 {
			continue
		}
		stats[index] += 1 / float64(scale)
	}
	fmt.Println(stats)
	if full {
		for i := 0; i < 100; i++ {
			search(0, uint32(i)+16, symbols, 1, results)
			result := <-results
			fmt.Printf("%c %f\n", From[result.Symbol], result.Score)
			symbols = append(symbols, byte(result.Symbol))
		}
	} else {
		max, sym := 0.0, 0
		for key, value := range stats {
			if value > max {
				max, sym = value, key
			}
		}
		return sym + 1
	}
	return 0
}
