// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	Input = 10 + 30 + 30 + 1
)

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	return sets
}

// Pixel is an image pixel
type Pixel struct {
	C uint8
	X int
	Y int
}

// Image is an image
type Image struct {
	W int
	H int
	I []Pixel
}

// Pair is an input output pair
type Pair struct {
	Class  int
	Input  Image
	Output Image
}

// Opt is an optimization
type Opt struct {
	Opt    matrix.Matrix
	Input  Pair
	Output Pair
}

// TargetOffset is the target offset
func (o Opt) TargetOffset() int {
	return len(o.Input.Input.I) + len(o.Input.Output.I) + len(o.Output.Input.I)
}

// TargetSize is the size of the target
func (o Opt) TargetSize() int {
	return len(o.Output.Output.I)
}

// GetTrainingData gets the training data
func GetTrainingData(sets []Set, s, t int) (opt []Opt, w [10]int) {
	train, test := make([]Pair, 0, 8), make([]Pair, 0, 8)
	set := sets[s]
	for _, t := range set.Train {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pair.Input.I = append(pair.Input.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output.I = append(pair.Output.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		train = append(train, pair)
	}
	for _, t := range set.Test {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pair.Input.I = append(pair.Input.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output.I = append(pair.Output.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		test = append(test, pair)
	}
	opt = make([]Opt, len(train))
	for i := range opt {
		opt[i].Input = train[i]
		opt[i].Output = test[t]
		opt[i].Opt = matrix.NewZeroMatrix(Input, opt[i].TargetOffset()+opt[i].TargetSize())
	}
	for i, pair := range train {
		index := 0
		for _, p := range pair.Input.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			index += Input
		}
		for _, p := range pair.Output.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			opt[i].Opt.Data[index+10+30+30] = 1
			index += Input
		}

		for _, p := range test[t].Input.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			index += Input
		}
	}
	return opt, w
}

// OptSingle is an optimization
type OptSingle struct {
	Count  int
	Opt    matrix.Matrix
	Input  Pair
	Output Pair
}

// TargetOffset is the target offset
func (o OptSingle) TargetOffset() int {
	return o.Count*(len(o.Input.Input.I)+len(o.Input.Output.I)) + len(o.Output.Input.I)
}

// TargetSize is the size of the target
func (o OptSingle) TargetSize() int {
	return len(o.Output.Output.I)
}

// GetSingleTrainingData gets the training data
func GetSingleTrainingData(sets []Set, s, t int) (opt []OptSingle, w [10]int) {
	train, test := make([]Pair, 0, 8), make([]Pair, 0, 8)
	set := sets[s]
	for _, t := range set.Train {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pair.Input.I = append(pair.Input.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output.I = append(pair.Output.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		train = append(train, pair)
	}
	for _, t := range set.Test {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pair.Input.I = append(pair.Input.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output.I = append(pair.Output.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		test = append(test, pair)
	}
	opt = make([]OptSingle, 1)
	opt[0].Count = len(train)
	for i := range opt {
		opt[i].Input = train[i]
		opt[i].Output = test[t]
		opt[i].Opt = matrix.NewZeroMatrix(Input, opt[i].TargetOffset()+opt[i].TargetSize())
	}
	index := 0
	for _, pair := range train {
		for _, p := range pair.Input.I {
			w[p.C]++
			opt[0].Opt.Data[index+int(p.C)] = 1
			opt[0].Opt.Data[index+10+p.X] = 1
			opt[0].Opt.Data[index+10+30+p.Y] = 1
			index += Input
		}
		for _, p := range pair.Output.I {
			w[p.C]++
			opt[0].Opt.Data[index+int(p.C)] = 1
			opt[0].Opt.Data[index+10+p.X] = 1
			opt[0].Opt.Data[index+10+30+p.Y] = 1
			opt[0].Opt.Data[index+10+30+30] = 1
			index += Input
		}
	}
	for _, p := range test[t].Input.I {
		w[p.C]++
		opt[0].Opt.Data[index+int(p.C)] = 1
		opt[0].Opt.Data[index+10+p.X] = 1
		opt[0].Opt.Data[index+10+30+p.Y] = 1
		index += Input
	}
	return opt, w
}

// Model model is the random matrix model
type Model struct {
	W1       matrix.RandomMatrix
	B1       matrix.RandomMatrix
	W2       matrix.RandomMatrix
	B2       matrix.RandomMatrix
	Query    matrix.RandomMatrix
	Key      matrix.RandomMatrix
	Value    matrix.RandomMatrix
	Solution matrix.RandomMatrix
}

// Generator is a generator
type Generator struct {
	W1       matrix.Generator
	B1       matrix.Generator
	W2       matrix.Generator
	B2       matrix.Generator
	Query    matrix.Generator
	Key      matrix.Generator
	Value    matrix.Generator
	Solution matrix.Generator
}

// Sample is a sample
type Sample struct {
	W1       matrix.Matrix
	B1       matrix.Matrix
	W2       matrix.Matrix
	B2       matrix.Matrix
	Query    matrix.Matrix
	Key      matrix.Matrix
	Value    matrix.Matrix
	Solution matrix.Matrix
	Cost     float64
	Grid     [][]int
}

// Stat is a statistic
type Stat struct {
	Count      float64
	Sum        float64
	SumSquared float64
}

func main() {
	rng := matrix.Rand(1)
	sets := Load()
	_ = sets
	//opts, ww := GetTrainingData(sets, 0, 0)
	opts, ww := GetSingleTrainingData(sets, 0, 0)
	inputSize := 10 * opts[0].Output.Input.W * opts[0].Output.Input.H
	outputSize := 10 * opts[0].Output.Output.W * opts[0].Output.Output.H
	model := Model{
		W1:       matrix.NewRandomMatrix(inputSize, 4*inputSize),
		B1:       matrix.NewRandomMatrix(4*inputSize, 1),
		W2:       matrix.NewRandomMatrix(4*inputSize, outputSize),
		B2:       matrix.NewRandomMatrix(outputSize, 1),
		Query:    matrix.NewRandomMatrix(Input, Input),
		Key:      matrix.NewRandomMatrix(Input, Input),
		Value:    matrix.NewRandomMatrix(Input, Input),
		Solution: matrix.NewRandomMatrix(10, opts[0].TargetSize()),
	}
	votes := make([][]int, opts[0].Output.Output.H*opts[0].Output.Output.W)
	stats := make([][]Stat, opts[0].Output.Output.H*opts[0].Output.Output.W)
	for v := range votes {
		votes[v] = make([]int, 10)
		stats[v] = make([]Stat, 10)
	}
	var auto, acc plotter.Values
	for i := 0; i < 4*1024; i++ {
		fmt.Println(i)
		samples := make([]Sample, 8)
		for i := range samples {
			generator := Generator{
				W1:       model.W1.Sample(&rng),
				B1:       model.B1.Sample(&rng),
				W2:       model.W2.Sample(&rng),
				B2:       model.B2.Sample(&rng),
				Query:    model.Query.Sample(&rng),
				Key:      model.Key.Sample(&rng),
				Value:    model.Value.Sample(&rng),
				Solution: model.Solution.Sample(&rng),
			}

			samples[i].W1 = generator.W1.Sample()
			samples[i].B1 = generator.B1.Sample()
			samples[i].W2 = generator.W2.Sample()
			samples[i].B2 = generator.B2.Sample()
			samples[i].Query = generator.Query.Sample()
			samples[i].Key = generator.Key.Sample()
			samples[i].Value = generator.Value.Sample()
			samples[i].Solution = generator.Solution.Sample()
		}

		done := make(chan bool, 8)
		process := func(sample *Sample) {
			opts, _ := GetTrainingData(sets, 0, 0)
			//opts, _ := GetSingleTrainingData(sets, 0, 0)
			sum := 0.0
			for _, opt := range opts {
				input := matrix.NewZeroMatrix(inputSize, 1)
				for i, value := range opt.Output.Input.I {
					input.Data[i*10+int(value.C)] = 1
				}
				//output := sample.W2.MulT(sample.W1.MulT(input).Add(sample.B1).Sigmoid()).Add(sample.B2).Sigmoid()
				params := opt.Opt.Data[Input*opt.TargetOffset():]
				for j := 0; j < sample.Solution.Rows; j++ {
					max, index := -math.MaxFloat64, 0
					for k := 0; k < sample.Solution.Cols; k++ {
						//value := float64(output.Data[j*10+k])
						value := float64(sample.Solution.Data[j*sample.Solution.Cols+k])
						if value > max {
							max, index = value, k
						}
					}
					params[j*Input+index] = 1
					params[j*Input+10+j%opt.Output.Output.W] = 1
					params[j*Input+10+30+j/opt.Output.Output.H] = 1
					params[j*Input+10+30+30] = 1
				}
				/*out := matrix.SelfAttention(
				sample.Query.MulT(opt.Opt),
				sample.Key.MulT(opt.Opt),
				sample.Value.MulT(opt.Opt))*/
				entropy := matrix.SelfEntropy(
					sample.Query.MulT(opt.Opt),
					sample.Key.MulT(opt.Opt),
					sample.Value.MulT(opt.Opt))
				for _, value := range entropy {
					sum += float64(value)
				}
				/*for j := 0; j < out.Rows; j++ {
					for k := 0; k < out.Cols; k++ {
						diff := out.Data[j*out.Cols+k] - opt.Opt.Data[j*out.Cols+k]
						sum += float64(diff*diff)
					}
				}*/
			}
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
		fmt.Println(samples[0].Cost)
		h, w := opts[0].Output.Output.H, opts[0].Output.Output.W
		grid := make([][]int, h)
		for j := range grid {
			grid[j] = make([]int, w)
		}
		input := matrix.NewZeroMatrix(inputSize, 1)
		for i, value := range opts[0].Output.Input.I {
			input.Data[i*10+int(value.C)] = 1
		}
		//output := samples[0].W2.MulT(samples[0].W1.MulT(input).Add(samples[0].B1).Sigmoid()).Add(samples[0].B2).Sigmoid()
		for j := 0; j < samples[0].Solution.Rows; j++ {
			max, index := -math.MaxFloat64, 0
			for k := 0; k < samples[0].Solution.Cols; k++ {
				//value := float64(output.Data[j*10+k])
				value := float64(samples[0].Solution.Data[j*samples[0].Solution.Cols+k])
				if value > max {
					max, index = value, k
				}
			}
			stats[j][index].Count++
			stats[j][index].Sum += max
			stats[j][index].SumSquared += max * max
			votes[j][index]++
			grid[j/h][j%w] = index
		}
		correct, count := 0.0, 0.0
		for i := 0; i < h; i++ {
			for j := 0; j < w; j++ {
				count++
				value := int(opts[0].Output.Output.I[i*w+j].C)
				if value == grid[i][j] {
					fmt.Printf("* ")
					correct++
					continue
				}
				fmt.Printf("%d ", grid[i][j])
			}
			fmt.Println()
		}
		fmt.Println(correct / count)
		auto = append(auto, samples[0].Cost)
		acc = append(acc, correct/count)

		//for i := 0; i < model.Solution.Rows; i++ {
		//	for j := 0; j < model.Solution.Cols; j++ {
		//avg := stats[i][j].Sum / stats[i][j].Count
		//model.Solution.Data[i*model.Solution.Cols+j].Mean = avg
		//model.Solution.Data[i*model.Solution.Cols+j].StdDev =
		//	math.Sqrt(stats[i][j].SumSquared/stats[i][j].Count - avg*avg)
		//		model.Solution.Data[i*model.Solution.Cols+j].StdDev = float64(votes[i][j])
		//	}
		//}
	}

	h, w := opts[0].Output.Output.H, opts[0].Output.Output.W
	grid := make([][]int, h)
	for j := range grid {
		grid[j] = make([]int, w)
	}
	counts := make([]int, 10)
	for i := range votes {
		max, index := 0, 0
		for j, value := range votes[i] {
			counts[j] += value
			if value > max {
				max, index = value, j
			}
		}
		grid[i/h][i%w] = index
		fmt.Printf("%d ", index)
		if (i+1)%w == 0 {
			fmt.Println()
		}
	}
	for i, count := range counts {
		if ww[i] == 0 {
			fmt.Printf("0 ")
			continue
		}
		fmt.Printf("%d ", count/ww[i])
	}
	fmt.Println()
	fmt.Println()
	correct, count := 0.0, 0.0
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			count++
			value := int(opts[0].Output.Output.I[i*w+j].C)
			if value == grid[i][j] {
				fmt.Printf("* ")
				correct++
				continue
			}
			fmt.Printf("%d ", grid[i][j])
		}
		fmt.Println()
	}
	fmt.Println()
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			value := int(opts[0].Output.Output.I[i*w+j].C)
			fmt.Printf("%d ", value)
		}
		fmt.Println()
	}
	for i := 0; i < model.Solution.Rows; i++ {
		for j := 0; j < model.Solution.Cols; j++ {
			avg := stats[i][j].Sum / stats[i][j].Count
			stddev := math.Sqrt(stats[i][j].SumSquared/stats[i][j].Count - avg*avg)
			fmt.Printf("%f ", avg/stddev)
		}
		fmt.Println()
	}
	fmt.Println(correct / count)

	p := plot.New()
	p.Title.Text = "acc histogram plot"

	hist, err := plotter.NewHist(acc, 10)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "acc_histogram.png")
	if err != nil {
		panic(err)
	}

	p = plot.New()
	p.Title.Text = "auto histogram plot"

	hist, err = plotter.NewHist(auto, 10)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "auto_histogram.png")
	if err != nil {
		panic(err)
	}

	x, y, xy, xx, yy := 0.0, 0.0, 0.0, 0.0, 0.0
	for i, X := range acc {
		Y := auto[i]
		x += X
		y += Y
		xy += X * Y
		xx += X * X
		yy += Y * Y
	}
	length := float64(len(acc))
	x /= length
	y /= length
	xy /= length
	xx /= length
	yy /= length
	corr := (xy - x*y) / (math.Sqrt(xx-x*x) * math.Sqrt(yy-y*y))
	fmt.Println("corr", corr)
}
