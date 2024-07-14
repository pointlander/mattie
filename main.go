// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/pointlander/matrix"
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
func GetTrainingData(sets []Set, s, t int) (opt []Opt) {
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
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			index += Input
		}
		for _, p := range pair.Output.I {
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			opt[i].Opt.Data[index+10+30+30] = 1
			index += Input
		}

		for _, p := range test[t].Input.I {
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+10+p.X] = 1
			opt[i].Opt.Data[index+10+30+p.Y] = 1
			index += Input
		}
	}
	return opt
}

func main() {
	sets := Load()
	_ = sets
}
