// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/mattie/original"
	"github.com/pointlander/mattie/random"
	"github.com/pointlander/mattie/text"
	"github.com/pointlander/mattie/text2"
)

var (
	// FlagOriginal original mode
	FlagOriginal = flag.Bool("original", false, "original mode")
	// RandomMode random mode
	FlagRandom = flag.Bool("random", false, "random mode")
	// TextMode is text mode
	FlagText = flag.Bool("text", false, "text mode")
	// Text2Mode is text 2 mode
	FlagText2 = flag.Bool("text2", false, "text 2 mode")
)

func main() {
	flag.Parse()

	if *FlagOriginal {
		original.Original()
		return
	} else if *FlagRandom {
		random.Random()
		return
	} else if *FlagText {
		text.Text()
		return
	} else if *FlagText2 {
		text2.Text2()
		return
	}
}
