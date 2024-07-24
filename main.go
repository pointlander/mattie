// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/mattie/original"
)

var (
	// FlagOriginal original mode
	FlagOriginal = flag.Bool("original", false, "original mode")
)

func main() {
	flag.Parse()

	if *FlagOriginal {
		original.Original()
		return
	}
}
