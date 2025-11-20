<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

The project is an AI chip inspired by Google's TPU. It multiply 8-bit floating-point valued matrices. It does so by tiling in 2x2 to fit on the chip's tiny area, so expect performance degradation compared to regular chips. However, the chip's I/O bandwidth will be fully utilized and saturated.

## How to test

Use cocotb and [pyuvm](https://github.com/pyuvm/pyuvm) to lean towards [IEEE-1800.2](https://blogs.sw.siemens.com/verificationhorizons/2015/07/30/uvm-the-next-ieee-standard-1800-2/).

## External hardware

Connect the PCB board with the Tiny Tapeout chips (a.k.a. a Raspberry Pi) to a personal computer via USB.