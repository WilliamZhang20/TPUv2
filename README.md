![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Tiny Tapeout Tensor Processing Unit Version 2

- After reading Elon Musk discuss the efficiency trade-offs between integer and floating-point computation for Tesla AI5/AI6 chips in an X thread, I thought of testing the idea myself using free ASIC tools on Tiny Tapeout.
- Since I already have a fully-fledged [TPU](https://github.com/WilliamZhang20/ECE298A-TPU) that does 8-bit integer (INT8) arithmetic we only just convert some logic to FP8 and make sure everything else works.
- With the previous project, I had tried doing FP8 with BF16 accumulation, but the chip area blew up. In retrospect, this was likely because I took extra space converting BF16 to FP8 for 8-bit ouptuts.
- Later on in the previous project, I got rid of 8-bit outputs and kept every output in 16-bit numbers. So with back-conversion waste eliminated, time to try it out again.

- [Read the documentation for project](docs/info.md)

## What is Tiny Tapeout?

Tiny Tapeout is a project that makes easier and cheaper to get digital and analog designs manufactured on a real chip.

To learn more, visit https://tinytapeout.com.

My Verilog code of the Application-Specific Integrated Circuit (ASIC) logic is defined in `/src`, and upon uploading to GitHub, each commit triggers a GitHub action.

The GitHub action will automatically build the ASIC files using [LibreLane](https://www.zerotoasiccourse.com/terminology/librelane/).

This action transforms the chip logic from software code to wired connections between transistors on the chip, allowing for designers to [view the chip](https://williamzhang20.github.io/TPUv2/) form when it is etched on the silicon die. 

Moreover, it enables logical and physical pre-silicon verification to ensure it works after tapeout. Due to ASIC indelibleness, the chip must work when taped out or else it is thrown away.