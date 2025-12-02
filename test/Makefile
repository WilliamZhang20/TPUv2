# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src

TIMESTAMP = $(shell TZ=US/Eastern date +%Y%m%d_%H:%M:%S)
WAVES_ROOT = $(PWD)
COMPILE_ARGS += -DTIMESTAMP=\"$(TIMESTAMP)\"

PROJECT_SOURCES = tpu.v \
					systolic_array_2x2.v \
					control_unit.v \
					normalize_fp.v \
					PE.v \
					memory.v

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD				= sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD				= sim_build/gl
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DUSE_POWER_PINS
COMPILE_ARGS    += -DSIM
COMPILE_ARGS    += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v

# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

endif

# Allow sharing configuration between design and testbench via `include`:
COMPILE_ARGS 		+= -I$(SRC_DIR)

.PHONY: all test-mac test-top

all:
	$(MAKE) test-mac test-top

test-mac:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=tb \
		MODULE=test_mac \
		VERILOG_SOURCES="$(WAVES_ROOT)/mac/tb.v $(addprefix $(SRC_DIR)/,PE.v normalize_fp.v)" \
		PYTHONPATH=$(WAVES_ROOT)/mac \
		WAVES_DIR=$(WAVES_ROOT)/mac/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/mac/wave/mac_tb_$(TIMESTAMP).vcd\""'

test-top:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=tb \
		MODULE=test \
		VERILOG_SOURCES="$(WAVES_ROOT)/tpu/tb.v $(addprefix $(SRC_DIR)/,tpu.v systolic_array_2x2.v PE.v normalize_fp.v control_unit.v memory.v)" \
		PYTHONPATH=$(WAVES_ROOT)/tpu \
		WAVES_DIR=$(WAVES_ROOT)/tpu/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/tpu/wave/tpu_tb_$(TIMESTAMP).vcd\""'

# Phony target for cleaning up
.PHONY: clean
clean::
	rm -rf sim_build* results.xml

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
