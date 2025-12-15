import re
import git
import os
import sys
import subprocess
import git

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
tbpath = os.path.dirname(os.path.realpath(__file__))
pymodule = "test_"+os.path.basename(tbpath)

import pytest

import cocotb

from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout, First
from cocotb.types import LogicArray, Range

from cocotb_test.simulator import run

from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiStreamSink, AxiStreamSource, AxiStreamBus
from cocotbext.uart import UartSource, UartSink

from pytest_utils.decorators import max_score, visibility, tags, leaderboard
   
import random
random.seed(42)

from functools import reduce

timescale = "1ps/1ps"
tests = ['simple_smoke_test']

@pytest.mark.parametrize("linewidth_px_p", [16])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_each(simulator, linewidth_px_p):
    parameters = dict(locals())
    del parameters['simulator']
    tbpath = os.path.dirname(os.path.realpath(__file__))
    runner(simulator, timescale, tbpath, parameters, testname='simple_smoke_test', pymodule='test_sobel')

@cocotb.test()
async def simple_smoke_test(dut):
    """Test sobel edge detection: send image with vertical edge and verify outputs."""
    clk = dut.clk_i
    reset = dut.reset_i
    linewidth_px_p = int(dut.linewidth_px_p.value)

    usrc = UartSource(dut.rx_serial_i, baud=115200, bits=8, stop_bits=1)
    usnk = UartSink(dut.tx_serial_o, baud=115200, bits=8, stop_bits=1)

    cocotb.start_soon(Clock(clk, 40, units='ns').start())

    reset.value = 1
    await ClockCycles(clk, 4)
    reset.value = 0
    await ClockCycles(clk, 2)

    # Create 5 rows with vertical edge (left=0, right=255)
    # Each pixel is sent as 4 bytes
    pattern = []
    for y in range(5):
        for x in range(linewidth_px_p):
            val = 0 if x < (linewidth_px_p // 2) else 255
            pattern.extend([val, val, val, 0])

    await usrc.write(pattern)
    await usrc.wait()

    await ClockCycles(clk, linewidth_px_p * 100)

    # Collect outputs
    out_bytes = []
    timeout = 0
    while timeout < 5000:
        if not usnk.empty():
            byte = await usnk.read(1)
            out_bytes.append(byte[0])
            timeout = 0
        else:
            await ClockCycles(clk, 10)
            timeout += 1

    assert len(out_bytes) > 0, "Expected edge detection output"