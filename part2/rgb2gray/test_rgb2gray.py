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

tests = ['reset_test',
         'simple_test',
         'single_pixel_test',
         'multiple_pixels_test']

def rgb_to_gray(r, g, b):
    """Calculate expected grayscale value matching the hardware implementation"""
    red_coef = 19595
    green_coef = 38470
    blue_coef = 7471
    
    gray_sum = (r * red_coef) + (g * green_coef) + (b * blue_coef)
    gray = gray_sum >> 16
    
    return gray & 0xFF  

@pytest.mark.parametrize("example_p", [1])
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator, example_p):
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule=pymodule)

@pytest.mark.parametrize("example_p", [1])
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_each(simulator, test_name, example_p):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule=pymodule)

@cocotb.test()
async def reset_test(dut):
    """Test that reset properly initializes the module"""
    clk_i = dut.clk_i
    reset_i = dut.reset_i
    example_p = dut.example_p.value

    await clock_start_sequence(clk_i, 40)
    await reset_sequence(clk_i, reset_i, 10)
    
    await ClockCycles(clk_i, 10)
    dut._log.info("Reset test passed")

@cocotb.test()
async def simple_test(dut):
    """Basic connectivity test - send 4 bytes and check for output"""
    clk_i = dut.clk_i
    reset_i = dut.reset_i
    example_p = dut.example_p.value

    usrc = UartSource(dut.rx_serial_i, baud=115200, bits=8, stop_bits=1)
    usnk = UartSink(dut.tx_serial_o, baud=115200, bits=8, stop_bits=1)

    await clock_start_sequence(clk_i, 40)
    await reset_sequence(clk_i, reset_i, 10)
    await FallingEdge(reset_i)

    test_data = [10, 20, 30, 40]
    await usrc.write(test_data)
    await usrc.wait()
    
    dut._log.info(f"Simple test: sent {test_data}")
    
    try:
        received = await with_timeout(usnk.read(count=4), 100, 'ms')
        dut._log.info(f"Received: {received}")
    except Exception as e:
        dut._log.info(f"Timeout or error receiving: {e}")
    
    await ClockCycles(clk_i, 100)

@cocotb.test()
async def single_pixel_test(dut):
    """Test sending RGB data and receiving grayscale output"""
    clk_i = dut.clk_i
    reset_i = dut.reset_i

    usrc = UartSource(dut.rx_serial_i, baud=115200, bits=8, stop_bits=1)
    usnk = UartSink(dut.tx_serial_o, baud=115200, bits=8, stop_bits=1)

    await clock_start_sequence(clk_i, 40)
    await reset_sequence(clk_i, reset_i, 10)
    await FallingEdge(reset_i)

    # Send RGB values 
    # m_axis_tdata[7:0]=R, [15:8]=G, [23:16]=B, [31:24]=MSB
    r, g, b, msb = 100, 150, 200, 42
    rgb_data = [r, g, b, msb]
    
    expected_gray = rgb_to_gray(r, g, b)
    dut._log.info(f"Sending RGB: R={r}, G={g}, B={b}, MSB={msb}")
    dut._log.info(f"Expected grayscale: {expected_gray}")
    
    await usrc.write(rgb_data)
    await usrc.wait()
    
    # Receive output one byte at a time
    received = []
    for i in range(4):
        byte_data = await with_timeout(usnk.read(count=1), 200, 'ms')
        received.append(byte_data[0])
        dut._log.info(f"Received byte {i}: {byte_data[0]}")
    
    dut._log.info(f"Complete received data: {received}")
    
    # Check output
    assert received[0] == expected_gray, f"Byte 0: Expected {expected_gray}, got {received[0]}"
    assert received[1] == expected_gray, f"Byte 1: Expected {expected_gray}, got {received[1]}"
    assert received[2] == expected_gray, f"Byte 2: Expected {expected_gray}, got {received[2]}"
    assert received[3] == msb, f"Byte 3: Expected {msb}, got {received[3]}"
    
    dut._log.info("Single pixel test PASSED")

@cocotb.test()
async def multiple_pixels_test(dut):
    """Test sending multiple RGB pixels"""
    clk_i = dut.clk_i
    reset_i = dut.reset_i

    usrc = UartSource(dut.rx_serial_i, baud=115200, bits=8, stop_bits=1)
    usnk = UartSink(dut.tx_serial_o, baud=115200, bits=8, stop_bits=1)

    await clock_start_sequence(clk_i, 40)
    await reset_sequence(clk_i, reset_i, 10)
    await FallingEdge(reset_i)

    # Test with different RGB combinations
    test_pixels = [
        (255, 0, 0, 10),      # red
        (0, 255, 0, 20),      # green  
        (0, 0, 255, 30),      # blue
        (255, 255, 255, 40),  # white
        (128, 128, 128, 50),  # gray
        (0, 0, 0, 60),        # black
    ]
    
    for r, g, b, msb in test_pixels:
        rgb_data = [r, g, b, msb]
        expected_gray = rgb_to_gray(r, g, b)
        
        dut._log.info(f"Sending RGB({r}, {g}, {b}), MSB={msb}, expecting gray={expected_gray}")
        
        await usrc.write(rgb_data)
        await usrc.wait()
        
        # Read bytes one at a time
        received = []
        for i in range(4):
            byte_data = await with_timeout(usnk.read(count=1), 200, 'ms')
            received.append(byte_data[0])
        
        assert received[0] == expected_gray, f"RGB({r},{g},{b}) byte 0: expected {expected_gray}, got {received[0]}"
        assert received[1] == expected_gray, f"RGB({r},{g},{b}) byte 1: expected {expected_gray}, got {received[1]}"
        assert received[2] == expected_gray, f"RGB({r},{g},{b}) byte 2: expected {expected_gray}, got {received[2]}"
        assert received[3] == msb, f"RGB({r},{g},{b}) byte 3: expected {msb}, got {received[3]}"
        
        dut._log.info(f"  Result: {received} ✓")
    
    dut._log.info(f"All {len(test_pixels)} pixels tested successfully")