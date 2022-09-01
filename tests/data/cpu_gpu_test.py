import unittest

# Hack to be able to import scripts in data knowing this isn't setup as a proper python package
import sys
sys.path.append('../..')

from data import cpu_gpu

class TestCPUTraceTimeConversion(unittest.TestCase):

    def test_process_cpu_data_PM(self):
        line = "04:34:52 PM  all    4.03    0.00    1.94    1.54    0.00    0.24    0.00    0.00    0.00   92.25"
        processed = cpu_gpu.split_cpu_trace_line_cols(line)
        self.assertEqual(processed[0], "16:34:52")

    def test_process_cpu_data_PM_limit(self):
        line = "11:59:59 PM  all    4.03    0.00    1.94    1.54    0.00    0.24    0.00    0.00    0.00   92.25"
        processed = cpu_gpu.split_cpu_trace_line_cols(line)
        self.assertEqual(processed[0], "23:59:59")

    def test_process_cpu_data_PM_next_day(self):
        line = "12:00:00 AM  all    4.03    0.00    1.94    1.54    0.00    0.24    0.00    0.00    0.00   92.25"
        processed = cpu_gpu.split_cpu_trace_line_cols(line)
        self.assertEqual(processed[0], "00:00:00")

    def test_process_cpu_data_noon(self):
        line = "12:00:00 PM  all    4.03    0.00    1.94    1.54    0.00    0.24    0.00    0.00    0.00   92.25"
        processed = cpu_gpu.split_cpu_trace_line_cols(line)
        self.assertEqual(processed[0], "12:00:00")


if __name__ == '__main__':
    unittest.main()