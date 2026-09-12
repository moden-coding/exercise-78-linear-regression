#!/usr/bin/python3

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from src.linear_regression import fit_line, main


class TestFitLine(unittest.TestCase):

    def test_worked_example(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2.5, 3]) + 1
        slope, intercept = fit_line(x, y)
        self.assertIsInstance(
            slope,
            float,
            msg="fit_line's slope must be a plain float, got %r." % (type(slope),),
        )
        self.assertAlmostEqual(
            slope,
            1.0,
            places=4,
            msg="fit_line(%r, %r) should give a slope of about 1.0, got %r."
            % (x, y, slope),
        )
        self.assertAlmostEqual(
            intercept,
            1.16666666667,
            places=4,
            msg="fit_line(%r, %r) should give an intercept of about "
            "1.1667, got %r." % (x, y, intercept),
        )

    def test_calls(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2.5, 3]) + 1
        with patch("src.linear_regression.LinearRegression") as linreg:
            fit_line(x, y)
            linreg.assert_called_once()
            a, b = linreg().fit.call_args[0]
            shape = (len(x), 1)
            self.assertEqual(
                a.shape,
                shape,
                msg="fit's first argument (the x values) must have shape "
                "%r (a column vector), got %r." % (shape, a.shape),
            )
            self.assertEqual(
                len(b),
                len(y),
                msg="fit's second argument (the y values) should have "
                "length %r, got %r." % (len(y), len(b)),
            )


class TestMain(unittest.TestCase):

    def test_output(self):
        with patch("src.linear_regression.plt.show"):
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                main()
            output = out_buf.getvalue()
            self.assertRegex(
                output,
                r"Slope:\s+(.*)",
                msg="main() must print a line matching 'Slope: <value>', "
                "got:\n%s" % (output,),
            )
            self.assertRegex(
                output,
                r"Intercept:\s+(.*)",
                msg="main() must print a line matching 'Intercept: "
                "<value>', got:\n%s" % (output,),
            )

    def test_plot(self):
        with patch("src.linear_regression.plt.show") as pshow, patch(
            "src.linear_regression.plt.scatter"
        ) as pscatter, patch("src.linear_regression.plt.plot") as pplot:
            main()
            pshow.assert_called_once_with()
            self.assertEqual(
                pplot.call_count + pscatter.call_count,
                2,
                msg="main() should make exactly two plotting calls "
                "(plt.plot and/or plt.scatter), got %d plot call(s) and "
                "%d scatter call(s)." % (pplot.call_count, pscatter.call_count),
            )


if __name__ == '__main__':
    unittest.main()
