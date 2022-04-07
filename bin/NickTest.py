import unittest

import numpy as np

import calculate_loq as calc_loq
import numpy
from sklearn import linear_model

class TestCalculateLoq(unittest.TestCase):
    areas = [0.16751, 0.17056, 0.18132, 0.17482, 0.17060, 0.16879, 0.17469, 0.17645, 0.16372, 0.17310, 0.17941,
             0.16681, 0.18539, 0.17096, 0.14598, 0.15127, 0.17316, 0.15454, 0.18983, 0.17845, 0.17239, 0.18395,
             0.18260, 0.18245, 0.19410, 0.16476, 0.17444, 0.18214, 0.17844, 0.16537]
    concentrations = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07,
                      0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0]

    def test_fit_conc_vs_area(self):
        result = calc_loq.fit_conc_vs_area(self.concentrations, self.areas)
        slope = result['params'][0]
        intercept = result['params'][1]
        baseline_height = result['params'][2]
        self.assertAlmostEqual(18.60910749, slope)
        self.assertAlmostEqual(-3.00525332, intercept)
        self.assertAlmostEqual(0.17916667, baseline_height)
        self.assertAlmostEqual(94.99307214782094, result['error'])
        self.assertAlmostEqual(0.3079829089774662, result['baseline_std'])

    def test_bilinear_fit(self):
        unique_concentrations, mean_areas = calc_loq.get_unique_conc_and_mean_areas(self.concentrations, self.areas)
        result = calc_loq.bilinear_fit(0.03, unique_concentrations, mean_areas)
        self.assertAlmostEqual(0.01500812, result['params'][0])  # slope
        self.assertAlmostEqual(0.16795589, result['params'][1])  # intercept
        self.assertAlmostEqual(0.17205111, result['params'][2])  # baseline
        self.assertAlmostEqual(0.07240250761116633, result['error'])
        self.assertAlmostEqual(0.0007680004501027473, result['baseline_std'])

    def test_polynomial_fit(self):
        x = [0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0]
        y = [0.17310666666666666, 0.1674433333333333, 0.15965666666666667, 0.18022333333333332, 0.18300000000000002,
             0.17776666666666666, 0.17531666666666668]
        weights = [20.0, 14.285714285714285, 10.0, 3.3333333333333335, 2.0, 1.4285714285714286, 1.0]
        result1 = numpy.polyfit(x, y, 1, w=weights)
        result2 = numpy.polynomial.polynomial.Polynomial.fit(x, y, 1, w=weights)
        regression = linear_model.LinearRegression()
        x_values = [[el] for el in x]
        result3 = regression.fit(x_values, y, weights)
        self.assertAlmostEqual(0.16795589141065997, result2[0])

    def test_linear_regression(self):
        x = [0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0]
        y = [0.17310666666666666, 0.1674433333333333, 0.15965666666666667, 0.18022333333333332, 0.18300000000000002,
         0.17776666666666666, 0.17531666666666668]
        weights = [20.0, 14.285714285714285, 10.0, 3.3333333333333335, 2.0, 1.4285714285714286, 1.0]
        result = calc_loq.linear_regression(x, y, w=weights)
        result2 = np.polyfit(x, y, 1, w=weights)
        self.assertEqual(len(result), len(result2))