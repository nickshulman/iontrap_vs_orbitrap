import unittest
import calculate_loq as calc_loq

class TestCalculateLoq(unittest.TestCase):
    areas = [0.16751, 0.17056, 0.18132, 0.17482, 0.17060, 0.16879, 0.17469, 0.17645, 0.16372, 0.17310, 0.17941,
             0.16681, 0.18539, 0.17096, 0.14598, 0.15127, 0.17316, 0.15454, 0.18983, 0.17845, 0.17239, 0.18395,
             0.18260, 0.18245, 0.19410, 0.16476, 0.17444, 0.18214, 0.17844, 0.16537]
    concentrations = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07,
                      0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0]
    config = {'grid_size': 100, 'max_boot_iters': 100, 'min_boot_iters_for_measure': 10,
              'min_same_loq_count_for_accept': 25, 'cv_threshold': 0.2, 'debug_plot': False,
              'optimize_type': 1, 'minimum_num_transitions': 4}

    def test_fit_conc_vs_area(self):
        result = calc_loq.fit_conc_vs_area(self.concentrations, self.areas)
        slope = result['params'][0]
        intercept = result['params'][1]
        baseline_height = result['params'][2]
        self.assertAlmostEqual(0.025854009062249942, slope)
        self.assertAlmostEqual(0.16317916521766687, intercept)
        self.assertAlmostEqual(0.172315, baseline_height)
        self.assertAlmostEqual(0.057632885215053185, result['error'])
        self.assertAlmostEqual(0.0008070195509128315, result['baseline_std'])

    def test_bilinear_fit(self):
        unique_concentrations, mean_areas = calc_loq.get_unique_conc_and_mean_areas(self.concentrations, self.areas)
        result = calc_loq.bilinear_fit(0.03, unique_concentrations, mean_areas)
        # [0.00239897 0.16965172 0.17205111]
        self.assertAlmostEqual(0.00239897, result['params'][0])  # slope
        self.assertAlmostEqual(0.16965172, result['params'][1])  # intercept
        self.assertAlmostEqual(0.17205111, result['params'][2])  # baseline
        self.assertAlmostEqual(0.07240250761116633, result['error'])
        self.assertAlmostEqual(0.0007680004501027473, result['baseline_std'])

    def test_compute_quantitative_limits(self):
        concentrations = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07,
                          0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0]

        areas = [0.14685, 0.1646, 0.21232, 0.15662, 0.16293, 0.13719, 0.17409, 0.16193, 0.13625, 0.16065, 0.16249, 0.16409, 0.14434, 0.17133, 0.16018, 0.16517, 0.16808, 0.14533, 0.16407, 0.1659, 0.16019, 0.17952, 0.16168, 0.16217, 0.19297, 0.14971, 0.16969, 0.17356, 0.16252, 0.15779]

        result = calc_loq.compute_quantitative_limits(concentrations, areas, self.config)
        self.assertAlmostEqual(0.8052500937105422, result[calc_loq.optimize_type.LOD])
        self.assertAlmostEqual(1, result[calc_loq.optimize_type.LOQ])

    def test_compute_bootstrapped_loq(self):
        result = calc_loq.compute_quantitative_limits(self.concentrations, self.areas, self.config)

    def test_compute_lod(self):
        best_fit = calc_loq.fit_conc_vs_area(self.concentrations, self.areas)
        unique_concentrations, mean_areas = calc_loq.get_unique_conc_and_mean_areas(self.concentrations, self.areas)
        result = calc_loq.compute_lod(best_fit, unique_concentrations)
        self.assertAlmostEqual(0.3845768874492954, result)
