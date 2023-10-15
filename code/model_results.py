import numpy as np
from typing import List, Tuple


def results_improved_data_regression_nsplits5_seed42_progress() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_rmse = [
        np.array([0.2477, 0.251, 0.2607, 0.2412, 0.2446]),  # df_paper
        np.array([0.2488, 0.2528, 0.2607, 0.2445, 0.2455]),  # df_paper_with_speciation
        np.array([0.2512, 0.2539, 0.2519, 0.2562, 0.2551]),  # df_improved
        np.array([0.2538, 0.2701, 0.247, 0.2604, 0.2417]),  # df_improved_env
        np.array([0.2015, 0.2036, 0.201, 0.2031, 0.2035]),  # df_reg_improved_env_biowin
        np.array([0.1924, 0.1877, 0.1853, 0.1936, 0.174]),  # df_improved_env_biowin_both
        np.array([0.1907, 0.1864, 0.1911, 0.1957, 0.1854]),  # df_improved_env_biowin_both_readded
        # np.array([0.1784, 0.1708, 0.1746, 0.1698, 0.1752]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.1801, 0.1795, 0.1932, 0.1912, 0.1833]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.1884, 0.1759, 0.1898, 0.1726, 0.1782]),  # df_improved_env_biowin_both_ready
        # np.array([0.1839, 0.1805, 0.1786, 0.1892, 0.1826]),  # df_improved_env_biowin_both_28days
    ]
    all_data_mae = [
        np.array([0.1872, 0.1875, 0.1962, 0.176, 0.1817]),  # df_paper
        np.array([0.1877, 0.1885, 0.197, 0.1794, 0.1833]),  # df_paper_with_speciation
        np.array([0.1856, 0.1932, 0.1879, 0.1893, 0.1948]),  # df_improved
        np.array([0.1906, 0.2008, 0.1837, 0.1953, 0.1791]),  # df_improved_env
        np.array([0.1436, 0.1433, 0.1426, 0.1475, 0.1453]),  # df_reg_improved_env_biowin
        np.array([0.1327, 0.1335, 0.1325, 0.1374, 0.126]),  # df_improved_env_biowin_both
        np.array([0.1337, 0.1381, 0.136, 0.1385, 0.1352]),  # df_improved_env_biowin_both_readded
        # np.array([0.1246, 0.1187, 0.1227, 0.1204, 0.1224]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.1246, 0.1254, 0.1367, 0.1339, 0.1316]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.1315, 0.1245, 0.133, 0.1221, 0.123]),  # df_improved_env_biowin_both_ready
        # np.array([0.1291, 0.1287, 0.1249, 0.1342, 0.1296]),  # df_improved_env_biowin_both_28days
    ]
    all_data_r2 = [
        np.array([0.5214, 0.5016, 0.4788, 0.5405, 0.5373]),  # df_paper
        np.array([0.517, 0.4943, 0.479, 0.5279, 0.5338]),  # df_paper_with_speciation
        np.array([0.52, 0.5174, 0.4867, 0.49, 0.5055]),  # df_improved
        np.array([0.5044, 0.4469, 0.5313, 0.4746, 0.5601]),  # df_improved_env
        np.array([0.7062, 0.7025, 0.7058, 0.7053, 0.7028]),  # df_reg_improved_env_biowin
        np.array([0.7285, 0.7498, 0.7555, 0.7465, 0.781]),  # df_improved_env_biowin_both
        np.array([0.7377, 0.7402, 0.7313, 0.7234, 0.7582]),  # df_improved_env_biowin_both_readded
        # np.array([0.7434, 0.7789, 0.7705, 0.772, 0.7714]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.7577, 0.7493, 0.7305, 0.7221, 0.7452]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.7454, 0.77, 0.73, 0.7859, 0.7706]),  # df_improved_env_biowin_both_ready
        # np.array([0.7455, 0.7712, 0.7506, 0.743, 0.757]),  # df_improved_env_biowin_both_28days
    ]
    all_data_mse = [
        np.array([0.0613, 0.063, 0.068, 0.0582, 0.0598]),  # df_paper
        np.array([0.0619, 0.0639, 0.0679, 0.0598, 0.0603]),  # df_paper_with_speciation
        np.array([0.0631, 0.0644, 0.0635, 0.0656, 0.0651]),  # df_improved
        np.array([0.0644, 0.0729, 0.061, 0.0678, 0.0584]),  # df_improved_env
        np.array([0.0406, 0.0414, 0.0404, 0.0413, 0.0414]),  # df_reg_improved_env_biowin
        np.array([0.037, 0.0352, 0.0343, 0.0375, 0.0303]),  # df_improved_env_biowin_both
        np.array([0.0364, 0.0348, 0.0365, 0.0383, 0.0344]),  # df_improved_env_biowin_both_readded
        # np.array([0.0318, 0.0292, 0.0305, 0.0288, 0.0307]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.0324, 0.0322, 0.0373, 0.0366, 0.0336]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.0355, 0.0309, 0.036, 0.0298, 0.0317]),  # df_improved_env_biowin_both_ready
        # np.array([0.0338, 0.0326, 0.0319, 0.0358, 0.0333]),  # df_improved_env_biowin_both_28days
    ]
    return all_data_rmse, all_data_mae, all_data_r2, all_data_mse


def results_improved_data_regression_nsplits5_seed42_comparison() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_rmse = [
        np.array([0.2477, 0.251, 0.2607, 0.2412, 0.2446]),  # df_paper
        np.array([0.2512, 0.2539, 0.2519, 0.2562, 0.2551]),  # df_improved
        np.array([0.2538, 0.2701, 0.247, 0.2604, 0.2417]),  # df_improved_env
        np.array([0.2079, 0.1995, 0.2017, 0.2048, 0.1963]),  # df_biowin
        np.array([0.2155, 0.2141, 0.1912, 0.2035, 0.1983]),  # df_improved_biowin
        np.array([0.2015, 0.2036, 0.201, 0.2031, 0.2035]),  # df_improved_env_biowin
        np.array([0.189, 0.1889, 0.1838, 0.1819, 0.1982]),  # df_biowin_both
        np.array([0.1866, 0.1834, 0.1964, 0.1889, 0.1927]),  # df_improved_biowin_both
        np.array([0.1924, 0.1877, 0.1853, 0.1936, 0.174]),  # df_improved_env_biowin_both
    ]
    all_data_mae = [
        np.array([0.1872, 0.1875, 0.1962, 0.176, 0.1817]),  # df_paper
        np.array([0.1856, 0.1932, 0.1879, 0.1893, 0.1948]),  # df_improved
        np.array([0.1906, 0.2008, 0.1837, 0.1953, 0.1791]),  # df_improved_env
        np.array([0.1504, 0.143, 0.1399, 0.1467, 0.1423]),  # df_biowin
        np.array([0.1545, 0.1509, 0.1374, 0.1473, 0.1419]),  # df_improved_biowin
        np.array([0.1436, 0.1433, 0.1426, 0.1475, 0.1453]),  # df_improved_env_biowin
        np.array([0.1349, 0.1339, 0.1297, 0.1271, 0.142]),  # df_biowin_both
        np.array([0.1338, 0.1302, 0.1396, 0.1356, 0.1368]),  # df_improved_biowin_both
        np.array([0.1327, 0.1335, 0.1325, 0.1374, 0.126]),  # df_improved_env_biowin_both
    ]
    all_data_r2 = [
        np.array([0.5214, 0.5016, 0.4788, 0.5405, 0.5373]),  # df_paper
        np.array([0.52, 0.5174, 0.4867, 0.49, 0.5055]),  # df_improved
        np.array([0.5044, 0.4469, 0.5313, 0.4746, 0.5601]),  # df_improved_env
        np.array([0.6781, 0.6923, 0.7071, 0.6903, 0.7195]),  # df_biowin
        np.array([0.6582, 0.6698, 0.733, 0.6926, 0.721]),  # df_improved_biowin
        np.array([0.7062, 0.7025, 0.7058, 0.7053, 0.7028]),  # df_improved_env_biowin
        np.array([0.7343, 0.7369, 0.7521, 0.7634, 0.7133]),  # df_biowin_both
        np.array([0.7477, 0.761, 0.7172, 0.7527, 0.7293]),  # df_improved_biowin_both
        np.array([0.7285, 0.7498, 0.7555, 0.7465, 0.781]),  # df_improved_env_biowin_both
    ]
    all_data_mse = [
        np.array([0.0613, 0.063, 0.068, 0.0582, 0.0598]),  # df_paper
        np.array([0.0631, 0.0644, 0.0635, 0.0656, 0.0651]),  # df_improved
        np.array([0.0644, 0.0729, 0.061, 0.0678, 0.0584]),  # df_improved_env
        np.array([0.0432, 0.0398, 0.0407, 0.042, 0.0385]),  # df_biowin
        np.array([0.0464, 0.0458, 0.0366, 0.0414, 0.0393]),  # df_improved_biowin
        np.array([0.0406, 0.0414, 0.0404, 0.0413, 0.0414]),  # df_improved_env_biowin
        np.array([0.0357, 0.0357, 0.0338, 0.0331, 0.0393]),  # df_biowin_both
        np.array([0.0348, 0.0336, 0.0386, 0.0357, 0.0371]),  # df_improved_biowin_both
        np.array([0.037, 0.0352, 0.0343, 0.0375, 0.0303]),  # df_improved_env_biowin_both
    ]
    return all_data_rmse, all_data_mae, all_data_r2, all_data_mse


def results_improved_data_regression_nsplits5_seed42_fixed_test_progress() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_rmse = [
        np.array([0.2101, 0.2108, 0.2172, 0.2193, 0.2184]),  # df_paper
        np.array([0.2166, 0.2134, 0.2225, 0.2244, 0.2221]),  # df_paper_with_speciation
        np.array([0.2168, 0.2244, 0.2278, 0.2314, 0.2272]),  # df_improved
        np.array([0.2156, 0.2178, 0.2293, 0.2317, 0.2252]),  # df_improved_env
        np.array([0.1951, 0.1968, 0.1982, 0.2049, 0.1918]),  # df_improved_env_biowin
        np.array([0.1993, 0.2003, 0.1899, 0.2057, 0.1891]),  # df_improved_env_biowin_both
        np.array([0.1887, 0.1864, 0.1906, 0.1978, 0.1857]),  # df_improved_env_biowin_both_readded
        # np.array([0.2128, 0.2083, 0.1975, 0.2216, 0.204]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.2017, 0.2019, 0.1918, 0.21, 0.1969]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.2045, 0.2018, 0.1917, 0.2092, 0.1969]),  # df_improved_env_biowin_both_ready
        # np.array([0.2131, 0.2074, 0.1999, 0.2144, 0.2016]),  # df_improved_env_biowin_both_28days
    ]
    all_data_mae = [
        np.array([0.1539, 0.1571, 0.1601, 0.1579, 0.1622]),  # df_paper
        np.array([0.1616, 0.1637, 0.1653, 0.1655, 0.168]),  # df_paper_with_speciation
        np.array([0.1634, 0.1697, 0.1691, 0.1738, 0.1704]),  # df_improved
        np.array([0.161, 0.1675, 0.1714, 0.1718, 0.1683]),  # df_improved_env
        np.array([0.1384, 0.1459, 0.1414, 0.1475, 0.1407]),  # df_improved_env_biowin
        np.array([0.1394, 0.1467, 0.1376, 0.1473, 0.139]),  # df_improved_env_biowin_both
        np.array([0.1324, 0.1381, 0.1351, 0.1403, 0.1346]),  # df_improved_env_biowin_both_readded
        # np.array([0.1504, 0.1498, 0.1462, 0.1552, 0.149]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.1465, 0.1487, 0.1405, 0.1495, 0.1446]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.1444, 0.1469, 0.1393, 0.1494, 0.1427]),  # df_improved_env_biowin_both_ready
        # np.array([0.1507, 0.1531, 0.1427, 0.1515, 0.1476]),  # df_improved_env_biowin_both_28days
    ]
    all_data_r2 = [
        np.array([0.6818, 0.6679, 0.653, 0.6529, 0.6643]),  # df_paper
        np.array([0.6616, 0.6596, 0.6359, 0.6363, 0.6529]),  # df_paper_with_speciation
        np.array([0.661, 0.6237, 0.6181, 0.6133, 0.6368]),  # df_improved
        np.array([0.6648, 0.6454, 0.6133, 0.6124, 0.6432]),  # df_improved_env
        np.array([0.7256, 0.7105, 0.7109, 0.6969, 0.7412]),  # df_improved_env_biowin
        np.array([0.7135, 0.7, 0.7348, 0.6947, 0.7484]),  # df_improved_env_biowin_both
        np.array([0.7432, 0.7402, 0.7328, 0.7176, 0.7574]),  # df_improved_env_biowin_both_readded
        # np.array([0.6736, 0.6758, 0.713, 0.6456, 0.7073]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.7065, 0.6954, 0.7294, 0.6816, 0.7271]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.6984, 0.6956, 0.7297, 0.6841, 0.7271]),  # df_improved_env_biowin_both_ready
        # np.array([0.6725, 0.6785, 0.706, 0.6681, 0.714]),  # df_improved_env_biowin_both_28days
    ]
    all_data_mse = [
        np.array([0.0441, 0.0444, 0.0472, 0.0481, 0.0477]),  # df_paper
        np.array([0.0469, 0.0455, 0.0495, 0.0504, 0.0493]),  # df_paper_with_speciation
        np.array([0.047, 0.0503, 0.0519, 0.0536, 0.0516]),  # df_improved
        np.array([0.0465, 0.0474, 0.0526, 0.0537, 0.0507]),  # df_improved_env
        np.array([0.0381, 0.0387, 0.0393, 0.042, 0.0368]),  # df_improved_env_biowin
        np.array([0.0397, 0.0401, 0.036, 0.0423, 0.0358]),  # df_improved_env_biowin_both
        np.array([0.0356, 0.0348, 0.0363, 0.0391, 0.0345]),  # df_improved_env_biowin_both_readded
        # np.array([0.0453, 0.0434, 0.039, 0.0491, 0.0416]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.0407, 0.0407, 0.0368, 0.0441, 0.0388]),  # df_improved_env_biowin_both_reliability1
        # np.array([0.0418, 0.0407, 0.0367, 0.0438, 0.0388]),  # df_improved_env_biowin_both_ready
        # np.array([0.0454, 0.043, 0.04, 0.046, 0.0406]),  # df_improved_env_biowin_both_28days
    ]
    return all_data_rmse, all_data_mae, all_data_r2, all_data_mse


def results_improved_data_regression_nsplits5_seed42_fixed_test_comparison() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_rmse = [
        np.array([0.2051, 0.2147, 0.2087, 0.2239, 0.2094]),  # df_paper
        np.array([0.2151, 0.2231, 0.2178, 0.2407, 0.2189]),  # df_improved
        np.array([0.2164, 0.2229, 0.219, 0.2396, 0.2162]),  # df_improved_env
        np.array([0.1874, 0.1809, 0.1835, 0.1936, 0.1733]),  # df_biowin
        np.array([0.1948, 0.1868, 0.188, 0.199, 0.1823]),  # df_improved_biowin
        np.array([0.1904, 0.1875, 0.1887, 0.1941, 0.1758]),  # df_improved_env_biowin
        np.array([0.1892, 0.1812, 0.1831, 0.1928, 0.1733]),  # df_biowin_both
        np.array([0.1983, 0.1884, 0.1943, 0.2009, 0.184]),  # df_improved_biowin_both
        np.array([0.1924, 0.1898, 0.1853, 0.1919, 0.174]),  # df_improved_env_biowin_both
    ]
    all_data_mae = [
        np.array([0.1468, 0.1604, 0.1535, 0.1638, 0.1542]),  # df_paper
        np.array([0.1575, 0.1655, 0.1635, 0.1798, 0.1643]),  # df_improved
        np.array([0.1596, 0.1653, 0.1661, 0.1787, 0.1612]),  # df_improved_env
        np.array([0.1269, 0.1268, 0.1257, 0.1336, 0.1216]),  # df_biowin
        np.array([0.1363, 0.1322, 0.1346, 0.143, 0.1307]),  # df_improved_biowin
        np.array([0.1331, 0.1328, 0.1345, 0.1383, 0.1277]),  # df_improved_env_biowin
        np.array([0.1299, 0.1274, 0.1263, 0.1323, 0.1224]),  # df_biowin_both
        np.array([0.1389, 0.1319, 0.1371, 0.1432, 0.1316]),  # df_improved_biowin_both
        np.array([0.1327, 0.1342, 0.1325, 0.1357, 0.126]),  # df_improved_env_biowin_both
    ]
    all_data_r2 = [
        np.array([0.6916, 0.6727, 0.6897, 0.6606, 0.6828]),  # df_paper
        np.array([0.6607, 0.6468, 0.662, 0.6079, 0.6532]),  # df_improved
        np.array([0.6565, 0.6475, 0.6582, 0.6116, 0.6618]),  # df_improved_env
        np.array([0.7424, 0.7678, 0.7602, 0.7463, 0.7826]),  # df_biowin
        np.array([0.7216, 0.7523, 0.7482, 0.732, 0.7595]),  # df_improved_biowin
        np.array([0.7342, 0.7504, 0.7463, 0.7451, 0.7763]),  # df_improved_env_biowin
        np.array([0.7376, 0.7669, 0.7611, 0.7485, 0.7828]),  # df_biowin_both
        np.array([0.7117, 0.7481, 0.7311, 0.7268, 0.755]),  # df_improved_biowin_both
        np.array([0.7285, 0.7442, 0.7555, 0.7507, 0.781]),  # df_improved_env_biowin_both
    ]
    all_data_mse = [
        np.array([0.0421, 0.0461, 0.0436, 0.0502, 0.0438]),  # df_paper
        np.array([0.0463, 0.0498, 0.0474, 0.0579, 0.0479]),  # df_improved
        np.array([0.0468, 0.0497, 0.048, 0.0574, 0.0467]),  # df_improved_env
        np.array([0.0351, 0.0327, 0.0337, 0.0375, 0.03]),  # df_biowin
        np.array([0.038, 0.0349, 0.0353, 0.0396, 0.0332]),  # df_improved_biowin
        np.array([0.0363, 0.0352, 0.0356, 0.0377, 0.0309]),  # df_improved_env_biowin
        np.array([0.0358, 0.0328, 0.0335, 0.0372, 0.03]),  # df_biowin_both
        np.array([0.0393, 0.0355, 0.0377, 0.0404, 0.0339]),  # df_improved_biowin_both
        np.array([0.037, 0.036, 0.0343, 0.0368, 0.0303]),  # df_improved_env_biowin_both
    ]
    return all_data_rmse, all_data_mae, all_data_r2, all_data_mse


def results_improved_data_classification_nsplits5_seed42_progress() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_accuracy = [
        np.array([0.8208, 0.8192, 0.8176, 0.8078, 0.8199]),  # df_paper
        np.array([0.8094, 0.8225, 0.8274, 0.7883, 0.828]),  # df_paper_with_speciation
        np.array([0.8217, 0.8199, 0.836, 0.819, 0.8224]),  # df_improved
        np.array([0.8187, 0.8363, 0.8224, 0.8296, 0.8241]),  # df_improved_env
        np.array([0.9531, 0.9411, 0.9315, 0.935, 0.9254]),  # df_improved_env_biowin
        np.array([0.9566, 0.9592, 0.946, 0.946, 0.9552]),  # df_improved_env_biowin_both
        np.array([0.9569, 0.9639, 0.9569, 0.9522, 0.9463]),  # df_improved_env_biowin_both_readded
        # np.array([0.9506, 0.9547, 0.952, 0.9506, 0.9547]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9409, 0.9662, 0.9423, 0.9451, 0.9494]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_sensitivity = [
        np.array([0.7413, 0.7343, 0.7599, 0.7459, 0.715]),  # df_paper
        np.array([0.7203, 0.7413, 0.7762, 0.7343, 0.7336]),  # df_paper_with_speciation
        np.array([0.7404, 0.7538, 0.741, 0.759, 0.7198]),  # df_improved
        np.array([0.7414, 0.7573, 0.7599, 0.7381, 0.7751]),  # df_improved_env
        np.array([0.9288, 0.8876, 0.8843, 0.9176, 0.8539]),  # df_improved_env_biowin
        np.array([0.9259, 0.9421, 0.9256, 0.8926, 0.9383]),  # df_improved_env_biowin_both
        np.array([0.9267, 0.9524, 0.9194, 0.9228, 0.9231]),  # df_improved_env_biowin_both_readded
        # np.array([0.9068, 0.9237, 0.9198, 0.9494, 0.9407]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9258, 0.9389, 0.913, 0.8957, 0.9174]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_specificity = [
        np.array([0.8636, 0.8648, 0.8486, 0.8411, 0.8761]),  # df_paper
        np.array([0.8573, 0.8661, 0.8548, 0.8173, 0.8786]),  # df_paper_with_speciation
        np.array([0.8652, 0.8554, 0.8871, 0.8512, 0.8774]),  # df_improved
        np.array([0.8604, 0.8789, 0.8561, 0.8789, 0.8504]),  # df_improved_env
        np.array([0.9646, 0.9664, 0.9539, 0.9433, 0.9592]),  # df_improved_env_biowin
        np.array([0.971, 0.9671, 0.9555, 0.971, 0.9632]),  # df_improved_env_biowin_both
        np.array([0.9709, 0.9692, 0.9744, 0.9658, 0.9572]),  # df_improved_env_biowin_both_readded
        # np.array([0.9716, 0.9696, 0.9675, 0.9512, 0.9614]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9481, 0.9793, 0.9563, 0.9688, 0.9647]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_f1 = [
        np.array([0.743, 0.7394, 0.7443, 0.7306, 0.7347]),  # df_paper
        np.array([0.7254, 0.7447, 0.7585, 0.7079, 0.7485]),  # df_paper_with_speciation
        np.array([0.7432, 0.7452, 0.7595, 0.7456, 0.7388]),  # df_improved
        np.array([0.7414, 0.7643, 0.75, 0.752, 0.7552]),  # df_improved_env
        np.array([0.9271, 0.9063, 0.8927, 0.9007, 0.8803]),  # df_improved_env_biowin
        np.array([0.9317, 0.9363, 0.9162, 0.9133, 0.9306]),  # df_improved_env_biowin_both
        np.array([0.9319, 0.9437, 0.9314, 0.9245, 0.9164]),  # df_improved_env_biowin_both_readded
        # np.array([0.9224, 0.9296, 0.9257, 0.9259, 0.9308]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9099, 0.9471, 0.9111, 0.9135, 0.9214]),  # df_improved_env_biowin_both_reliability1
    ]
    return all_data_accuracy, all_data_f1, all_data_sensitivity, all_data_specificity


def results_improved_data_classification_nsplits5_seed42_comparison() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_accuracy = [
        np.array([0.8208, 0.8192, 0.8176, 0.8078, 0.8199]),  # df_paper
        np.array([0.8217, 0.8199, 0.836, 0.819, 0.8224]),  # df_improved
        np.array([0.8187, 0.8363, 0.8224, 0.8296, 0.8241]),  # df_improved_env
        np.array([0.9403, 0.9372, 0.9166, 0.9248, 0.9372]),  # df_biowin
        np.array([0.9337, 0.9414, 0.9238, 0.9381, 0.9403]),  # df_improved_biowin
        np.array([0.9531, 0.9411, 0.9315, 0.935, 0.9254]),  # df_improved_env_biowin
        np.array([0.9548, 0.9434, 0.9446, 0.9548, 0.9479]),  # df_biowin_both
        np.array([0.9587, 0.9563, 0.949, 0.9417, 0.9392]),  # df_improved_biowin_both
        np.array([0.9566, 0.9592, 0.946, 0.946, 0.9552]),  # df_improved_env_biowin_both
    ]
    all_data_sensitivity = [
        np.array([0.7413, 0.7343, 0.7599, 0.7459, 0.715]),  # df_paper
        np.array([0.7404, 0.7538, 0.741, 0.759, 0.7198]),  # df_improved
        np.array([0.7414, 0.7573, 0.7599, 0.7381, 0.7751]),  # df_improved_env
        np.array([0.8949, 0.8917, 0.8758, 0.8762, 0.8952]),  # df_biowin
        np.array([0.9017, 0.9119, 0.8716, 0.9153, 0.9186]),  # df_improved_biowin
        np.array([0.9288, 0.8876, 0.8843, 0.9176, 0.8539]),  # df_improved_env_biowin
        np.array([0.9258, 0.9223, 0.9081, 0.9117, 0.9117]),  # df_biowin_both
        np.array([0.9135, 0.9248, 0.9173, 0.9211, 0.8947]),  # df_improved_biowin_both
        np.array([0.9259, 0.9421, 0.9256, 0.8926, 0.9383]),  # df_improved_env_biowin_both
    ]
    all_data_specificity = [
        np.array([0.8636, 0.8648, 0.8486, 0.8411, 0.8761]),  # df_paper
        np.array([0.8652, 0.8554, 0.8871, 0.8512, 0.8774]),  # df_improved
        np.array([0.8604, 0.8789, 0.8561, 0.8789, 0.8504]),  # df_improved_env
        np.array([0.9619, 0.9589, 0.9361, 0.9482, 0.9573]),  # df_biowin
        np.array([0.9492, 0.9557, 0.9491, 0.9491, 0.9507]),  # df_improved_biowin
        np.array([0.9646, 0.9664, 0.9539, 0.9433, 0.9592]),  # df_improved_env_biowin
        np.array([0.9684, 0.9534, 0.9617, 0.975, 0.965]),  # df_biowin_both
        np.array([0.9803, 0.9713, 0.9641, 0.9515, 0.9604]),  # df_improved_biowin_both
        np.array([0.971, 0.9671, 0.9555, 0.971, 0.9632]),  # df_improved_env_biowin_both
    ]
    all_data_f1 = [
        np.array([0.743, 0.7394, 0.7443, 0.7306, 0.7347]),  # df_paper
        np.array([0.7432, 0.7452, 0.7595, 0.7456, 0.7388]),  # df_improved
        np.array([0.7414, 0.7643, 0.75, 0.752, 0.7552]),  # df_improved_env
        np.array([0.9065, 0.9018, 0.8716, 0.8832, 0.9024]),  # df_biowin
        np.array([0.8986, 0.9103, 0.8821, 0.906, 0.9094]),  # df_improved_biowin
        np.array([0.9271, 0.9063, 0.8927, 0.9007, 0.8803]),  # df_improved_env_biowin
        np.array([0.9291, 0.9126, 0.913, 0.9281, 0.9181]),  # df_biowin_both
        np.array([0.9346, 0.9318, 0.9208, 0.9108, 0.9049]),  # df_improved_biowin_both
        np.array([0.9317, 0.9363, 0.9162, 0.9133, 0.9306]),  # df_improved_env_biowin_both
    ]
    return all_data_accuracy, all_data_f1, all_data_sensitivity, all_data_specificity


def results_improved_data_classification_nsplits5_seed42_fixed_testset_progress() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_accuracy = [
        np.array([0.9126, 0.9277, 0.8986, 0.9102, 0.8938]),  # df_paper
        np.array([0.9219, 0.9254, 0.8974, 0.9067, 0.902]),  # df_paper_with_speciation
        np.array([0.9126, 0.9336, 0.9149, 0.9207, 0.8938]),  # df_improved
        np.array([0.9009, 0.9103, 0.8928, 0.8938, 0.8856]),  # df_improved_env
        np.array([0.9452, 0.9429, 0.9487, 0.937, 0.9475]),  # df_improved_env_biowin
        np.array([0.9487, 0.9476, 0.9452, 0.9498, 0.9522]),  # df_improved_env_biowin_both
        np.array([0.9557, 0.958, 0.9534, 0.9568, 0.9487]),  # df_improved_env_biowin_both_readded
        # np.array([0.9382, 0.9429, 0.9441, 0.9545, 0.9487]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9487, 0.9394, 0.9429, 0.9487, 0.944]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_sensitivity = [
        np.array([0.8645, 0.8974, 0.8645, 0.8676, 0.8388]),  # df_paper
        np.array([0.8718, 0.8901, 0.8352, 0.8676, 0.8608]),  # df_paper_with_speciation
        np.array([0.8645, 0.8864, 0.8864, 0.8934, 0.8425]),  # df_improved
        np.array([0.8388, 0.8608, 0.8535, 0.8346, 0.8315]),  # df_improved_env
        np.array([0.9048, 0.9084, 0.8938, 0.9265, 0.9121]),  # df_improved_env_biowin
        np.array([0.9011, 0.9194, 0.9121, 0.9154, 0.9267]),  # df_improved_env_biowin_both
        np.array([0.9194, 0.9341, 0.9194, 0.9338, 0.9194]),  # df_improved_env_biowin_both_readded
        # np.array([0.8828, 0.9158, 0.8938, 0.9191, 0.9084]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9011, 0.9084, 0.8938, 0.9044, 0.9121]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_specificity = [
        np.array([0.935, 0.9419, 0.9145, 0.9299, 0.9195]),  # df_paper
        np.array([0.9453, 0.9419, 0.9265, 0.9248, 0.9212]),  # df_paper_with_speciation
        np.array([0.935, 0.9556, 0.9282, 0.9333, 0.9178]),  # df_improved
        np.array([0.9299, 0.9333, 0.9111, 0.9214, 0.911]),  # df_improved_env
        np.array([0.9641, 0.959, 0.9744, 0.9419, 0.964]),  # df_improved_env_biowin
        np.array([0.9709, 0.9607, 0.9607, 0.9658, 0.964]),  # df_improved_env_biowin_both
        np.array([0.9726, 0.9692, 0.9692, 0.9675, 0.9623]),  # df_improved_env_biowin_both_readded
        # np.array([0.9641, 0.9556, 0.9675, 0.9709, 0.9675]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9709, 0.9538, 0.9658, 0.9692, 0.9589]),  # df_improved_env_biowin_both_reliability1
    ]
    all_data_f1 = [
        np.array([0.8629, 0.8877, 0.8444, 0.8597, 0.8342]),  # df_paper
        np.array([0.8766, 0.8836, 0.8382, 0.8551, 0.8484]),  # df_paper_with_speciation
        np.array([0.8629, 0.8946, 0.8689, 0.8773, 0.8348]),  # df_improved
        np.array([0.8435, 0.8592, 0.8351, 0.833, 0.8225]),  # df_improved_env
        np.array([0.9131, 0.9101, 0.9173, 0.9032, 0.9171]),  # df_improved_env_biowin
        np.array([0.9179, 0.9177, 0.9138, 0.9205, 0.925]),  # df_improved_env_biowin_both
        np.array([0.9296, 0.9341, 0.9262, 0.9321, 0.9194]),  # df_improved_env_biowin_both_readded
        # np.array([0.9009, 0.9107, 0.9104, 0.9276, 0.9185]),  # df_improved_env_biowin_both_no_DOC
        # np.array([0.9179, 0.9051, 0.9088, 0.9179, 0.9121]),  # df_improved_env_biowin_both_reliability1
    ]
    return all_data_accuracy, all_data_f1, all_data_sensitivity, all_data_specificity


def results_improved_data_classification_nsplits5_seed42_fixed_testset_comparison() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_accuracy = [
        np.array([0.8869, 0.9061, 0.9218, 0.9239, 0.9307]),  # df_paper
        np.array([0.8869, 0.9079, 0.9218, 0.9022, 0.9176]),  # df_improved
        np.array([0.8759, 0.9024, 0.9106, 0.904, 0.9139]),  # df_improved_env
        np.array([0.9708, 0.9558, 0.9665, 0.971, 0.9551]),  # df_biowin
        np.array([0.9763, 0.9503, 0.9646, 0.9728, 0.9607]),  # df_improved_biowin
        np.array([0.9599, 0.9448, 0.9516, 0.9728, 0.9532]),  # df_improved_env_biowin
        np.array([0.9672, 0.9448, 0.9721, 0.9692, 0.9532]),  # df_biowin_both
        np.array([0.9745, 0.9448, 0.9683, 0.9656, 0.9494]),  # df_improved_biowin_both
        np.array([0.9507, 0.9392, 0.9572, 0.9638, 0.9476]),  # df_improved_env_biowin_both
    ]
    all_data_sensitivity = [
        np.array([0.8352, 0.8293, 0.8957, 0.8786, 0.878]),  # df_paper
        np.array([0.8297, 0.8354, 0.9018, 0.8208, 0.872]),  # df_improved
        np.array([0.8242, 0.8476, 0.8896, 0.8439, 0.8598]),  # df_improved_env
        np.array([0.967, 0.9451, 0.9816, 0.9595, 0.9207]),  # df_biowin
        np.array([0.967, 0.9451, 0.9632, 0.9653, 0.9268]),  # df_improved_biowin
        np.array([0.9231, 0.9085, 0.9509, 0.9538, 0.9207]),  # df_improved_env_biowin
        np.array([0.956, 0.9329, 0.9816, 0.9653, 0.9207]),  # df_biowin_both
        np.array([0.956, 0.9329, 0.9877, 0.9653, 0.9207]),  # df_improved_biowin_both
        np.array([0.9396, 0.9207, 0.9632, 0.9595, 0.9207]),  # df_improved_env_biowin_both
    ]
    all_data_specificity = [
        np.array([0.9126, 0.9393, 0.9332, 0.9446, 0.9541]),  # df_paper
        np.array([0.9153, 0.9393, 0.9305, 0.9393, 0.9378]),  # df_improved
        np.array([0.9016, 0.9261, 0.9198, 0.9314, 0.9378]),  # df_improved_env
        np.array([0.9727, 0.9604, 0.9599, 0.9763, 0.9703]),  # df_biowin
        np.array([0.9809, 0.9525, 0.9652, 0.9763, 0.9757]),  # df_improved_biowin
        np.array([0.9781, 0.9604, 0.9519, 0.9815, 0.9676]),  # df_improved_env_biowin
        np.array([0.9727, 0.9499, 0.9679, 0.971, 0.9676]),  # df_biowin_both
        np.array([0.9836, 0.9499, 0.9599, 0.9657, 0.9622]),  # df_improved_biowin_both
        np.array([0.9563, 0.9472, 0.9545, 0.9657, 0.9595]),  # df_improved_env_biowin_both
    ]
    all_data_f1 = [
        np.array([0.8306, 0.8421, 0.8743, 0.8786, 0.8862]),  # df_paper
        np.array([0.8297, 0.8457, 0.875, 0.8402, 0.8667]),  # df_improved
        np.array([0.8152, 0.8399, 0.858, 0.8464, 0.8598]),  # df_improved_env
        np.array([0.9565, 0.9281, 0.9467, 0.954, 0.9264]),  # df_biowin
        np.array([0.9644, 0.9199, 0.9429, 0.957, 0.9354]),  # df_improved_biowin
        np.array([0.9385, 0.9085, 0.9226, 0.9565, 0.9235]),  # df_improved_env_biowin
        np.array([0.9508, 0.9107, 0.9552, 0.9516, 0.9235]),  # df_biowin_both
        np.array([0.9613, 0.9107, 0.9499, 0.9462, 0.9179]),  # df_improved_biowin_both
        np.array([0.9268, 0.9015, 0.9318, 0.9432, 0.9152]),  # df_improved_env_biowin_both
    ]
    return all_data_accuracy, all_data_f1, all_data_sensitivity, all_data_specificity


def results_lazy_regressors_nsplit5_seed42() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_rmse = [
        np.array([0.1888, 0.1902, 0.1884, 0.1917, 0.1806]),  # LGBMRegressor
        np.array([0.1926, 0.1908, 0.1946, 0.1952, 0.19]),  # HistGradientBoostingRegressor
        np.array([0.2116, 0.2174, 0.2126, 0.2113, 0.2091]),  # RandomForestRegressor
        np.array([0.1889, 0.1924, 0.1888, 0.1906, 0.1882]),  # XGBRegressor
        np.array([0.2129, 0.2118, 0.208, 0.2146, 0.212]),  # SVR
    ]
    all_data_mae = [
        np.array([0.1307, 0.1334, 0.1328, 0.1341, 0.1293]),  # LGBMRegressor
        np.array([0.1371, 0.1351, 0.1399, 0.1407, 0.1382]),  # HistGradientBoostingRegressor
        np.array([0.16, 0.1646, 0.1675, 0.1646, 0.1629]),  # RandomForestRegressor
        np.array([0.1338, 0.1372, 0.1383, 0.139, 0.1389]),  # XGBRegressor
        np.array([0.1579, 0.1604, 0.159, 0.1624, 0.1617]),  # SVR
    ]
    all_data_r2 = [
        np.array([0.7388, 0.7352, 0.7484, 0.7285, 0.7665]),  # LGBMRegressor
        np.array([0.728, 0.7335, 0.7316, 0.7182, 0.7416]),  # HistGradientBoostingRegressor
        np.array([0.6718, 0.654, 0.6796, 0.6698, 0.6869]),  # RandomForestRegressor
        np.array([0.7385, 0.729, 0.7473, 0.7315, 0.7464]),  # XGBRegressor
        np.array([0.6678, 0.6715, 0.6932, 0.6597, 0.678]),  # SVR
    ]
    all_data_mse = [
        np.array([0.0356, 0.0362, 0.0355, 0.0367, 0.0326]),  # LGBMRegressor
        np.array([0.0371, 0.0364, 0.0379, 0.0381, 0.0361]),  # HistGradientBoostingRegressor
        np.array([0.0448, 0.0472, 0.0452, 0.0447, 0.0437]),  # RandomForestRegressor
        np.array([0.0357, 0.037, 0.0356, 0.0363, 0.0354]),  # XGBRegressor
        np.array([0.0453, 0.0449, 0.0433, 0.046, 0.045]),  # SVR
    ]
    return all_data_rmse, all_data_mae, all_data_r2, all_data_mse


def results_lazy_classifiers_nsplit5_seed42() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    all_data_accuracy = [
        np.array([0.9557, 0.9639, 0.9557, 0.9568, 0.9522]),  # XGBClassifier
        np.array([0.9522, 0.958, 0.9569, 0.9522, 0.944]),  # LGBMClassifier
        np.array([0.9569, 0.9569, 0.9441, 0.9522, 0.951]),  # ExtraTreesClassifier
        np.array([0.9429, 0.9406, 0.9534, 0.9475, 0.9428]),  # RandomForestClassifier
        np.array([0.9569, 0.9534, 0.9452, 0.9452, 0.9487]),  # BaggingClassifier
    ]
    all_data_sensitivity = [
        np.array([0.9267, 0.9451, 0.9304, 0.9412, 0.9414]),  # XGBClassifier
        np.array([0.9121, 0.9341, 0.9304, 0.9154, 0.9304]),  # LGBMClassifier
        np.array([0.9194, 0.9414, 0.9194, 0.9228, 0.9231]),  # ExtraTreesClassifier
        np.array([0.9304, 0.9084, 0.9231, 0.9118, 0.9158]),  # RandomForestClassifier
        np.array([0.9158, 0.9304, 0.9121, 0.9081, 0.9158]),  # BaggingClassifier
    ]
    all_data_specificity = [
        np.array([0.9692, 0.9726, 0.9675, 0.9641, 0.9572]),  # XGBClassifier
        np.array([0.9709, 0.9692, 0.9692, 0.9692, 0.9503]),  # LGBMClassifier
        np.array([0.9744, 0.9641, 0.9556, 0.9658, 0.964]),  # ExtraTreesClassifier
        np.array([0.9487, 0.9556, 0.9675, 0.9641, 0.9555]),  # RandomForestClassifier
        np.array([0.9761, 0.9641, 0.9607, 0.9624, 0.964]),  # BaggingClassifier
    ]
    all_data_f1 = [
        np.array([0.9301, 0.9433, 0.9304, 0.9326, 0.9261]),  # XGBClassifier
        np.array([0.9239, 0.9341, 0.9321, 0.9239, 0.9137]),  # LGBMClassifier
        np.array([0.9314, 0.9328, 0.9127, 0.9245, 0.9231]),  # ExtraTreesClassifier
        np.array([0.912, 0.9068, 0.9265, 0.9168, 0.9107]),  # RandomForestClassifier
        np.array([0.9311, 0.927, 0.9138, 0.9131, 0.9191]),  # BaggingClassifier
    ]
    return all_data_accuracy, all_data_f1, all_data_sensitivity, all_data_specificity
