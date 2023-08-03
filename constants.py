import numpy as np

# N=88 main analyses subjects
FR2_valid_subjects = np.array(['LTP093', 'LTP106', 'LTP115', 'LTP117', 'LTP123', 'LTP133',
                               'LTP138', 'LTP207', 'LTP210', 'LTP228', 'LTP229', 'LTP236',
                               'LTP246', 'LTP249', 'LTP250', 'LTP251', 'LTP258', 'LTP259',
                               'LTP265', 'LTP269', 'LTP273', 'LTP278', 'LTP279', 'LTP280',
                               'LTP283', 'LTP285', 'LTP287', 'LTP293', 'LTP296', 'LTP297',
                               'LTP299', 'LTP301', 'LTP302', 'LTP303', 'LTP304', 'LTP305',
                               'LTP306', 'LTP307', 'LTP310', 'LTP311', 'LTP312', 'LTP316',
                               'LTP317', 'LTP318', 'LTP321', 'LTP322', 'LTP323', 'LTP324',
                               'LTP325', 'LTP326', 'LTP327', 'LTP328', 'LTP329', 'LTP331',
                               'LTP334', 'LTP336', 'LTP339', 'LTP341', 'LTP342', 'LTP343',
                               'LTP344', 'LTP346', 'LTP347', 'LTP348', 'LTP349', 'LTP354',
                               'LTP355', 'LTP357', 'LTP360', 'LTP361', 'LTP362', 'LTP364',
                               'LTP365', 'LTP366', 'LTP367', 'LTP371', 'LTP372', 'LTP373',
                               'LTP374', 'LTP376', 'LTP377', 'LTP385', 'LTP386', 'LTP387', 
                               'LTP389', 'LTP390', 'LTP391', 'LTP393'])

# participants whose power files are stored in home dir because scratch is full...
subjects_powerfile_elsewhere = ['LTP357', 'LTP360', 'LTP361', 'LTP365', 'LTP366']

# sorted by recall performance
FR2_subjects_sorted = np.array(['LTP348', 'LTP278', 'LTP228', 'LTP301', 'LTP210', 'LTP273',
       'LTP393', 'LTP287', 'LTP362', 'LTP305', 'LTP390', 'LTP207',
       'LTP386', 'LTP250', 'LTP360', 'LTP376', 'LTP229', 'LTP339',
       'LTP317', 'LTP280', 'LTP349', 'LTP354', 'LTP377', 'LTP312',
       'LTP321', 'LTP336', 'LTP385', 'LTP366', 'LTP364', 'LTP093',
       'LTP343', 'LTP296', 'LTP302', 'LTP329', 'LTP325', 'LTP297',
       'LTP133', 'LTP249', 'LTP357', 'LTP344', 'LTP324', 'LTP299',
       'LTP269', 'LTP373', 'LTP326', 'LTP293', 'LTP318', 'LTP285',
       'LTP279', 'LTP123', 'LTP258', 'LTP251', 'LTP341', 'LTP322',
       'LTP259', 'LTP391', 'LTP304', 'LTP355', 'LTP365', 'LTP334',
       'LTP310', 'LTP389', 'LTP311', 'LTP138', 'LTP306', 'LTP236',
       'LTP361', 'LTP283', 'LTP265', 'LTP328', 'LTP307', 'LTP115',
       'LTP374', 'LTP387', 'LTP342', 'LTP246', 'LTP117', 'LTP372',
       'LTP347', 'LTP316', 'LTP346', 'LTP323', 'LTP331', 'LTP106',
       'LTP303', 'LTP367', 'LTP371', 'LTP327'])

# subjects with significant encoding classifiers
enc_sig_subjects = np.array(['LTP093', 'LTP106', 'LTP115', 'LTP117', 'LTP123', 'LTP133',
                            'LTP138', 'LTP207', 'LTP210', 'LTP228', 'LTP229', 'LTP236',
                            'LTP246', 'LTP249', 'LTP250', 'LTP251', 'LTP258', 'LTP259',
                            'LTP265', 'LTP269', 'LTP273', 'LTP278', 'LTP279', 'LTP280',
                            'LTP283', 'LTP285', 'LTP287', 'LTP293', 'LTP296', 'LTP297',
                            'LTP299', 'LTP301', 'LTP302', 'LTP303', 'LTP304', 'LTP305',
                            'LTP306', 'LTP307', 'LTP310', 'LTP311', 'LTP312', 'LTP316',
                            'LTP317', 'LTP318', 'LTP321', 'LTP322', 'LTP323', 'LTP324',
                            'LTP325', 'LTP326', 'LTP327', 'LTP328', 'LTP329', 'LTP331',
                            'LTP334', 'LTP336', 'LTP339', 'LTP341', 'LTP342', 'LTP343',
                            'LTP344', 'LTP346', 'LTP347', 'LTP348', 'LTP349', 'LTP354',
                            'LTP355', 'LTP357', 'LTP360', 'LTP361', 'LTP362', 'LTP364',
                            'LTP365', 'LTP366', 'LTP367', 'LTP371', 'LTP372', 'LTP373',
                            'LTP374', 'LTP376', 'LTP377', 'LTP385', 'LTP386', 'LTP387',
                            'LTP389', 'LTP390', 'LTP391', 'LTP393'], dtype='<U6')

# subjects with significant retrieval classifiers
ret_sig_subjects = np.array(['LTP106', 'LTP115', 'LTP117', 'LTP123', 'LTP133', 'LTP138',
                            'LTP207', 'LTP210', 'LTP229', 'LTP251', 'LTP258', 'LTP259',
                            'LTP273', 'LTP278', 'LTP279', 'LTP280', 'LTP285', 'LTP287',
                            'LTP296', 'LTP297', 'LTP301', 'LTP304', 'LTP306', 'LTP307',
                            'LTP310', 'LTP312', 'LTP316', 'LTP317', 'LTP318', 'LTP321',
                            'LTP322', 'LTP323', 'LTP325', 'LTP326', 'LTP329', 'LTP334',
                            'LTP336', 'LTP339', 'LTP341', 'LTP342', 'LTP343', 'LTP346',
                            'LTP347', 'LTP348', 'LTP349', 'LTP354', 'LTP357', 'LTP361',
                            'LTP362', 'LTP364', 'LTP365', 'LTP366', 'LTP367', 'LTP371',
                            'LTP372', 'LTP373', 'LTP376', 'LTP377', 'LTP385', 'LTP386',
                            'LTP387', 'LTP389', 'LTP390', 'LTP393'], dtype='<U6')

# eeg cap ROI assignment (excluding electrodes close to face/neck)
ROIs = {'egi':{'LAS':['E12','E13','E19','E20','E24','E28','E29'],
               'LAI':['E23','E26','E27','E33','E34','E39','E40'],
               'LPS':['E37','E42','E52','E53','E54','E60','E61'],
               'LPI':['E51','E58','E59','E64','E65','E66','E69'],
               'RAS':['E4','E5','E111','E112','E117','E118','E124'],
               'RAI':['E2','E3','E109','E115','E116','E122','E123'],
               'RPS':['E78','E79','E85','E86','E87','E92','E93'],
               'RPI':['E84','E89','E90','E91','E95','E96','E97']},
        'biosemi':{'LAS':['C24','C25','D2','D3','D4','D11','D12','D13'],
                   'LAI':['C31','C32','D5','D6','D9','D10','D21','D22'],
                   'LPS':['D29','A5','A6','A7','A8','A17','A18'],
                   'LPI':['D30','D31','A9','A10','A11','A15','A16'],
                   'RAS':['B30','B31','B32','C2','C3','C4','C11','C12'],
                   'RAI':['B24','B25','B28','B29','C5','C6','C9','C10'],
                   'RPS':['A30','A31','A32','B3','B4','B5','B13'],
                   'RPI':['A28','A29','B6','B7','B8','B11','B12']}}

non_peripheral_egi = np.array(['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10',
                               'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
                               'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30',
                               'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
                               'E40', 'E41', 'E42', 'E44', 'E45', 'E46', 'E47', 'E50',
                               'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59',
                               'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69',
                               'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78',
                               'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87',
                               'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97',
                               'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106',
                               'E108', 'E109', 'E110', 'E111', 'E112', 'E114', 'E115', 'E116',
                               'E117', 'E118', 'E121', 'E122', 'E123', 'E124'])

non_peripheral_biosemi = np.array(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
                                   'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22',
                                   'A23', 'A24', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2',
                                   'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11', 'B12', 'B13',
                                   'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22',
                                   'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31',
                                   'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C10', 'C11', 'C12', 'C13',
                                   'C14', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26',
                                   'C27', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                                   'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18',
                                   'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27',
                                   'D28', 'D29', 'D30', 'D31'])

all_biosemi_channels = np.array(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
                               'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
                               'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29',
                               'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                               'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
                               'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26',
                               'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4',
                               'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                               'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23',
                               'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32',
                               'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11',
                               'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',
                               'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29',
                               'D30', 'D31', 'D32'])

frequencies = np.array([  2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  22.,
                          24.,  26.,  32.,  38.,  44.,  50.,  56.,  62.,  68.,  74.,  80.,
                          86.,  92.,  98., 104., 110., 116., 122., 128.])

best_C = np.logspace(np.log10(1e-7),np.log10(1e0), num = 10)[3]