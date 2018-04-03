import json
import os
from os.path import join

data = {

    'NV3-04, Lime (2,18mW) LED': 'NV3-04/led2_0_gain0_coverOff/Movie_2018-02-21-14-28-26',  # 1,6  gain0, coverOff
    'NV3-04, Lime (2,18mW) LED': 'NV3-04/led2_0_gain1_coverOff/Movie_2018-02-21-14-28-42',  # 1.6  gain1, coverOff
    'NV3-04, Lime (2,18mW) LED': 'NV3-04/led2_0_gain2_coverOff/Movie_2018-02-21-14-28-58',  # 1.6  gain2, coverOff
    'NV3-04, Lime (2,18mW) LED': 'NV3-04/led2_0_gain3_coverOff/Movie_2018-02-21-14-29-17',  # 1.6  gain3, coverOff
    'NV3-04, Lime (0,85mW) LED': 'NV3-04/led2_3_gain1_coverOff/Movie_2018-02-21-14-32-07',  # 0.6  gain1, coverOff
    'NV3-04, Lime (0,56mW) LED': 'NV3-04/led2_4_gain1_coverOff/Movie_2018-02-21-14-31-52',  # 0.4  gain1, coverOff
    'NV3-01, Blue (0,4mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-13-18',  # 0.4  gain1, coverOff
    'NV3-01, Blue (0,6mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-13-34',  # 0.4  gain1, coverOff
    'NV3-01, Blue (0,8mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-13-50',  # 0.4  gain1, coverOff
    'NV3-01, Blue (1,0mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-14-11',  # 0.4  gain1, coverOff
    'NV3-01, Blue (1,6mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-14-33',  # 0.4  gain1, coverOff
    'NV3-01, Blue (1,8mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-14-50',  # 0.4  gain1, coverOff
    'NV3-01, Blue (2,0mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-15-08',  # 0.4  gain1, coverOff
    'NV3-01, Lime (0,2mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-17-44',  # 0.4  gain1, coverOff
    'NV3-01, Lime (0,4mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-17-56',  # 0.4  gain1, coverOff
    'NV3-01, Lime (0,6mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-18-14',  # 0.4  gain1, coverOff
    'NV3-01, Lime (0,8mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-18-31',  # 0.4  gain1, coverOff
    'NV3-01, Lime (1,2mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-18-50',  # 0.4  gain1, coverOff
    'NV3-01, Lime (1,4mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-19-05',  # 0.4  gain1, coverOff
    'NV3-01, Lime (1,6mW) LED': 'NV3-01/20180314/Movie_2018-03-14-11-19-19',  # 0.4  gain1, coverOff
}
    # 'NV3-01, Blue (2mW) LED': 'NV3-01/led1_0/Movie_2018-01-26-09-47-45',
    # 'NV3-01, Blue (1,5mW) LED': 'NV3-01/led1_1/Movie_2018-01-26-09-48-42',
    # 'NV3-01, Blue (1mW) LED': 'NV3-01/led1_2/Movie_2018-01-26-09-49-31',
    # 'NV3-01, Lime (1,6mW) LED': 'NV3-01/led2_0/Movie_2018-01-26-09-50-31',
    # 'NV3-01, Lime (1,2mW) LED': 'NV3-01/led2_1/Movie_2018-01-26-09-51-20',
    # 'NV3-01, Lime (0,8mW) LED': 'NV3-01/led2_2/Movie_2018-01-26-09-52-14',
    # 'NV3-01, Blue (2mW) + Lime (1,6mW) LED': 'NV3-01/led12_0/Movie_2018-01-26-09-53-15',
    # 'NV3-01, Blue (1,5mW) + Lime (1,2mW) LED': 'NV3-01/led12_1/Movie_2018-01-26-09-55-18',
    # 'NV3-01, Blue (1mW) + Lime (0,8mW) LED': 'NV3-01/led12_2/Movie_2018-01-26-09-57-16',
    # 'NV3-04, Blue (2mW) LED': 'NV3-04/led1_0/Movie_2018-01-26-10-02-08',
    # 'NV3-04, Blue (1,5mW) LED': 'NV3-04/led1_1/Movie_2018-01-26-10-09-16',
    # 'NV3-04, Blue (1mW) LED': 'NV3-04/led1_2/Movie_2018-01-26-10-10-00',
    # 'NV3-04, Lime (1,6mW) LED': 'NV3-04/led2_0/Movie_2018-01-26-10-10-53',
    # 'NV3-04, Lime (1,2mW) LED': 'NV3-04/led2_1/Movie_2018-01-26-10-12-01',
    # 'NV3-04, Lime (0,8mW) LED': 'NV3-04/led2_2/Movie_2018-01-26-10-17-29',
    # 'NV3-04, Lime (0,8mW) LED': 'NV3-04/led2_2_new/Movie_2018-02-21-14-35-30',
    # 'NV3-04, Lime (0,6mW) LED': 'NV3-04/led2_3/Movie_2018-02-21-14-32-48',
    # 'NV3-04, Lime (0,4mW) LED': 'NV3-04/led2_4/Movie_2018-02-21-14-33-23',  # todo: gain1, coverON
    # 'NV3-04, Lime (0,2mW) LED': 'NV3-04/led2_5/Movie_2018-02-21-14-35-13',
    # 'NV3-04, Lime (0,4mW) LED': 'NV3-04/led2_4_gain0/Movie_2018-02-21-14-33-58',  ## todo: change back to 0.4
    # 'NV3-04, Lime (0,4mW) LED': 'NV3-04/led2_4_gain2/Movie_2018-02-21-14-34-18',
    # 'NV3-04, Lime (0,4mW) LED': 'NV3-04/led2_4_gain3/Movie_2018-02-21-14-34-31',
    # 'NV3-04, Lime (0,56mW) LED': 'NV3-04/led2_4_coverOff/Movie_2018-02-21-14-31-52',  # todo: gain1, coverOff
    # 'NV3-04, Blue (2mW) + Lime (1,6mW) LED': 'NV3-04/led12_0/Movie_2018-01-26-10-18-32',
    # 'NV3-04, Blue (1,5mW) + Lime (1,2mW) LED': 'NV3-04/led12_1/Movie_2018-01-26-10-19-34',
    # 'NV3-04, Blue (1mW) + Lime (0,8mW) LED': 'NV3-04/led12_2/Movie_2018-01-26-10-20-30',

shared_path = '/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/Scope_Autofluorescence/'
for key in data:
    data[key] = os.path.join(shared_path, data[key])

with open('LOOKUP_stray_light_correction.txt', 'w') as outfile:
    json.dump(data, outfile)


# with open('data.txt') as json_file:
#     data = json.load(json_file)
#     print('Movie_2017-08-07-15-01-03: ' + data['Movie_2017-08-07-15-01-03'])


data = json.load(open('LOOKUP_stray_light_correction.txt'))
print('Stray_NV3-01, Blue (2mW) LED: ' + data['NV3-01, Blue (2mW) LED'])