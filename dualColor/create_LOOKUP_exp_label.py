import json

data = {
    'Movie_2018-01-26-09-47-45': 'Stray, NV3-01, Blue (2mW) LED',
    'Movie_2018-01-26-09-48-42': 'Stray, NV3-01, Blue (1,5mW) LED',
    'Movie_2018-01-26-09-49-31': 'Stray, NV3-01, Blue (1mW) LED',
    'Movie_2018-01-26-09-50-31': 'Stray, NV3-01, Lime (2,02mW) LED', #1,6
    'Movie_2018-01-26-09-51-20': 'Stray, NV3-01, Lime (1,54mW) LED', #1.2
    'Movie_2018-01-26-09-52-14': 'Stray, NV3-01, Lime (1,04mW) LED', #0,8
    'Movie_2018-01-26-09-53-15': 'Stray, NV3-01, Blue (2mW) + Lime (2,02mW) LED',
    'Movie_2018-01-26-09-55-18': 'Stray, NV3-01, Blue (1,5mW) + Lime (1,54mW) LED',
    'Movie_2018-01-26-09-57-16': 'Stray, NV3-01, Blue (1mW) + Lime (1,04mW) LED',
    'Movie_2018-01-26-10-02-08': 'Stray, NV3-04, Blue (2mW) LED',
    'Movie_2018-01-26-10-09-16': 'Stray, NV3-04, Blue (1,5mW) LED',
    'Movie_2018-01-26-10-10-00': 'Stray, NV3-04, Blue (1mW) LED',
    'Movie_2018-01-26-10-10-53': 'Stray, NV3-04, Lime (2,18mW) LED', #1,6
    'Movie_2018-01-26-10-12-01': 'Stray, NV3-04, Lime (1,67mW) LED', #1,2
    'Movie_2018-01-26-10-17-29': 'Stray, NV3-04, Lime (1,13mW) LED', #0.8
    'Movie_2018-02-21-14-35-30': 'Stray, NV3-04, Lime (1,13mW) LED', #0.8
    'Movie_2018-02-21-14-32-48': 'Stray, NV3-04, Lime (0,85mW) LED', #0.6
    'Movie_2018-02-21-14-33-23': 'Stray, NV3-04, Lime (0,56mW) LED', #0.4
    'Movie_2018-02-21-14-35-13': 'Stray, NV3-04, Lime (0,25mW) LED', #0.2
    'Movie_2018-02-21-14-33-58': 'Stray, NV3-04, Lime (0,56mW) LED', #0.4
    'Movie_2018-02-21-14-34-18': 'Stray, NV3-04, Lime (0,56mW) LED', #0.4
    'Movie_2018-02-21-14-34-31': 'Stray, NV3-04, Lime (0,56mW) LED', #0.4
    'Movie_2018-02-21-14-28-26': 'Stray, NV3-04, Lime (2,18mW) LED', #1,6  gain0, coverOff
    'Movie_2018-02-21-14-28-42': 'Stray, NV3-04, Lime (2,18mW) LED', #1.6  gain1, coverOff
    'Movie_2018-02-21-14-28-58': 'Stray, NV3-04, Lime (2,18mW) LED', #1.6  gain2, coverOff
    'Movie_2018-02-21-14-29-17': 'Stray, NV3-04, Lime (2,18mW) LED', #1.6  gain3, coverOff
    'Movie_2018-02-21-14-32-07': 'Stray, NV3-04, Lime (0,85mW) LED', #0.6  gain1, coverOff
    'Movie_2018-02-21-14-31-52': 'Stray, NV3-04, Lime (0,56mW) LED', #0.4  gain1, coverOff
    'Movie_2018-01-26-10-18-32': 'Stray, NV3-04, Blue (2mW) + Lime (2,18mW) LED',
    'Movie_2018-01-26-10-19-34': 'Stray, NV3-04, Blue (1,5mW) + Lime (1,67mW) LED',
    'Movie_2018-01-26-10-20-30': 'Stray, NV3-04, Blue (1mW) + Lime (1,13mW) LED',
    'Movie_2017-08-07-15-01-03': 'Autofluo, NV3-01, Blue (2mW) LED',
    'Movie_2017-08-07-15-04-14': 'Autofluo, NV3-01, Lime (1,6mW) LED',
    'Movie_2017-08-07-15-07-35': 'Autofluo, NV3-01, Blue (2mW) + Lime (1,6mW) LED',
    'Movie_2017-07-17-10-39-25': 'GCaMP, NV3-01, Blue (1,2mW) LED',
    'Movie_2017-07-14-16-30-00': 'GCaMP, NV3-01, Lime (1,6mW) LED',
    'Movie_2017-08-07-14-01-28': 'GCaMP, NV3-01, Blue (2mW) LED',
    'Movie_2017-08-07-13-58-45': 'GCaMP, NV3-01, Lime (1,6mW) LED',
    'Movie_2017-07-17-16-20-59': 'RGeco, NV3-01, Blue (2mW) LED',
    'Movie_2017-07-10-15-11-31': 'RGeco, NV3-01, Lime (0,5mW) LED',
    'Movie_2017-11-27-11-45-17': 'GCaMP+RGeco, NV3-01, Blue (1,8mW) LED',
    'Movie_2017-11-27-11-51-02': 'GCaMP+RGeco, NV3-01, Lime (0,9mW) LED',
    'Movie_2017-11-27-11-56-39': 'GCaMP+RGeco, NV3-01, Blue (1,8mW) + Lime (0,9mW) LED',
    'Movie_2018-02-14-11-58-25': 'tdTomato, NV3-04, Lime (0,4mW) LED',
    'Movie_2017-11-27-11-17-27': 'tdTomato+GCaMP, NV3-01, Blue (1,8mW) LED',
    'Movie_2017-11-27-11-22-51': 'tdTomato+GCaMP, NV3-01, Lime (0,5mW) LED',
    'Movie_2017-11-27-11-29-56': 'tdTomato+GCaMP, NV3-01, Blue (1,8mW) LED + Lime (1,27mW) LED',
    'Movie_2017-08-16-11-32-22': 'GCaMP+RGeco, NV3-01, Blue (1,6mW) LED',
    'Movie_2017-08-16-11-35-04': 'GCaMP+RGeco, NV3-01, Lime (0,4mW) LED',
    'Movie_2017-08-16-11-37-29': 'GCaMP+RGeco, NV3-01, Blue (1,6mW) + Lime (0,4mW) LED'
}

with open('LOOKUP_exp_label.txt', 'w') as outfile:
    json.dump(data, outfile)


# with open('data.txt') as json_file:
#     data = json.load(json_file)
#     print('Movie_2017-08-07-15-01-03: ' + data['Movie_2017-08-07-15-01-03'])


data = json.load(open('LOOKUP_exp_label.txt'))
print('Movie_2017-08-07-15-01-03: ' + data['Movie_2017-08-07-15-01-03'])