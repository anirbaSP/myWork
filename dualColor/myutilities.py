def find(search_list, elem):
    return [[i for i, x in enumerate(search_list) if e in x] for e in elem]