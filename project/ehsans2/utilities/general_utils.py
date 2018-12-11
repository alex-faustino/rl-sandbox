def dict_append(main_dict, update_dict):
    for key, val in update_dict.items():
        if not key in main_dict.keys():
            main_dict[key] = []
        main_dict[key].append(val)