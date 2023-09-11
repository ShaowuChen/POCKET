def process_result(my_result, descriptions, new_result, new_description, run, results_index):
    if run == 0:
        my_result.append([new_result])
        descriptions.append(new_description)
    else:
        my_result[results_index].append(new_result)
    return my_result, descriptions