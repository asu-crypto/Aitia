valid_data = ['abalone', 'arrhythmia', 'income', 'liver_disorder', 'ncep']

valid_ci_pair = {
    'abalone': [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)],
    'arrhythmia': [(0, 1), (0, 2), (0, 3)],
    'income': [(0, 1), (0, 2)],
    'liver_disorder': [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
    'ncep': [(0, 1), (2, 3), (4, 5), (6, 7)],
}

def check_valid_pair(name, feature, target):
    if name not in valid_data:
        return False
    if (feature, target) in valid_ci_pair[name]:
        return True
    else:
        print(f"Valid pairs for {name} are {valid_ci_pair[name]}")
        return False

sigma_map =  {
    'abalone': 0.01,
    'liver_disorder': 0.01,
    # 'arrhythmia': 0.001, # this is for 0->{2, 3}
    'arrhythmia': 0.01,
    'income': 0.01,
    'ncep': 0.01,
}

tol_map = {
    'abalone': 0.01,
    'liver_disorder': 0.05,
    # 'arrhythmia': 0.01, # this is for 0->{2, 3}
    'arrhythmia': 0.01,
    'income': 0.001,
    'ncep': 0.01,
}
