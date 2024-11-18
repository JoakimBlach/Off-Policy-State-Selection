import os 
import sys
import yaml
import numpy as np
import subprocess

# running other file using run()
with open("config/example3_specs.yaml", 'r') as file:
    specs = yaml.safe_load(file)

test_folder = "config/example3_tests"
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# load all specs
all_current_specs = []
for spec_name in os.listdir(test_folder):
    with open(os.path.join(test_folder, spec_name), 'r') as file:
        all_current_specs.append(yaml.safe_load(file))

# Get current counter
config_numbers = [int(i.split(".")[0]) for i in os.listdir(test_folder)]

k = 0
if config_numbers:
    k = max(config_numbers) + 1

for m_u1 in [np.float64(-5), np.float64(0), np.float64(5)]:
    specs['variables']['M']['kernel']['terms'][0]['param'] = m_u1.item()

    for a_u1 in [np.float64(-1), np.float64(1)]:
        specs['variables']['A']['kernel']['terms'][0]['param'] = a_u1.item()

        for r_a in [np.float64(-10), np.float64(-1), np.float64(0.1)]:
            specs['variables']['R']['kernel']['terms'][1]['param'] = r_a.item()

            for r_u2 in [np.float64(-1), np.float64(1)]:
                specs['variables']['R']['kernel']['terms'][2]['param'] = r_u2.item()

                # check if spec already exists
                already_exists = False
                for current_spec in all_current_specs:
                    if specs == current_spec:
                        already_exists = True

                if already_exists:
                    print("Spec already exists!")
                    continue

                with open(os.path.join(test_folder, f"{k}.yaml"), 'w') as outfile:
                    yaml.dump(specs, outfile)

                k += 1

                print("Saved new spec.")


# for l1_intercept in [np.float64(-3), np.float64(0), np.float64(3)]:
#     specs['variables']['L1']['kernel']['intercept'] = l1_intercept.item()

#     for l1_a2 in [np.float64(1), np.float64(2), np.float64(4)]:
#         specs['variables']['L1']['kernel']['terms'][1]['param'] = l1_a2.item()

#         for a2_intercept in [np.float64(-3), np.float64(0), np.float64(3)]:
#             specs['variables']['A2']['kernel']['intercept'] = a2_intercept.item()

#             for a2_l1 in [np.float64(1), np.float64(4)]:
#                 specs['variables']['A2']['kernel']['terms'][1]['param'] = a2_l1.item()

#                 # check if spec already exists
#                 already_exists = False
#                 for current_spec in all_current_specs:
#                     if specs == current_spec:
#                         already_exists = True

#                 if already_exists:
#                     print("Spec already exists!")
#                     continue

#                 with open(os.path.join(test_folder, f"{k}.yaml"), 'w') as outfile:
#                     yaml.dump(specs, outfile)

#                 k += 1

#                 print("Saved new spec.")
