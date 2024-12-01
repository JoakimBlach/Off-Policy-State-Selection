import sys
import yaml

# Load the YAML file
config_files = [
    "config/pricing_specs.yaml",
    "config/pricing_no_markov_0.yaml",
    "config/pricing_no_markov_0.1.yaml",
    "config/pricing_no_markov_0.5.yaml",
    "config/pricing_no_markov_0.9.yaml",
    "config/pricing_dyn_back_door_0.yaml",
    "config/pricing_dyn_back_door_1.yaml",
    "config/pricing_dyn_back_door_2.yaml",
    "config/pricing_dyn_back_door_4.yaml",
    "config/pricing_confounded_0.yaml",
    "config/pricing_confounded_1.yaml",
    "config/pricing_confounded_3.yaml",
    "config/pricing_confounded_5.yaml"
]

variables = [
    r"$C$",
    r"$p_D$",
    r"$p_{\hat{D}}$",
    r"$p_{B^{1}}$",
    r"$\beta^1_0$",
    r"$\beta^1_1$",
    r"$\beta^1_2$",
    r"$\beta^1_3$",
    r"$\beta^1_4$",
    r"$p_{B^{2}}$",
    r"$\beta^2_0$",
    r"$\beta^2_1$",
    r"$\beta^2_2$",
    r"$\beta^2_3$",
    r"$\beta^2_4$",
    r"$p_{P^{1}}$",
    r"$p_{P^{c, 1}}$",
    r"$\alpha^1_0$",
    r"$\alpha^1_1$",
    r"$\alpha^1_2$",
    r"$\alpha^1_3$",
    r"$\alpha^1_4$",
    r"$\xi^1_0$",
    r"$\xi^1_1$",
    r"$p_{P^{2}}$",
    r"$p_{P^{c, 2}}$",
    r"$\alpha^2_0$",
    r"$\alpha^2_1$",
    r"$\alpha^2_2$",
    r"$\alpha^2_3$",
    r"$\alpha^2_4$",
    r"$\xi^2_0$",
    r"$\xi^2_1$" # ,
    # r"$p_R$"
]

def lookup_keys(specs):
    return [
        specs["L1"]["kernel"]["limit"]["value"],
        specs["D"]["kernel"]["noise"],
        specs["DE"]["kernel"]["noise"],
        specs["L1"]["kernel"]["noise"],
        specs["L1"]["kernel"]["terms"][1]["intercept"],
        specs["L1"]["kernel"]["terms"][1]["value"],
        specs["L1"]["kernel"]["terms"][2]["value"],
        specs["L1"]["kernel"]["terms"][0]["intercept"],
        0,
        specs["L2"]["kernel"]["noise"],
        specs["L2"]["kernel"]["terms"][1]["intercept"],
        specs["L2"]["kernel"]["terms"][1]["value"],
        specs["L2"]["kernel"]["terms"][2]["value"],
        0,
        specs["L2"]["kernel"]["terms"][0]["intercept"],
        specs["A1"]["kernel"]["noise"],
        specs["A1"]["kernel"]["mixed_probs"][0],
        specs["A1"]["kernel"]["kernels"][0]["terms"][1]["intercept"],
        specs["A1"]["kernel"]["kernels"][0]["terms"][1]["value"],
        specs["A1"]["kernel"]["kernels"][0]["terms"][0]["intercept"],
        0,
        0,
        specs["A1"]["kernel"]["kernels"][1]['terms'][0]["intercept"],
        specs["A1"]["kernel"]["kernels"][1]['terms'][0]["value"],
        specs["A2"]["kernel"]["noise"],
        1,
        specs["A2"]["kernel"]['terms'][1]["intercept"],
        0,
        specs["A2"]["kernel"]["terms"][0]["intercept"],
        specs["A2"]["kernel"]["terms"][0]["value"],
        specs["A2"]["kernel"]["terms"][1]["value"],
        0,
        0
        # specs["R"]["kernel"]["noise"]
    ]

lines = variables.copy()

for config_file in config_files:
    with open(config_file, "r") as file:
        specs = yaml.safe_load(file)

    for i, (var, value) in enumerate(zip(variables, lookup_keys(specs))):
        lines[i] += f" & {value}"

# Print table
degrees = [file.split("/")[1].split("_")[-1].split(".yaml")[0] for file in config_files][1:]
degree_line = "Degree & - & " + ' & '.join(degrees) + r"\\ \midrule"

columns = ' '.join(["c" for _ in range(1 + len(config_files))])
print(f"\\begin{{tabular}}{{{columns}}} \\toprule")
print(f"& \\multicolumn{{{len(config_files)}}}{{c}}{{Graph}}" + r" \\")
print(f"& $\\gG$ & \\multicolumn{{4}}{{c}}{{$\\overset{{\\textcolor{{yellow}}{{\\longrightarrow}}}}{{\\gG}}$}} & \\multicolumn{{4}}{{c}}{{$\\overset{{\\textcolor{{green}}{{\\longrightarrow}}}}{{\\gG}}$}} & \\multicolumn{{4}}{{c}}{{$\\overset{{\\textcolor{{red}}{{\\longrightarrow}}}}{{\\gG}}$}}"   + r" \\")
print(f"  \\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-10}} \\cmidrule(lr){{11-14}}")
print(degree_line)
for line in lines:
    print(line + r" \\")
print("\\bottomrule")
print("\\end{tabular}")

sys.exit()
