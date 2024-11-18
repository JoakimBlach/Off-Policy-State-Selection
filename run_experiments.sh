
# confounding
alpha_values=(3)  # Adjust values as needed

# Loop through each value in the intercept list
for alpha in "${alpha_values[@]}"; do
    # Use yq to replace the intercept and save to a new file for each iteration

        # ( .variables.L1.kernel.terms[0].param) = 0 |
        # ( .variables.L1.kernel.terms[1].param) = 0 |

        # ( .variables.A1.kernel.indicator_terms[] | select(.type == \"greater_than_value\") | .intercept ) = 0 |
        # ( .variables.A1.kernel.indicator_terms[] | select(.type == \"greater_than_value\") | .terms[0].param ) = 0 |
        # ( .variables.A1.kernel.indicator_terms[] | select(.type == \"greater_than_value\") | .terms[1].param ) = 0 |

    # TODO: Make behavior policy better:
    # 1. Exogenous variables that ensure that when P_1 and B_1 are high B_2 will be high.

    yq "( .variables.L1.kernel.intercept ) = 0 |
        ( .variables.L1.kernel.indicator_terms[1].terms[0].param) = 1 |
        ( .variables.L1.kernel.indicator_terms[2].intercept) = ${alpha} |
        ( .variables.A1.kernel.indicator_terms[] | select(.type == \"smaller_or_greater_than_value\") | .intercept ) = 0 |
        ( .variables.A1.kernel.indicator_terms[] | select(.type == \"smaller_or_greater_than_value\") | .terms[0].param ) = 0 |
        ( .variables.A1.kernel.indicator_terms[] | select(.type == \"smaller_or_greater_than_value\") | .terms[1].param ) = 0 |
        ( .variables.A1.kernel.terms[] | select(.variables.\"0\"[] == \"U1\") | .param ) = 1 |
        ( .variables.A1.kernel.intercept ) = -1" \
        config/pricing_specs.yaml > "config/pricing_confounded_${alpha}.yaml"

    python3 run.py --num_episodes=5000 --config_file="config/pricing_confounded_${alpha}.yaml" --output_file="output/pricing_confounded_${alpha}.pkl" --save_results="True" --plot="False"

    echo "Experiment done!"
done