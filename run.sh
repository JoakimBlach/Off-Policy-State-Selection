
# RUN TOY SIM
spec_values=(1 2 3)

for spec in "${spec_values[@]}"; do
    python3 run.py --num_episodes=5000 --episode_length=500 --num_eval_episodes=1000 --state_specs="example_${spec}" --file_name="example_${spec}" --save_results="True" --plot="False"
done

# RUN REAL SIM
episode_length=500
num_episodes=100000
num_eval_episodes=10000

# 1. Run pricing specs
python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=${num_eval_episodes} --episode_length=${episode_length} --state_specs="pricing_specs" --file_name="pricing_specs" --save_results="True" --plot="False"

# 2. No-Markov
alpha_values=(0 0.1 0.25 0.5 0.75 0.9)

for alpha in "${alpha_values[@]}"; do
    prob=$(echo "scale=2; 1 - ${alpha}" | bc)  # scale=2 ensures 2 decimal places for better precision
    prob=$(printf "%.2f" "$prob")  # This will ensure 2 decimal places with a trailing zero if needed

    echo "prob value for alpha=${alpha}: ${prob}"

    yq "( .D.kernel.noise) = ${prob}" \
        config/pricing_specs.yaml > "config/pricing_no_markov_${alpha}.yaml"

    python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=${num_eval_episodes} --episode_length=${episode_length} --state_specs="pricing_no_markov" --file_name="pricing_no_markov_${alpha}" --save_results="True" --plot="False"

    echo "... done with ${alpha}"
done

# 3. Dyn. back-door (150.000 should be enough.)
alpha_values=(0 1 2 4)
for alpha in "${alpha_values[@]}"; do

    if [ $alpha == 0 ]; then
        yq eval "( .L2.kernel.terms[0].intercept) = ${alpha} |
                 ( .A2.kernel.terms[1].value) = 0" \
            config/pricing_specs.yaml > "config/pricing_dyn_back_door_${alpha}.yaml"
    else
        yq eval "( .L2.kernel.terms[0].intercept) = ${alpha} |
                 ( .A2.kernel.terms[1].value) = -1" \
            config/pricing_specs.yaml > "config/pricing_dyn_back_door_${alpha}.yaml"
    fi

    python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=${num_eval_episodes} --episode_length=${episode_length} --state_specs="pricing_dyn_back_door" --file_name="pricing_dyn_back_door_${alpha}" --save_results="True" --plot="False" --append_data="True"

    echo "... done with ${alpha}"

done

# 4. Confounding
alpha_values=(0 1 3 5)
probs_values=(1 0.5 0.25 0.1)

for i in "${!alpha_values[@]}"; do
    alpha=${alpha_values[i]}
    prob1=${probs_values[i]}

    prob2=$(printf "%.1f" "$(echo "1 - ${prob1}" | bc)")

    yq eval "( .L1.kernel.terms[0].intercept) = ${alpha} |
             ( .A1.kernel.mixed_probs) = [${prob1}, ${prob2}]" \
        config/pricing_specs.yaml > "config/pricing_confounded_${alpha}.yaml"

    python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=${num_eval_episodes} --episode_length=${episode_length} --state_specs="pricing_confounded" --file_name="pricing_confounded_${alpha}" --save_results="True" --plot="False"

done