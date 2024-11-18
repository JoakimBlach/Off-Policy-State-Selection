episode_length=500

# RUN TOY SIM
spec_values=(1 2 3 4)

num_episodes=1000

# for spec in "${spec_values[@]}"; do
#     python3 run.py --num_episodes=${num_episodes} --episode_length=${episode_length} --file_name="example_${spec}" --save_results="True" --plot="False"
# done

# RUN REAL SIM
num_episodes=50000

## 1. Run pricing specs
python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=1000 --episode_length=${episode_length} --file_name="pricing_specs" --save_results="True" --plot="False"

# 2. No-Markov
# alpha_values=(0 0.1 0.25 0.5 0.75 0.9)
alpha_values=(0 0.1 0.5 0.9)

# for alpha in "${alpha_values[@]}"; do
#     prob=$(echo "scale=2; 1 - ${alpha}" | bc)  # scale=2 ensures 2 decimal places for better precision
#     prob=$(printf "%.2f" "$prob")  # This will ensure 2 decimal places with a trailing zero if needed

#     echo "prob value for alpha=${alpha}: ${prob}"

#     yq "( .variables.D.kernel.noise.prob) = ${prob}" \
#         config/pricing_specs.yaml > "config/pricing_no_markov_${alpha}.yaml"

#     echo "... running setting ${alpha}"

#     python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=10000 --episode_length=${episode_length} --file_name="pricing_no_markov_${alpha}" --save_results="True" --plot="False"

#     echo "... done with ${alpha}"
# done

# # 3. Dyn. back-door
# alpha_values=(0.1 0.5 1)

# for alpha in "${alpha_values[@]}"; do
#     yq "( .variables.L2.kernel.terms[0].param) = ${alpha}" \
#         config/pricing_specs.yaml > "config/pricing_dyn_back_door_${alpha}.yaml"

#     python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=5000 --episode_length=${episode_length} --file_name="pricing_dyn_back_door_${alpha}" --save_results="True" --plot="False"
# done

# 4. Confounding
alpha_values=(0 1 3 5)
probs_values=(1 0.9 0.5 0.1)

for i in "${!alpha_values[@]}"; do
    alpha=${alpha_values[i]}
    prob1=${probs_values[i]}

    prob2=$(printf "%.1f" "$(echo "1 - ${prob1}" | bc)")

    yq eval "( .variables.L1.kernel.indicator_terms[0].intercept) = ${alpha} |
             ( .variables.A1.kernel.mixed_probs) = [${prob1}, ${prob2}]" \
        config/pricing_specs.yaml > "config/pricing_confounded_${alpha}.yaml"

    python3 run.py --num_episodes=${num_episodes} --num_eval_episodes=5000 --episode_length=${episode_length} --file_name="pricing_confounded_${alpha}" --save_results="True" --plot="False"

done