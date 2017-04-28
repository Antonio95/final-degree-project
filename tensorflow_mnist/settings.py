
VERBOSE = True

# Training
N_STEPS = 10000
PRINT_FREQ = 10
KEEP_PROB = 0.5
BATCH_SIZE = 50

# Logs
SUMMARY_DIR = 'tensorboard_logs'

# Save & restore
MODEL_DIR = 'models'

# Save & restore
NETWORK = 'mejnet'

# Hypertuning
HYPER_ID = 'broad_mejnet'
RESULTS_DIR = 'results'
SWEEP_LAMBDAS = [0.000003, 0.000007, 0.00001, 0.00003, 0.00007]
SWEEP_KEEP_RATES = [0.25, 0.4, 0.5, 0.6, 0.75]
