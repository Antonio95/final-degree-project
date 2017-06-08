
VERBOSE = True

# Training
N_STEPS = 5000
PRINT_FREQ = 100
KEEP_PROB = 0.5
BATCH_SIZE = 50

# Logs
SUMMARY_DIR = 'tensorboard_logs'

# Save & restore
MODEL_DIR = 'models'

# Save & restore
NETWORK = 'vishnet'

# Hypertuning
HYPER_ID = 'vishnet_cat'
RESULTS_DIR = 'results'
SWEEP_LAMBDAS = [0.0000003, 0.0000003, 0.0000003, 0.0000003]
SWEEP_KEEP_RATES = [0.4]


# Utility function to keep several execution configurations separate
# (when working in a queue system, the settings file is read only 
# when the job enters the queue)
def override_settings(conf):

    global N_STEPS, NETWORK, HYPER_ID, SWEEP_LAMBDAS, SWEEP_KEEP_RATES


    if conf == 'test':
        N_STEPS = 100
        NETWORK = 'lenet'
        HYPER_ID = 'lenet_test'
        SWEEP_LAMBDAS = [0.004, 0.004]
        SWEEP_KEEP_RATES = [0.5]
    elif conf == 'l60':
        N_STEPS = 60000
        NETWORK = 'lenet'
        HYPER_ID = 'lenet_60000'
        SWEEP_LAMBDAS = [0.004, 0.004, 0.004, 0.004]
        SWEEP_KEEP_RATES = [0.5]
    elif conf == 'l80':
        N_STEPS = 80000
        NETWORK = 'lenet'
        HYPER_ID = 'lenet_80000'
        SWEEP_LAMBDAS = [0.004, 0.004, 0.004, 0.004]
        SWEEP_KEEP_RATES = [0.5]
    elif conf == 't10':
        N_STEPS = 10000
        NETWORK = 'tfnet'
        HYPER_ID = 'tfnet_10000'
        SWEEP_LAMBDAS = [0.00005, 0.00005, 0.00005, 0.00005]
    elif conf == 't20':
        N_STEPS = 20000
        NETWORK = 'tfnet'
        HYPER_ID = 'tfnet_20000'
        SWEEP_LAMBDAS = [0.00005, 0.00005, 0.00005, 0.00005]
        SWEEP_KEEP_RATES = [0.75]
    elif conf == 't40':
        N_STEPS = 40000
        NETWORK = 'tfnet'
        HYPER_ID = 'tfnet_40000'
        SWEEP_LAMBDAS =	[0.00005, 0.00005, 0.00005, 0.00005]
        SWEEP_KEEP_RATES = [0.75]
    elif conf == 't60':
        N_STEPS = 60000
        NETWORK = 'tfnet'
        HYPER_ID = 'tfnet_60000'
        SWEEP_LAMBDAS = [0.00005, 0.00005, 0.00005, 0.00005]
    elif conf == 't80':
        N_STEPS = 80000
        NETWORK = 'tfnet'
        HYPER_ID = 'tfnet_80000'
        SWEEP_LAMBDAS = [0.00005, 0.00005, 0.00005, 0.00005]
    elif conf == 'v5':
        N_STEPS = 5000
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_5000'
        SWEEP_LAMBDAS = [0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    elif conf == 'v5':
        N_STEPS = 5000
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_5000'
        SWEEP_LAMBDAS = [0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    elif conf == 'v10':
        N_STEPS = 10000
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_10000'
        SWEEP_LAMBDAS =	[0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    elif conf == 'v20':
        N_STEPS = 20000 
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_20000'
        SWEEP_LAMBDAS =	[0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    elif conf == 'v40':
        N_STEPS = 40000 
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_40000'
        SWEEP_LAMBDAS =	[0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    if conf == 'v60':
        N_STEPS = 60000
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_60000'
        SWEEP_LAMBDAS = [0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]
    if conf == 'v80':
        N_STEPS = 80000
        NETWORK = 'vishnet'
        HYPER_ID = 'vishnet_80000'
        SWEEP_LAMBDAS = [0.0000003, 0.0000003, 0.0000003, 0.0000003]
        SWEEP_KEEP_RATES = [0.4]

