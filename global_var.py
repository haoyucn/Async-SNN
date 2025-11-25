global isTraining
isTraining = False

global SIG_TYPES
SIG_TYPES = {'regular': 1, 'ISI': 2, 'FBNSI': 3}

global OUTPUT_MODE
OUTPUT_MODE = {'accumulative_pos_compare', 'accumulative', 'final_reading', 'acculuative_final_reading', 'accumulative_final_pos_compare', 'final_last_reading'}

global LOG
LOG = True

def sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    
    return 0


global PRINT_FIRING_POSITIVE
PRINT_FIRING_POSITIVE = True

global NETWORK_FULLY_CONNECTED
NETWORK_FULLY_CONNECTED = False

global LEARNING_RATE_ADOPTIVE
LEARNING_RATE_ADOPTIVE = 1