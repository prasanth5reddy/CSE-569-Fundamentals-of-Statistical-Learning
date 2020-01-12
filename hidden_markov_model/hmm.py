import numpy as np


def initialise_hmm_parameters():
    # parameters for HMM 1
    prior_matrix_1 = np.array([0, 1, 0, 0])
    transition_matrix_1 = np.array([[1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0.4, 0.3, 0.3],
                                    [0.3, 0.2, 0.2, 0.3]])
    emission_matrix_1 = np.array([[1, 0, 0, 0, 0],
                                  [0, 0.5, 0.5, 0, 0],
                                  [0, 0.2, 0.2, 0.3, 0.3],
                                  [0, 0, 0, 0.5, 0.5]])
    hmm_parameters_1 = {'P': prior_matrix_1, 'A': transition_matrix_1, 'B': emission_matrix_1}
    # parameters for HMM 2
    prior_matrix_2 = np.array([0, 0, 0, 1])
    transition_matrix_2 = np.array([[1, 0, 0, 0],
                                    [0.1, 0.3, 0.5, 0.1],
                                    [0.1, 0.4, 0.3, 0.2],
                                    [0.1, 0.4, 0.2, 0.3]])
    emission_matrix_2 = np.array([[1, 0, 0, 0, 0],
                                  [0, 0, 0.5, 0, 0.5],
                                  [0, 0, 0.5, 0.5, 0],
                                  [0, 0.5, 0, 0, 0.5]])
    hmm_parameters_2 = {'P': prior_matrix_2, 'A': transition_matrix_2, 'B': emission_matrix_2}
    return hmm_parameters_1, hmm_parameters_2


def generate_sequence(hmm):
    # dictionary to map emission matrix indices to observations
    observation_map = {0: 'S', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    # initialize empty list of sequence
    sequence = []
    # generate a random start state based on prior matrix. Note that here state numbers start from 0
    state = np.random.choice(hmm['P'].shape[0], p=hmm['P'])
    while True:
        # generate random observation index based on the previous state
        observation_index = np.random.choice(hmm['B'].shape[1], p=hmm['B'][state])
        # get observation from observation_index using observation_map
        observation = observation_map[observation_index]
        # add observation to the sequence
        sequence.append(observation)
        # break the loop if we reached stop symbol 'S'
        if observation == 'S':
            break
        # generate random next state based on the previous state
        state = np.random.choice(hmm['A'].shape[1], p=hmm['A'][state])
    return sequence


def initialize_sequences():
    no_of_seq = 10
    seq_list = [] * no_of_seq
    seq_list.append(['A', 'D', 'C', 'B', 'D', 'C', 'C', 'S'])
    seq_list.append(['B', 'D', 'S'])
    seq_list.append(['B', 'C', 'C', 'B', 'D', 'D', 'C', 'A', 'C', 'S'])
    seq_list.append(['A', 'C', 'D', 'S'])
    seq_list.append(['A', 'D', 'A', 'C', 'S'])
    seq_list.append(['D', 'B', 'B', 'S'])
    seq_list.append(['A', 'B', 'S'])
    seq_list.append(['D', 'D', 'B', 'D', 'D', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'C', 'D', 'B', 'B', 'B', 'S'])
    seq_list.append(['D', 'B', 'D', 'S'])
    seq_list.append(['A', 'A', 'A', 'A', 'D', 'C', 'B', 'S'])
    return seq_list


def forward(sequence, hmm):
    # dictionary to map observations to emission matrix indices
    observation_map = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    # get total number of states from prior matrix
    no_of_states = hmm['P'].shape[0]
    # initialise alpha and alpha_prev to zeros and size equal to number of states
    alpha = np.zeros(no_of_states)
    alpha_prev = np.zeros(no_of_states)

    for t in range(len(sequence)):
        if t == 0:
            # Here t == 0 means first observation and time = 1
            # get alpha using prior and emission matrix
            alpha = hmm['P'] * hmm['B'][:, observation_map[sequence[t]]]
        else:
            # get alpha using alpha_prev, emission and transition matrices
            alpha = hmm['B'][:, observation_map[sequence[t]]] * np.sum(np.transpose(hmm['A']) * alpha_prev, axis=1)
        # update alpha_prev with current alpha
        alpha_prev = np.copy(alpha)

    return np.sum(alpha)


def viterbi(sequence, hmm):
    # dictionary to map observations to emission matrix indices
    observation_map = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    # get total number of states from prior matrix
    no_of_states = hmm['P'].shape[0]
    # initialise delta and psi matrices with size no_of_states x length of sequence (N x T)
    delta = np.zeros((no_of_states, len(sequence)))
    psi = np.zeros((no_of_states, len(sequence)), dtype=np.int)

    for t in range(len(sequence)):
        # iterate through each observation in the sequence
        if t == 0:
            # Here t == 0 means first observation and time = 1
            # get delta for all states at time = 1 from prior and emission matrix
            delta[:, t] = hmm['P'] * hmm['B'][:, observation_map[sequence[t]]]
        else:
            # get delta for all states at time = t other than 1
            # from delta at time = t - 1, emission and transition matrices
            delta[:, t] = hmm['B'][:, observation_map[sequence[t]]] * np.max(np.transpose(hmm['A']) * delta[:, t - 1],
                                                                             axis=1)
            # get psi for all states at time = t other than 1
            # from delta at time = t - 1 and transition matrices
            psi[:, t] = np.argmax(np.transpose(hmm['A']) * delta[:, t - 1], axis=1)

    # initialize hidden state with zeros of size equal to length of sequence
    states = np.zeros(len(sequence), dtype=np.int)
    for t in range(len(sequence) - 1, -1, -1):
        # backtrack from reverse to get hidden states
        if t == len(sequence) - 1:
            # get last state from delta at time = T
            states[t] = np.argmax(delta[:, t])
        else:
            # get previous states from psi and state at next matrix
            states[t] = psi[states[t + 1]][t + 1]

    return states


def main():
    # generate HMMs from given parameters
    hmm_1, hmm_2 = initialise_hmm_parameters()
    # set up random seed for reproducibility
    np.random.seed(123)

    # a) generate 10 sequences using HMM 1
    no_of_sequences = 10
    print(f'Generated {no_of_sequences} sequences')
    for i in range(no_of_sequences):
        # generate a sequence of observation from HMM 1
        seq = generate_sequence(hmm_1)
        print(f'{i + 1}. {seq}')

    # b) classify sequences based on forward algorithm
    # initialize given sequences
    sequences_list = initialize_sequences()
    print('Classification of given sequences')
    for i in range(len(sequences_list)):
        # evaluate sequence with HMM 1
        evaluation_1 = forward(sequences_list[i], hmm_1)
        # evaluate sequence with HMM 2
        evaluation_2 = forward(sequences_list[i], hmm_2)
        # classify the sequences based on evaluation scores
        # print(evaluation_1, evaluation_2)
        if evaluation_1 > evaluation_2:
            print(f'{sequences_list[i]} : HMM 1')
        else:
            print(f'{sequences_list[i]} : HMM 2')

    # c) decode hidden states for the sequences using HMM 2
    print('Decoded hidden states for given sequences')
    for i in range(len(sequences_list)):
        hidden_states = viterbi(sequences_list[i], hmm_2)
        print(f'{sequences_list[i]} : {list(hidden_states + 1)}')

    # verification
    # print(np.log(forward(['A', 'C', 'D', 'D', 'C', 'C', 'S'], hmm_1)))
    # print(np.log(forward(['A', 'C', 'D', 'D', 'C', 'C', 'S'], hmm_2)))
    # print(viterbi(['A', 'C', 'D', 'D', 'C', 'C', 'S'], hmm_1) + 1)
    # print(viterbi(['A', 'C', 'D', 'D', 'C', 'C', 'S'], hmm_2) + 1)
    #
    # print(np.log(forward(['A', 'D', 'S'], hmm_1)))
    # print(np.log(forward(['A', 'D', 'S'], hmm_2)))
    # print(viterbi(['A', 'D', 'S'], hmm_1) + 1)
    # print(viterbi(['A', 'D', 'S'], hmm_2) + 1)


if __name__ == '__main__':
    main()
