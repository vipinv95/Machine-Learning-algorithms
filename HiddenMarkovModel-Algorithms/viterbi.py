from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
    """
    Forward algorithm
    
    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)
    
    Returns:
    - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
    """
    S = len(pi)
    N = len(O)
    alpha = np.zeros([S, N])
    for iter in range(N):
        for si in range(S):
            if iter == 0:
                alpha[si,iter] = pi[si]*B[si,O[iter]]
            else:
                alpha[si,iter] = sum([alpha[sj,iter-1]*A[sj,si]*B[si,O[iter]] for sj in range(S)])
        
    return alpha


def backward(pi, A, B, O):
    """
    Backward algorithm
    
    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)
    
    Returns:
    - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
    """
    S = len(pi)
    N = len(O)
    beta = np.zeros([S, N])
    last_ones = []
    def recback(iters,state):
        if iters == 0:
            return pi[state]*B[state,O[iters]]
        return sum([recback(iters-1,si)*A[si,state]*B[state,O[iters]] for si in range(S)])
    beta = sum([recback(N-1,si) for si in range(S)])
    return beta

def seqprob_forward(alpha):
    """
    Total probability of observing the whole sequence using the forward algorithm
    
    Inputs:
    - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
    
    Returns:
    - prob: A float number of P(x_1:x_T)
    """
    prob = 0
    S,N = alpha.shape
    prob = sum([alpha[s,N-1] for s in range(S)])
    return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  prob = float(beta)
  return prob

def viterbi(pi, A, B, O):
    """
    Viterbi algorithm
    
    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)
    
    Returns:
    - path: A list of the most likely hidden state path k* (in terms of the state index)
      argmax_k P(s_k1:s_kT | x_1:x_T)
    """
    path = []
    S = len(pi)
    N = len(O)
    max_state = np.zeros([S,N])
    for iter in range(N):
        for si in range(S):
            if iter == 0:
                max_state[si,iter] = pi[si]*B[si,O[iter]]
            else:
                max_state[si,iter] = max([max_state[sj,iter-1]*A[sj,si]*B[si,O[iter]] for sj in range(S)])
    path.append(np.argmax([max_state[s,N-1] for s in range(S)]))
    for iter in reversed(range(N)):
        if iter != 0:
            curr_state = path[len(path)-1]
            path.append(np.argmax([max_state[s,iter-1]*A[s,curr_state]*B[curr_state,O[iter]] for s in range(S)]))
        else:
            path = reversed(path)
    return path


def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
