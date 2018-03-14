import numpy as np
num_observs = 10
num_states = 10
time_steps = 20

def get_prob(num_states):
	prob = np.random.rand(num_states)
	s = sum(prob)
	#normalising probabilities
	for i in range(len(prob)):
		prob[i] = prob[i]/s
	prob = map(float,list(prob))
	return prob

#get initial probabilities
initial_prob = get_prob(num_states)

#get transition probabilities
transition_prob = [get_prob(num_states) for i in range(num_states)]

#observation sequence
obs_seq = [int(np.random.randint(num_observs)) for _ in range(time_steps)]

#define a state class
class state:
	def __init__(self,iden):
		self.iden = iden
		self.obsProbs = map(float,list(np.random.rand(num_observs)))
		self.score = 0
		self.state_seq = []

#get list of states
state_list = [state(str(i)) for i in range(num_states)]

#function to initialize states 
def init_state(state):
	index = int(state.iden)
	state.score = initial_prob[index]*state.obsProbs[obs_seq[0]]
	state.state_seq = [state]

#initialize all the states
for i in state_list:
	init_state(i)

def maxIndex(l):
	return l.index(max(l))

#the main viterbi algorithm
for i in range(time_steps):
	for j in range(num_states):
		m = [(state_list[k].score*transition_prob[k][j]) for k in range(num_states)]
		maximum = max(m)
		index = maxIndex(m)
		state_list[j].score = maximum*(state_list[j].obsProbs[obs_seq[i]])
		state_list[j].state_seq = [state_list[index]] + state_list[j].state_seq

temp = [state_list[i].score for i in range(num_states)]
#find max score
max_prob = max(temp)

#find best sequence
l = state_list[maxIndex(temp)].state_seq
l.reverse()
optimal_seq = [i.iden for i in l]

print("Max_prob: " + str(max_prob) + "\n")
print("Optimal sequence of states: " + str(optimal_seq) + "\n")
print("Given observation sequence: " + str(obs_seq) + "\n")
print("Initial probabilities: " + str(initial_prob) + "\n") 
print("Transition probabilities: " + str(transition_prob) + "\n")

