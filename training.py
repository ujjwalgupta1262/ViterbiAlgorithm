from viterbi import decode

def get_data(filename):
	temp = open(filename,"r")
	l = temp.readlines()
	for i in xrange(len(l)):
		l[i] = l[i].strip().split()
	return l

def get_seq(l):
	list_seq = []
	temp = []
	for i in l:
		if(i == []):
			list_seq.append(temp)
			temp = []
		else:
			temp.append(i)
	return list_seq

def process_seq(list_seq):
	#mapping obs to integers
	dict_obs = {}
	#mapping tags to states
	dict_states = {}
	#mapping from states to dict of transitions
	dict_trans = {}
	#mapping from states to observations from state
	dict_obsInState = {}
	#mapping from states to number of times it was initial state
	dict_initStates = {}

	for seq in list_seq:
		for i in xrange(len(seq)):
			if seq[i][0] not in dict_obs:
				dict_obs[seq[i][0]] = len(dict_obs)
			
			if seq[i][1] not in dict_states:
				dict_states[seq[i][1]] = len(dict_states)

			if i != 0:
				prev_state = dict_states[seq[i-1][1]]
				next_state = dict_states[seq[i][1]]
				if(prev_state not in dict_trans):
					dict_trans[prev_state] = {}
				if(next_state not in dict_trans[prev_state]):
					dict_trans[prev_state][next_state] = 1
				else:
					dict_trans[prev_state][next_state] += 1

			current_state = dict_states[seq[i][1]]
			current_obs = dict_obs[seq[i][0]]
			if(current_state not in dict_obsInState):
				dict_obsInState[current_state] = {}
			if(current_obs not in dict_obsInState[current_state]):
				dict_obsInState[current_state][current_obs] = 1
			else:
				dict_obsInState[current_state][current_obs] += 1

		if(dict_states[seq[0][1]] not in dict_initStates):
			dict_initStates[dict_states[seq[0][1]]] = 1
		else:
			dict_initStates[dict_states[seq[0][1]]] += 1 

	return [dict_obs,dict_states,dict_trans,dict_obsInState,dict_initStates]

def get_initial_probs(dict_initStates, num_states):
	l = [0]*num_states
	s = sum(dict_initStates.values())
	for i in dict_initStates:
		l[i] = float(dict_initStates[i])/float(s)
	return l

def normalise(dict, length):
	l = [0]*length
	s = sum(dict.values())
	for i in sorted(dict):
		l[i] = float(dict[i])/float(s)
	return l

def get_trans_probs(dict_trans,num_states):
	trans_probs = []
	for i in xrange(num_states):
		trans_probs.append([0]*num_states)
	for i in xrange(num_states):
		trans_probs[i] = normalise(dict_trans[i],num_states)
	return trans_probs

def get_obs_probs(dict_obsInState, num_states, num_obs):
	obs_probs = []
	for i in xrange(num_states):
		obs_probs.append([0]*num_obs)
	for i in xrange(num_states):
		obs_probs[i] = normalise(dict_obsInState[i],num_obs)
	return obs_probs


l = get_data("train.txt")
list_seq = get_seq(l)
a = process_seq(list_seq)
first_seq = list_seq[0]

init_probs = get_initial_probs(a[-1],len(a[1]))
trans_probs = get_trans_probs(a[2],len(a[1]))
obs_probs = get_obs_probs(a[-2],len(a[1]),len(a[0]))
num_states = len(a[1])
num_observs = len(a[0]) 
obs_seq = [a[0][i[0]] for i in first_seq]
time_steps = len(obs_seq)
l = decode(num_observs,num_states,time_steps,init_probs,trans_probs,obs_probs,obs_seq)


