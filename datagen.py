from igloo1d import *

def data_copymemory_regressionmulti(nb_items,vector_size_inside,nb_features,vector_size_whole,vector_size_outside):

    nb_classes=8

    ALL_DATA=np.empty((nb_items,vector_size_whole,nb_features))
    ALL_LABELS=np.empty((nb_items,10,nb_features))

    for elem in range(nb_items):

        BLOCK_LEFT=[]
        BLOCK_MIDDLE=[]
        BLOCK_RIGHT=[]

        for feature in range(nb_features):
            LEFT=np.random.randint(low=0,high=8, size=vector_size_outside)
            MIDDLE=8*np.ones((vector_size_inside-1))
            RIGHT=9*np.ones((vector_size_outside+1))

            BLOCK_LEFT.append(LEFT)
            BLOCK_MIDDLE.append(MIDDLE)
            BLOCK_RIGHT.append(RIGHT)


        BLOCK_LEFT=np.stack(BLOCK_LEFT,axis=-1)
        BLOCK_MIDDLE=np.stack(BLOCK_MIDDLE,axis=-1)
        BLOCK_RIGHT=np.stack(BLOCK_RIGHT,axis=-1)


        ALL_DATA[elem]=np.concatenate((BLOCK_LEFT,BLOCK_MIDDLE,BLOCK_RIGHT))

        ALL_LABELS[elem]=BLOCK_LEFT
        #ALL_LABELS[elem]=np.concatenate((np.zeros((vector_size_outside+vector_size_inside,nb_features)),BLOCK_LEFT))

    #ALL_DATA=np.expand_dims(ALL_DATA,axis=-1)
    #ALL_LABELS=np.expand_dims(ALL_LABELS,axis=-1)

    print("ALL_DATA.shape",ALL_DATA.shape)
    print("ALL_LABELS.shape",ALL_LABELS.shape)

    print(ALL_LABELS[0])



    return ALL_DATA,ALL_LABELS


def data_copymemory_classification(nb_items,vector_size_inside,nb_features,vector_size_whole,vector_size_outside,sparse=False,regenerate=False):



    nb_classes=8

    ALL_DATA=np.empty((nb_items,vector_size_whole,nb_features))

    if regenerate:

        if not sparse:
            ALL_LABELS=np.empty((nb_items,10,nb_classes))
        else:
            ALL_LABELS=np.empty((nb_items,10,1))

        for elem in range(nb_items):

            BLOCK_LEFT=[]
            BLOCK_MIDDLE=[]
            BLOCK_RIGHT=[]

            for feature in range(nb_features):
                LEFT=np.random.randint(low=0,high=8, size=vector_size_outside)
                MIDDLE=8*np.ones((vector_size_inside-1))
                RIGHT=9*np.ones((vector_size_outside+1))

                BLOCK_LEFT.append(LEFT)
                BLOCK_MIDDLE.append(MIDDLE)
                BLOCK_RIGHT.append(RIGHT)


            BLOCK_LEFT=np.stack(BLOCK_LEFT,axis=-1)
            BLOCK_MIDDLE=np.stack(BLOCK_MIDDLE,axis=-1)
            BLOCK_RIGHT=np.stack(BLOCK_RIGHT,axis=-1)


            ALL_DATA[elem]=np.concatenate((BLOCK_LEFT,BLOCK_MIDDLE,BLOCK_RIGHT))

            if sparse==False:
                qq=to_categorical(LEFT, num_classes=nb_classes)
                ALL_LABELS[elem]=qq
            else:
                ALL_LABELS[elem]=BLOCK_LEFT


        print("ALL_DATA.shape",ALL_DATA.shape)
        print("ALL_LABELS.shape",ALL_LABELS.shape)

        with open('./data/copymemory_classification1_{}.pickle'.format(vector_size_inside), 'wb') as handle:
            pickle.dump((ALL_DATA,ALL_LABELS), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("FULL DATA copy memory saved...")

        return ALL_DATA, ALL_LABELS

    else:

        ####loading this
        with open('./data/copymemory_classification1_{}.pickle'.format(vector_size_inside), 'rb') as handle:
            ALL_DATA,ALL_LABELS= pickle.load(handle)


        print("FULL DATA copy memory IMPORTED:---->",'./data/copymemory_classification1_{}.pickle'.format(vector_size_inside))
        return ALL_DATA, ALL_LABELS


def sample_vec(num_channels_input):
    zz=np.random.normal(loc=0.0, scale=1.0, size=(num_channels_input))
    max_lower_vec=np.max(zz[:int(num_channels_input/2)])
    max_higher_vec=np.max(zz[int(num_channels_input/2):])

    if max_lower_vec<max_higher_vec :
        return 1,zz
    else:
        return 0,zz


def data_findmax(nb_items,vector_size,nb_classes,num_channels_input,regenerate=False):


    #### generate random number which is gonna be the category to find
    ALL_DATA_INT=[]
    ALL_DATA=[]
    classes_vec=[]

    if regenerate:

        for qq in range(nb_items):

            if qq % 1000==0:
                print("step-->",qq)

            class_number=np.random.choice(range(nb_classes),1, replace=True)[0]

            classes_vec.append(class_number)

            ### generate the indices where the max on top 50% will be higher than max lower50%

            steps_indices=np.random.choice(range(vector_size),class_number, replace=False)      ##no repetition

            ALL_DATA_INT=[]

            for stepo in range(vector_size):






                if (stepo in steps_indices):
                    score=0
                    while score==0:
                        score,current_vec=sample_vec(num_channels_input)


                    ALL_DATA_INT.append(current_vec)

                else:

                    score=1
                    while score==1:
                        score,current_vec=sample_vec(num_channels_input)

                    ALL_DATA_INT.append(current_vec)

            bb=np.stack(ALL_DATA_INT)
            ALL_DATA.append(bb)


        ALL_DATA=np.stack(ALL_DATA)



        print("ALL_DATA.shape",ALL_DATA.shape)

        ALL_LABELS=to_categorical(classes_vec, num_classes=nb_classes)


        with open('./data/findmax1.pickle', 'wb') as handle:
            pickle.dump((ALL_DATA,ALL_LABELS), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("FULL DATA saved...")

        return ALL_DATA, ALL_LABELS

    else:

        ####loading this
        with open('./data/findmax1.pickle', 'rb') as handle:
            ALL_DATA,ALL_LABELS= pickle.load(handle)


        print("FULL DATA IMPORTED")
        return ALL_DATA, ALL_LABELS



def data_addition(nb_items,vector_size, regenerate=False):

    if regenerate:
        ALL_DATA=np.random.uniform(size=(nb_items, vector_size,1))             ### 100000,100,1
        ALL_DATA=np.concatenate( (ALL_DATA, np.zeros((nb_items, vector_size,1))  ) ,axis=2)  ### 100000,100,2

        print("ALL_DATA.shape",ALL_DATA.shape)

        ### choose two different indices at random and put a 1 there
        ### for each line
        for jj in range(nb_items):
            a = np.arange(vector_size)
            np.random.shuffle(a)
            rand_numpy=a[:2]

            ALL_DATA[jj,rand_numpy,1]=1

        #print(ALL_DATA[0])


        ALL_LABELS=[]
        for jj in range(nb_items):
            current_block=ALL_DATA[jj]
            first=current_block[:,0]
            second=current_block[:,1]
            prod=first*second
            sum=np.sum(prod)
            ALL_LABELS.append(sum)

        print("ALL_DATA.shape",ALL_DATA.shape)


        with open('./data/data_addition1_{}.pickle'.format(vector_size), 'wb') as handle:
            pickle.dump((ALL_DATA,ALL_LABELS), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("FULL DATA copy memory saved...")

        return ALL_DATA, ALL_LABELS

    else:

        ####loading this
        with open('./data/data_addition1_{}.pickle'.format(vector_size), 'rb') as handle:
            ALL_DATA,ALL_LABELS= pickle.load(handle)


        print("FULL DATA copy memory IMPORTED")
        return ALL_DATA, ALL_LABELS



def data_msearranged(nb_items,vector_size,diff_to_predict,num_channels_input,regenerate=False):

    randy=vector_size*np.ones((nb_items),dtype=np.int32)

    ALL_DATA=[]

    if regenerate:

        for ii in range(nb_items):
            FIRST=np.random.uniform(low=-10, high=np.random.randint(low=1,high=20, size=1),size=(vector_size,num_channels_input))

            ALL_DATA.append(FIRST)

        ALL_DATA=np.stack(ALL_DATA)

        print("ALL_DATA.shape",ALL_DATA.shape)

        SUMS=np.sum(ALL_DATA,axis=-1)

        print("SUMS.shape",SUMS.shape)      ### SUMS.shape (3000, 320)

        ALL_LABELS=np.zeros((nb_items,vector_size,diff_to_predict))

        for kk in range(nb_items):
            for jj in range(vector_size):
                for ss in range(diff_to_predict):
                    ALL_LABELS[kk,jj, ss]=SUMS[kk,jj]-SUMS[kk,max(jj-ss-1,0)]-SUMS[kk,0]


        print("ALL_LABELS.shape",ALL_LABELS.shape)


        with open('./data/msearranged_{}_{}.pickle'.format(vector_size,num_channels_input), 'wb') as handle:
            pickle.dump((ALL_DATA,ALL_LABELS), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("FULL DATA saved for msearranged...")



    else:

        ####loading this
        with open('./data/msearranged_{}_{}.pickle'.format(vector_size,num_channels_input), 'rb') as handle:
            ALL_DATA,ALL_LABELS= pickle.load(handle)


        print("FULL DATA IMPORTED for msearranged...",'./data/msearranged_{}_{}.pickle'.format(vector_size,num_channels_input))


    return ALL_DATA,ALL_LABELS


def data_msearranged_masking(nb_items,vector_size,diff_to_predict,num_channels_input):

    ###drawing the seq-lens : np.random.randint(5, size=10)  sampling [0..4]

#    upper_limit=vector_size
    upper_limit=30

    seqlen=np.random.randint(5,upper_limit, size=nb_items)


    randy=vector_size*np.ones((nb_items),dtype=np.int32)

    ALL_DATA=[]

    for ii in range(nb_items):
        FIRST=np.random.uniform(low=-10, high=np.random.randint(low=1,high=20, size=1),size=(vector_size,num_channels_input))

        ###fill anything post seqlen with zeros

        FIRST[seqlen[ii]:,:]=0


        ALL_DATA.append(FIRST)

    ALL_DATA=np.stack(ALL_DATA)

    print("ALL_DATA.shape",ALL_DATA.shape)

    SUMS=np.sum(ALL_DATA,axis=-1)

    print("SUMS.shape",SUMS.shape)      ### SUMS.shape (3000, 320)

    ALL_LABELS=np.zeros((nb_items,vector_size,diff_to_predict))

    for kk in range(nb_items):
        for jj in range(vector_size):
            for ss in range(diff_to_predict):
                ALL_LABELS[kk,jj, ss]=SUMS[kk,jj]-SUMS[kk,max(jj-ss-1,0)]-SUMS[kk,0]

    ####masking the LABELS
    for kk in range(nb_items):
        ALL_LABELS[kk,seqlen[kk]:,:]=0



    print("ALL_LABELS.shape",ALL_LABELS.shape)

    return ALL_DATA,ALL_LABELS,seqlen


def data_msearranged_masking_nonesize(nb_items,vector_size,diff_to_predict,num_channels_input):

#    upper_limit=vector_size
    upper_limit=30

    seqlen=np.random.randint(5,upper_limit, size=nb_items)


    randy=vector_size*np.ones((nb_items),dtype=np.int32)

    ALL_DATA=[]

    for ii in range(nb_items):
        FIRST=np.random.uniform(low=-10, high=np.random.randint(low=1,high=20, size=1),size=(vector_size,seqlen[ii]))

        ALL_DATA.append(FIRST)

#    ALL_DATA=np.stack(ALL_DATA)

    SUMS=[np.sum(xxx,axis=-1) for xxx in ALL_DATA]

    ALL_LABELS=[]

    for kk in range(nb_items):
        LABELS=np.zeros((seqlen[kk],diff_to_predict))
        for jj in range(seqlen[kk]):
            for ss in range(diff_to_predict):


                LABELS[jj, ss]=SUMS[kk][jj]-SUMS[kk][max(jj-ss-1,0)]-SUMS[kk][0]


        ALL_LABELS.append(LABELS)


    return ALL_DATA,ALL_LABELS,seqlen


#####################################################################################################
#### GRAMAR TASK generation, from
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Train recurrent neural network
# License: For legal information see LICENSE in the home directory.
#####################################################################################################

# Reber grammar
#
states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
transitions = {
	1: [2, 7],
	2: [3, 4],
	3: [3, 4],
	4: [5, 6],
	5: [12],
	6: [8, 9],
	7: [8, 9],
	8: [8, 9],
	9: [10, 11],
	10: [5, 6],
	11: [12],
}
aliases = {
	1: 'B', 2: 'T', 3: 'S', 4: 'X', 5: 'S', 6: 'X',
	7: 'P', 8: 'T', 9: 'V', 10: 'P', 11: 'V', 12: 'E',
}
encoding = {'B': 0, 'E': 1, 'P': 2, 'S': 3, 'T': 4, 'V': 5, 'X': 6}

# Data dimensions
#


##########################################################################################
# Utilities
##########################################################################################

def make_chain():
	chain = [1]
	while chain[-1] != states[-1]:
		choices = transitions[chain[-1]]
		j = np.random.randint(len(choices))
		chain.append(choices[j])
	return chain

def valid_chain(chain):
	if len(chain) == 0:
		return False
	if chain[0] != states[0]:
		return False
	for i in range(1, len(chain)):
		if chain[i] not in transitions[chain[i-1]]:
			return False
	return True

def convert_chain(chain):
	sequence = ''
	for value in chain:
		sequence += aliases[value]
	return sequence

def data_grammar(nb_items,vector_size,num_channels_input,regenerate=False):

    if regenerate:

        xs_train = np.zeros((nb_items, vector_size, num_channels_input))
        ls_train = np.zeros(nb_items)
        ys_train = np.zeros(nb_items)

        for i in range(nb_items):

        	chain = make_chain()
        	valid = 1.0
        	if np.random.rand() >= 0.5:	# Randomly insert a single typo with proability 0.5
        		hybrid = chain
        		while valid_chain(hybrid):
        			chain_ = make_chain()
        			j = np.random.randint(len(chain))
        			j_ = np.random.randint(len(chain_))
        			hybrid = chain[:j]+chain_[j_:]
        		chain = hybrid
        		valid = 0.0
        	sequence = convert_chain(chain)
        	for j, symbol in enumerate(sequence):
        		k = encoding[sequence[j]]
        		xs_train[i,j,k] = 1.0
        	ls_train[i] = len(sequence)
        	ys_train[i] = valid




        with open('./data/grammar_{}_{}.pickle'.format(vector_size,num_channels_input), 'wb') as handle:
            pickle.dump((xs_train,ys_train), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("FULL DATA saved for grammar...")

        return xs_train,ys_train

    else:

        ####loading this
        with open('./data/grammar_{}_{}.pickle'.format(vector_size,num_channels_input), 'rb') as handle:
            ALL_DATA,ALL_LABELS= pickle.load(handle)


        print("FULL DATA IMPORTED for GRAMMAR...",'./data/grammar_{}_{}.pickle'.format(vector_size,num_channels_input))


        return ALL_DATA,ALL_LABELS
