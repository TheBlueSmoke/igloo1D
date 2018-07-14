import sys
sys.path.append('C:\dev python\Tensor Flow code')

from mandarin_common import *
from datagen import *


vector_size=1000
nb_items=25000

batch_size=100
epochs=300


LEARNING_RATE = 0.005

###used with IGLOO
nb_patches=2000
patch_size=4
CONV1D_dim=5
MAXPOOL_size=2
CONV1D_kernel=3

nb_stacks=3

padding_style="causal"


return_sequences=False

################################################################################
#### Generate Data
################################################################################

ALL_DATA,ALL_LABELS=data_addition(nb_items,vector_size,regenerate=True)

TRAIN_DATA,TEST_DATA,TRAIN_LABELS,TEST_LABELS = train_test_split(ALL_DATA,ALL_LABELS, train_size=0.9, random_state=211)   ##211


################################################################################
#### model startup
################################################################################

def get_model():
    ###input is a 200 long vectors of word indices
    input_layer = Input(shape=(vector_size,2,), name='AA_input')

    x=IGLOO1D(input_layer,nb_patches,CONV1D_dim,return_sequences=return_sequences,padding_style=padding_style,nb_stacks=nb_stacks,max_pooling_kernel=MAXPOOL_size)


    x = Dense(1)(x)
    x = Activation('linear', name='output_dense')(x)
    output_layer = x

    model = Model(input_layer, output_layer)
    adam = optimizers.Adam(lr=LEARNING_RATE, clipnorm=1.)
    model.compile(adam, loss='mean_squared_error')


    return model



#########################################################################
##### Running model
#########################################################################

PRE11_train = ADDITION_Evaluation(validation_data=(TRAIN_DATA,TRAIN_LABELS))
PRE11 = ADDITION_Evaluation(validation_data=(TEST_DATA,TEST_LABELS))
histo_call = LossHistory()

train_new_model=True

save_model=False

model=get_model()
print(model.summary())

if train_new_model:
    print("fitting...")
    history = model.fit(TRAIN_DATA,TRAIN_LABELS, batch_size=batch_size, epochs=epochs, validation_data=(TEST_DATA,TEST_LABELS), callbacks=[PRE11_train,PRE11,histo_call], verbose=1)
    print(history.history.keys())
