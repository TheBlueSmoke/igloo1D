import sys
sys.path.append('C:\dev python\Tensor Flow code')

from datagen import *
from igloo1d import *


vector_size_inside=10000
vector_size_outside=10          ### DO NOT change
nb_features=1       ### how many features in the input data
vector_size_whole=vector_size_inside+2*vector_size_outside


nb_items=50000        ### number of items in the training set

batch_size=128
epochs=300



###used with IGLOO
nb_patches=6200

nb_filters_conv1d=5
padding_style="causal"

nb_stacks=1

mDR=0.1
m_add_residual=True



################################################################################
#### Generate Data
################################################################################

#get data in memory

ALL_DATA,ALL_LABELS=data_copymemory_classification(nb_items,vector_size_inside,nb_features,vector_size_whole,vector_size_outside,sparse=True,regenerate=True)


ALL_LABELS=to_categorical(ALL_LABELS, 8)

TRAIN_DATA,TEST_DATA,TRAIN_LABELS,TEST_LABELS = train_test_split(ALL_DATA,ALL_LABELS, train_size=0.9, random_state=211)   ##211


################################################################################
#### model startup
################################################################################


def get_model():

    input_layer = Input(name='input_layer', shape=(vector_size_whole,nb_features))



    x=IGLOO1D(input_layer,nb_patches,nb_filters_conv1d,return_sequences=True,padding_style=padding_style,nb_stacks=nb_stacks,add_residual=m_add_residual,nb_sequences=10,psy=0.1)



    x = Dense(8,activation='softmax')(x)



    model = Model(input_layer, x)

    madam = optimizers.Adam(lr=0.01, decay=0.005,clipnorm=1.0)

    model.compile( loss ='categorical_crossentropy', optimizer=madam,  metrics=['accuracy'])            ####  'MAE' , 'MSE'


    return model


#########################################################################
##### Running model
#########################################################################


PRE11 =CATEGORICALEVAL(validation_data=(TEST_DATA,TEST_LABELS))


train_new_model=True

save_model=False

model=get_model()
print(model.summary())

if train_new_model:
    print("fitting...")


    hist = model.fit(TRAIN_DATA,TRAIN_LABELS, batch_size=batch_size, epochs=epochs, validation_data=(TEST_DATA,TEST_LABELS), callbacks=[PRE11], verbose=1)
    if save_model:
        print("saving the model...")
        model.save('./models/nomatrix1.h5')  # creates a HDF5 file 'my_model.h5'
