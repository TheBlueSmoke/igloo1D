# IGLOO1D
Implementation of IGLOO for sequences, based on the paper "IGLOO: Slicing the Feature Space to Represent Long Sequences".


# Basic Usage

from igloo1d import *

1-a returning a golbal representation (of size L=1000) of a sequence by slicing a feature space of size K=50:

```
x=IGLOO1D(y,nb_patches=L,nb_filters_conv1d=K,return_sequences=False)
```

1-b returning a sequence of the same length as the input sequence by slicing a feature space of size K=50 (using L=50):

```
x=IGLOO1D(y,nb_patches=L,nb_filters_conv1d=K,return_sequences=True)
```

1-b returning the last 10 elements of a sequence (used in the copy memory task):

```
x=IGLOO1D(y,nb_patches=L,nb_filters_conv1d=K,return_sequences=True,nb_sequences=10)
```

live example in "N_copymemory_classification_igloo.py"


# Advanced Usage

```
IGLOO1D(input_layer,nb_patches,nb_filters_conv1d,return_sequences,patch_size=4,
        padding_style="causal",stretch_factor=1,nb_stacks=1,l2reg=0.00001,conv1d_kernel=3,
        max_pooling_kernel=1,DR=0.0,add_residual=True,nb_sequences=-1,build_backbone=False,psy=0.15)
```

**input_layer**:                        Keras layer used as input.  
**nb_patches**:                         Number L of patches taken at random from the feature space. This is the main dimension                                                   parameter. A fair value to use is around the number of steps in the sequence to study if                                                 return_sequences=False, or can use 30 to 70 if return_sequences=True.  
**nb_filters_conv1d**:                  Size of the internal convolution K whc\ich transform the input_layer  
**return_sequences**:                   False to return only the full sequence representation  
**patch_size**:                         This is the number of slices taken to form a patch .Typical value is 4. Adding more increases                                           fitting and adds the number of parameters.  
**padding_style**:                      "causal", "same"  
**stretch_factor**:                     Stretch factor from the paper. When returning a full sequence this allows to reuse weights. The                                         stretch factor should divide the number of steps exactly. A value around 10-20 usualy works and                                         allows to divide the number of parameters by as much.  
**nb_stacks**:                          Number of levels of granularity that IGLOO will consider. More stacks increases accuracy but                                             also the number of parameters. Most of the time 1 stack should be enough, unless the number of                                           paramaters is low to start with. Setting more than 1 stack should be used only when                                                     return_sequences=False, otherwise the number of paramters will turn out to be too large.  
**l2_reg**:                             L2 regularization factor. Can use a value around 0.1 to combat over fitting.  
**conv1d_kernel**:                      Kernel of the initial convolution. Few reasons to change that.  
**max_pooling_kernel**                  Only when return_sequences=False, this allows to reduce the number of steps and therefore the                                           number of paramters. Some tasks work well with this. A typical value can be 2 to6.  
**DR**                                  Dropout rate.  
**add_residual**                        If return_sequences=True, this improves convergence and generally should be set to True. If                                             return_sequences=False, this has no effect.  
**nb_sequences**                        If return_sequences=True, this allows to return the last "nb_sequences" steps of the sequences 
                                        (For the copy memory task for example).  
**build_backbone**                      When the number of patches is large, IGLOO can use some patches arranged in a non random way to                                         better cover the input space. This is set to True by default for return_sequences=False and to                                           False for  return_sequences=True. If set to True, then nb_steps/3 patches are required as a                                             minimum.  
**psy**                                 When return_sequences=True, this is the proportion of Local Patches (as per paper description)                                           with respect to the number L of global patches. A typical value is 0.15.  
