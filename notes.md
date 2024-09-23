model benchmarking: 

We don't rely on the fancy MMLMU type benchmarkings and all that kind of stuff. That will be useful for getting money or getting a paper out but not for actually knowing we are improving the model. 

What we will do is first the classical approach of train-test splitting the synthetic data and looking at the loss on the training set. 

Then the great benchmarking is with finetuning.

Let's say we have a pretrained llama3 which we finetune on dataset D directly. Then let's say we have a pretrained llama3 that has been subsequently pretrained on synthetic data ("synthetic self-tuned") and we then finetune it on the same dataset D as before. If we see that this selftuned llama3 will generalize better and achieve a better test loss than the vanilla llama3 we'll hit the bingo. 