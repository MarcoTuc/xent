
# Highlighting synthetic data guidelines

- Take two pre-prompts: X1 and X2. 
- Then take a phrase Y. 

- Then compute <D(Y) = H(Y|X1) - H(Y|X2)> for all the tokens in Y. 
    > When a token in Y is likely to happen given X, its H(Y|X) will be lower. 

- When at token y€Y we have D(y) < 0 then it is more likely for X1 and viceversa. 

- Use this mechanism to construct a dataset with elements of the following form: 

### Higlighting task

>>> TEXT <<<
@$@$@red-blue [beg_y, end_y] | [beg_x1, end_x1] | [beg_x2, end_x2]
{series of tokens from text in decreasing order given their D(y)}


---------------------------------
Interpret @$@$@red-blue as the name of a function with three arguments which are the beginning and end tokens of the three sentences one has to process. Then the function should give back the tokens of Y in decreasing order relative to their D(Y) value. 