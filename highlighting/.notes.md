Three tasks of increasing complexity: 

1. 
Take wikipedia articles as Y
Take wikipedia titles as X1 and X2 
Set a lower quality threshold for how much X1 and X2 should be representative:
If avg highlighting is below some threshold discard the sample. 

2. 
Generate Y with the trained model 
Take wikipedia titles as X1 and X2 

3. 
Generate a long text 
Pick Y, X1, X2 as random windows inside of it
