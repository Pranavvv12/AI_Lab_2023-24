# Ex.No: 4   Implementation of Alpha Beta Pruning 
### DATE:    23-09-2024                                                                    
### REGISTER NUMBER : 212222220052
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

### Program:

```
import math
def minimax (curDepth, nodeIndex, maxTurn, scores,targetDepth):
  # base case : targetDepth reached
  if (curDepth == targetDepth):
    return scores[nodeIndex]
  if (maxTurn):
    return max(minimax(curDepth + 1, nodeIndex * 2,False, scores,
targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,False, scores,
targetDepth))
  else:
    return min(minimax(curDepth + 1, nodeIndex * 2, True, scores,
targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,True, scores,
targetDepth))
# Driver code
scores = [3, 5, 2, 9, 12, 5, 23, 20]
treeDepth = math.log(len(scores), 2) # calculate depth of node log 8 (base2) = 3)
print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))

```


### Output:

![image](https://github.com/santhoshkumar24263/AI_Lab_2023-24/assets/127171952/08ce753c-b552-46a0-a474-d77c847cda7e)


### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
