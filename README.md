# MineRL Agents

This repository contains scripts to run trained ppo and rainbow agents. 

The [models](https://github.com/ndrwmlnk/MineRL/tree/master/models) directory containts the trained models for the two different agents.
By running `test_ppo.py` and `test_rainbow.py` you can test the two agents respectively.

## Rainbow agent
When testing the rainbow agent, the q-values are printed at every step and steps are executed with a 0.5 second delay in order to enable you to inspect how the values change in different situations. The q-values are accessed via [`action_value.q_values`](https://github.com/ndrwmlnk/MineRL/blob/a0d5dfe661d077e96f721b48c626f0c57ddcfe77/test_rainbow.py#L124)
When looking at the q-values while testing the agent, you can see that all q-values are higher when the agent is standing in front of a tree.
Based on this, the script `test_rainbow.py` also contains the approach of detecting when the agent stands in front of a tree based on the difference in average q-values. the function `agent.get_statistics()` provides the average q-value at a given step and by observing how this value changes, I noticed that the difference in average q-values is higher (greater than ~0.002) when the agent is standing in front of a tree. When this threshold is surpassed, the message "CHOP TREE" is printed to the terminal. This might be used for chopping trees in the `MineRLObtainDiamond-v0` environment. A demonstration of this can be seen in a video [here](https://github.com/ndrwmlnk/MineRL/tree/master/videos).
Another, more versatile way to go about this might be to decide when the agent is standing in front of a tree based on the individual q-values that are now printed at every step.
