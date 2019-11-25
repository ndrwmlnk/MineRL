# MineRL Agents

This repository contains scripts to run trained ppo and rainbow agents. 

The [models](https://github.com/ndrwmlnk/MineRL/tree/master/models) directory contains the trained models for the two different agents.
By running `test_ppo.py` and `test_rainbow.py` you can test the two agents respectively.

## Rainbow agent
When testing the rainbow agent, the q-values are printed at every step and steps are executed with a 0.5 second delay in order to enable you to inspect how the values change in different situations. The q-values are accessed via [`action_value.q_values`](https://github.com/ndrwmlnk/MineRL/blob/a0d5dfe661d077e96f721b48c626f0c57ddcfe77/test_rainbow.py#L124)
When looking at the q-values while testing the agent, you can see that all q-values are higher when the agent is standing in front of a tree.
Based on this, the script `test_rainbow.py` also contains the approach of detecting when the agent stands in front of a tree based on the difference in average q-values. the function `agent.get_statistics()` provides the average q-value at a given step and by observing how this value changes, I noticed that the difference in average q-values is higher (greater than ~0.002) when the agent is standing in front of a tree. When this threshold is surpassed, the message "CHOP TREE" is printed to the terminal. This might be used for chopping trees in the `MineRLObtainDiamond-v0` environment. A demonstration of this can be seen in a video [here](https://github.com/ndrwmlnk/MineRL/tree/master/videos).
Another, more versatile way to go about this might be to decide when the agent is standing in front of a tree based on the individual q-values that are now printed at every step.

**EDIT**
Now the state and advantage values can be accessed at any step from the [`agent.act()`](https://github.com/ndrwmlnk/MineRL/blob/b061573894f6b8a543fff9864588584599932f17/test_rainbow.py#L120) function. The state value can be used to detect when the agent is standing in front of a tree. I included a new video in the [video folder](https://github.com/ndrwmlnk/MineRL/tree/master/videos) where you can see that the state value is higher than ~400 if and only if the agent is standing in front of a tree. [Here](https://github.com/ndrwmlnk/MineRL/blob/b061573894f6b8a543fff9864588584599932f17/test_rainbow.py#L136) I included an idea how you may approach using this information. I chose a value of 450 here because in an experiment where I set this value to 400, the agent occasionally stopped too early and could not reach the tree yet so you might have to experiment with adjusting this value but in general, this should work.
