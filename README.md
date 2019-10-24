# HW2_Q-learning

## Code

* Adaptive epsilon-greedy

![](https://i.imgur.com/fnAPgPL.png)

First, use **epsilong_decay** to control its decaying rate. Secondly, keep it bigger than a threshold via **MAX** function.

* Action

![](https://i.imgur.com/pIuvVGK.png)

* Updating Q-table

![](https://i.imgur.com/2puOwLN.png)

Just follow the definition. Nothing special.

## Results

This graph below shows how many steps we need before arriving the goal at each episode.
![](https://i.imgur.com/eBR3eB9.png)

This graph shows "# of rewards/# of episodes". According to the graph, we are getting better as time goes on. If we kept running the program, finally, it would be very close to 1.

![](https://i.imgur.com/4Xzz1nr.png)

### Trajectory
![](https://i.imgur.com/ZQTgWlS.png)
![](https://i.imgur.com/kiKUDGS.png)






