This repo is based off of early work done on Q-Learning 


For this assigned project we were asked to build a parkinglot representation like in the one in the image below.

<p align="center">
    <img src="https://github.com/DenimHorton/ParkinglotANN/blob/master/Images/a%20simple%20parkinglot.PNG">
</p>

  The code written for this project was also written in a way where the agent must park as a spcifically desginated parking spot and with a envviorment (parkinglot) that can be easily manipulated with move barriers, parking spaces, and even the reward for the different states can be changed if desired.  Even the starting state of the agent can be manipulated easily  Regardless of the shape of the parking lot or the location and amount of barriers and parking spaces in the parkinglot there are some rules that always apply to the agent navigating the enviroment.  
  
1) First: the agent is only allowed four actions, 'Up', 'Down', 'Left', or 'Right'
2) Second: the agent is not allowed to move into the barries of the parkinglot.  
3) Third: the agent can not move pull through parking lot.  In other words, the agent can only pull into a parking space from on direction 
