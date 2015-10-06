Prerequisites
--------------
[Torch](http://torch.ch/docs/getting-started.html)  
Also make sure you've setup the game environment using [these instructions](https://github.com/mrkulk/text-world).

Lua Package requirements  
---------------
1. [luasocket](http://w3.impa.br/~diego/software/luasocket/)  
2. [rnn](https://github.com/Element-Research/rnn)  
3. underscore  

Most lua packages can be installed using `luarocks install <pkg>`

Runtime options
----------------
`run_cpu` has the list of user-defined settings used by the program. You can run the script as `./run_cpu <game_num>` where \<game_num\> would be 1,2,3,etc. This allows us to choose the game server if multiple are running at the same time. See the script `start.sh` in the [text-world code](https://github.com/mrkulk/text-world/tree/master/evennia) for details on starting multiple game servers. 

The main options you should care about inside the `run_cpu` file are:  
1. (**important**) text_world_location: Set this to the location of the text-world directory on your machine. You should have cloned the directory from [here](https://github.com/mrkulk/text-world).   
2. STEP_SIZE: This defines the number of steps taken by the agent in the game
in an epoch.   
3. max_steps: Maximum number of steps per episode of gameplay.  
4. recurrent: Set this to 1 if using the LSTM for the Representation Generator.  
5. bigram: Set this to 1 to use a bag-of-bigrams representation.  
6. netfile: Choose the model to use for the Representation Generator. Remember to set the `recurrent` option appropriately.
