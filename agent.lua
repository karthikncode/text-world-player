-- agent

require 'client'
local framework = require 'framework.lua'
require 'utils'
require 'xlua'
require 'optim'
require 'hdf5'

---------------------------------------------------------------

if not dqn then
	dqn = {}
	require 'nn'
	require 'nngraph'
	require 'nnutils'
	require 'Scale'
	require 'NeuralQLearner'
	require 'TransitionTable'
	require 'Rectifier'
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-exp_folder', '', 'name of folder where current exp state is being stored')

cmd:option('-framework', '', 'name of training framework')

cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-game_num', 1, 'game number (for parallel game servers)')


cmd:text()

local opt = cmd:parse(arg)
print(opt)


--  agent login
local port = 4000 + opt.game_num
print(port)
client_connect(port)
login('root', 'root')

framework.makeSymbolMapping('../text-world/evennia/contrib/text_sims/build.ev')


--- General setup.

if opt.agent_params then
    opt.agent_params = str_to_table(opt.agent_params)
    opt.agent_params.gpu       = opt.gpu
    opt.agent_params.best      = opt.best
    opt.agent_params.verbose   = opt.verbose
    if opt.network ~= '' then
        opt.agent_params.network = opt.network
    end

    opt.agent_params.actions = framework.getActions()
	opt.agent_params.objects = framework.getObjects()

    if string.match(opt.agent_params.network, "bow_embedding") then
        if vector_function == convert_text_to_bow2 then            
            opt.agent_params.state_dim = 2 * (#symbols)
        else
            opt.agent_params.state_dim = (#symbols)
        end
    end
end	
print("state_dim", opt.agent_params.state_dim)
local agent = dqn[opt.agent](opt.agent_params) -- calls dqn.NeuralQLearner:init

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local bestq_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

local state, reward, terminal = framework.newGame() 

print("Started RL based training ...")
local pos_reward_cnt = 0


print('[Start] Network weight sum:',agent.w:sum())

while step < opt.steps do
    step = step + 1
    xlua.progress(step, opt.steps)

    local action_index, object_index = agent:perceive(reward, state, terminal)

    if reward > 0 then 
        pos_reward_cnt = pos_reward_cnt + 1
    end
        
    -- game over? get next game!
    if not terminal then
        state, reward, terminal = framework.step(action_index, object_index)
    else
        -- if opt.random_starts > 0 then
        --     state, reward, terminal = framework.nextRandomGame()
        -- else
        state, reward, terminal = framework.newGame()
        -- end
    end

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("\nSteps: ", step, " | Achieved quest level, current reward:" , pos_reward_cnt)
        agent:report()
        pos_reward_cnt = 0
        collectgarbage()
    end

    if step%1000 == 0 then 
        collectgarbage() 
    end

		--Testing
    if step % opt.eval_freq == 0 and step > learn_start then
        print('Testing Starts ... ')
        test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
        test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))

        gameLogger = gameLogger or io.open(paths.concat(opt.exp_folder, 'game.log'), 'w')

        state, reward, terminal = framework.newGame(gameLogger)

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            xlua.progress(estep, opt.eval_steps)
            local action_index, object_index, q_func = agent:perceive(reward, state, terminal, true, 0.05)

             -- print Q function for previous state
            if q_func then
                gameLogger:write(table.tostring(q_func), '\n')
            else
                gameLogger:write("Random action\n")
            end

            -- Play game in test mode (episodes don't end when losing a life)
		        state, reward, terminal = framework.step(action_index, object_index, gameLogger)

           

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                state, reward, terminal = framework.newGame(gameLogger)
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "V avg:", v_history[ind])

        --saving and plotting
        test_avg_R:add{['% Average Reward'] = total_reward}
        test_avg_Q:add{['% Average Q'] = agent.v_avg}
        
        test_avg_R:style{['% Average Reward'] = '-'}; test_avg_R:plot()
        test_avg_Q:style{['% Average Q'] = '-'}; test_avg_Q:plot()

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
        print('Testing Ends ... ')
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            print('Network weight sum:', w:sum())
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
end

