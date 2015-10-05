require 'utils'
require 'nn'
require 'rnn'
require 'nngraph'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    if vector_function == convert_text_to_ordered_list2 then
        self.state_dim_multiplier = 2
    else
        self.state_dim_multiplier = 1
    end
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.objects    = args.objects
    self.n_objects  = #self.objects
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local err, msg = pcall(require, self.network)
    if not err then
        print('Preloading network file:', self.network)
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = msg
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Float or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions, numObjects = self.n_objects,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastObject = nil
    self.lastAvailableObjects = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta, o, available_objects
    local q, q2, q2_max

    s = args.s
    a = args.a
    o = args.o
    r = args.r
    s2 = args.s2
    term = args.term
    available_objects = args.available_objects -- this is for s2

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute {max_a Q(s_2, a), max_o Q(s_2, o)}.
    if RECURRENT == 0 then
        q2_max = target_q_net:forward(s2)
    else
        local s2_tmp = tensor_to_table(s2, self.state_dim, self.hist_len)
        q2_max = target_q_net:forward(s2_tmp)
    end

    q2_max[1] = q2_max[1]:float():max(2) --actions
    q2_max[2] = q2_max[2]:float() -- objects

    q2_max[2]:cmul(available_objects:float())
    q2_max[2][q2_max[2]:eq(0)] = -1e30
    q2_max[2] = q2_max[2]:max(2)

    q2_max = (q2_max[1]+q2_max[2])/2 --take avg. of action and object

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = {}
    q2[1] = q2_max:clone():mul(self.discount):cmul(term)
    q2[2] = q2_max:clone():mul(self.discount):cmul(term)


    delta = {r:clone():float(), r:clone():float()}

    delta[1]:add(q2[1])
    delta[2]:add(q2[2])

    -- q = Q(s,a)
    local q_all
    if RECURRENT == 0 then
        q_all = self.network:forward(s)
    else
        local s_tmp = tensor_to_table(s, self.state_dim, self.hist_len)
        -- print(s_tmp)
        q_all = self.network:forward(s_tmp)
    end


    q_all[1] = q_all[1]:float()
    q_all[2] = q_all[2]:float()

    q = {torch.FloatTensor(q_all[1]:size(1)), torch.FloatTensor(q_all[2]:size(1))}
    for i=1,q_all[1]:size(1) do
        q[1][i] = q_all[1][i][a[i]]
    end
    for i=1,q_all[2]:size(1) do
        q[2][i] = q_all[2][i][o[i]]
    end

    delta[1]:add(-1, q[1])
    delta[2]:add(-1, q[2])

    if self.clip_delta then
        delta[1][delta[1]:ge(self.clip_delta)] = self.clip_delta
        delta[1][delta[1]:le(-self.clip_delta)] = -self.clip_delta
        delta[2][delta[2]:ge(self.clip_delta)] = self.clip_delta
        delta[2][delta[2]:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = {torch.zeros(self.minibatch_size, self.n_actions):float(),
                    torch.zeros(self.minibatch_size, self.n_objects):float()}
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[1][i][a[i]] = delta[1][i]
    end
    for i=1,math.min(self.minibatch_size,o:size(1)) do
        targets[2][i][o[i]] = delta[2][i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw

    local priority_ratio = 0.25-- fraction of samples from 'priority' transitions
    local s, a, o, r, s2, term, available_objects = self.transitions:sample(self.minibatch_size, priority_ratio)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, o=o, r=r, s2=s2,
        term=term, update_qmax=true, available_objects=available_objects}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    if RECURRENT == 0 then
        self.network:backward(s, targets)
    else
        local s_tmp = tensor_to_table(s, self.state_dim, self.hist_len)
        self.network:backward(s_tmp, targets)
    end

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    --grad normalization
    -- local max_norm = 50
    -- local grad_norm = self.dw:norm()
    -- if grad_norm > max_norm then
    --   local scale_factor = max_norm/grad_norm
    --   self.dw:mul(scale_factor)
    --   if false and grad_norm > 1000 then
    --       print("Scaling down gradients. Norm:", grad_norm)
    --   end
    -- end

    -- use gradients (original)
    -- self.g:mul(0.95):add(0.05, self.dw)
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:mul(0.95):add(0.05, self.tmp) -- g2 is g squared
    -- self.tmp:cmul(self.g, self.g)
    -- self.tmp:mul(-1)
    -- self.tmp:add(self.g2)
    -- self.tmp:add(0.01)
    -- self.tmp:sqrt()

    --rmsprop
    local smoothing_value = 1e-8
    self.tmp:cmul(self.dw, self.dw)
    self.g:mul(0.9):add(0.1, self.tmp)
    self.tmp = torch.sqrt(self.g)
    self.tmp:add(smoothing_value)  --negative learning rate

    --AdaGrad
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:cmul(self.g, self.g)
    -- self.g2:add(self.tmp)
    -- self.g = torch.sqrt(self.g2)
    -- self.tmp = self.g:clone()



    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)

    self.w:add(self.deltas)

    -- print(self.network:parameters())
    -- -- print(self.deltas:size(), self.w:size(), self.dw:size())
    -- -- print(self.deltas:eq(0):sum(), self.w:eq(0):sum(), self.dw:eq(0):sum())
    -- EMBEDDING:updateParameters(self.lr)
    -- print("Deltas: ", self.deltas:norm())
    -- print("W:", self.w:norm())
    -- print("Embedding:",EMBEDDING.weight:norm())
    -- print("Embedding grad weight:",EMBEDDING.gradWeight:norm())
    -- print("Embedding2:",EMBEDDING:forward(torch.range(1, #symbols+1)):norm())
    -- assert(EMBEDDING.gradWeight:norm() > 0)
end


function nql:sample_validation_data()
    local s, a, o, r, s2, term, available_objects = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_o    = o:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
    self.valid_available_objects = available_objects:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, o = self.valid_o, r=self.valid_r, s2=self.valid_s2,
        term=self.valid_term, available_objects=self.valid_available_objects}

    self.v_avg = self.q_max * (q2_max[1]:mean() + q2_max[2]:mean())/2
    self.tderr_avg = (delta[1]:clone():abs():mean() + delta[2]:clone():abs():mean())/2
end


function nql:perceive(reward, state, terminal, testing, testing_ep, available_objects, priority)
    -- Preprocess state (will be set to nil if terminal)
    local curState

    --bounding box for rewards
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    --Store transition s, a, r, s' (only if not testing)
    -- IMP: available_objects is for selecting an action at s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, self.lastObject, reward,
                             self.lastTerminal, table_to_binary_tensor(available_objects, self.n_objects))
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent() -- IMP: curState is with history - not same as 'state'

    -- Select action
    local actionIndex = 1
    local objectIndex = 1
    if not terminal then
        actionIndex, objectIndex, q_func = self:eGreedy(curState, testing_ep, available_objects)
        -- actionIndex, objectIndex, q_func = self:sample_action(curState, testing_ep, available_objects)
    end

    self.transitions:add_recent_action({actionIndex, objectIndex}) --not used

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastObject = objectIndex
    self.lastTerminal = terminal
    self.lastAvailableObjects = available_objects

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex, objectIndex, q_func
    else
        return 0, 0
    end
end

-- sample an action and object instead of just taking max
function nql:sample_action(state, testing_ep, available_objects)
   if self.gpu >= 0 then
        state = state:cuda()
    end
    local q
    if RECURRENT == 0 then
        q = self.network:forward(state)
    else
        -- print("state", state)
        local state_tmp = tensor_to_table(state, self.state_dim, self.hist_len)
        -- print("state_tmp", state_tmp)
        q = self.network:forward(state_tmp)
    end

    q[1] = q[1]:float():squeeze()
    q[2] = q[2]:float():squeeze()

    orig_q = {q[1]:clone(), q[2]:clone()}

    q[1] = q[1]:double()
    q[2] = q[2]:double()
    q[1] = q[1] - q[1]:min()
    q[2] = q[2] - q[2]:min()

    q[1]:exp()
    -- q[1]:div(q[1]:norm())
    -- convert available objects to tensor
    available_objects = table_to_binary_tensor(available_objects, self.n_objects)
    q[2]:exp()
    q[2]:cmul(available_objects)
    -- q[2]:div(q[2]:norm())

    -- if q[1]:sum()<=0 or q[1]:max() > 1e3 or q[1]:min() < -1e3 then
    --     print("q[1]", q[1])
    -- end

    -- if q[2]:sum()<=0 or q[2]:max() > 1e3 or q[2]:min() < -1e3 then
    --     print("q[2]", q[2])
    -- end

    local status, sample_a = pcall(torch.multinomial, q[1], 1)
    if not status then
        print(q[1], orig_q[1])
        print(q[2], orig_q[2])
        assert(false)
    else
        sample_a = sample_a[1]
    end
    local sample_o = torch.multinomial(q[2], 1)[1]

    self.lastAction = sample_a
    self.lastObject = sample_o
    self.bestq = (q[1][sample_a] + q[2][sample_o])/2

    return sample_a, sample_o, q
end



function nql:eGreedy(state, testing_ep, available_objects)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))

    -- print("EPS:", self.ep)
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        if not available_objects then
            return torch.random(1, self.n_actions), torch.random(1, self.n_objects)
        else
            return torch.random(1, self.n_actions), available_objects[math.random(#available_objects)]
        end
    else
        return self:greedy(state, available_objects)
    end
end

-- Evaluate all actions (with random tie-breaking)
function nql:getBestRandom(q, N, available_objects)
    if available_objects then
        -- print("avail OBJECTSSSS",available_objects)
        local maxq = q[available_objects[1]]
        local besta = {available_objects[1]}
        for i=2, #available_objects do
            a = available_objects[i]
            if q[a] > maxq then
                besta = { a }
                maxq = q[a]
            elseif q[a] == maxq then
                besta[#besta+1] = a
            end
        end
        local r = torch.random(1, #besta)
        -- print("best object:", besta[r], maxq, q)
        return besta[r], maxq
    end

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, N do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end

    local r = torch.random(1, #besta)

    return besta[r], maxq
end


function nql:greedy(state, available_objects)
    if self.gpu >= 0 then
        state = state:cuda()
    end
    local q
    if RECURRENT == 0 then
        q = self.network:forward(state)
    else
        -- print("state", state)
        local state_tmp = tensor_to_table(state, self.state_dim, self.hist_len)
        -- print("state_tmp", state_tmp)
        q = self.network:forward(state_tmp)
    end

    q[1] = q[1]:float():squeeze()
    q[2] = q[2]:float():squeeze()

    local best = {}
    local maxq = {}
    best[1], maxq[1] = self:getBestRandom(q[1], self.n_actions)
    best[2], maxq[2] = self:getBestRandom(q[2], self.n_objects, available_objects)

    self.lastAction = best[1]
    self.lastObject = best[2]
    self.bestq = (maxq[1] + maxq[2])/2

    local avail_obj_tensor = table_to_binary_tensor(available_objects, self.n_objects):float()
    q[2] = q[2]:clone():cmul(avail_obj_tensor)

    return best[1], best[2], q
end

function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
