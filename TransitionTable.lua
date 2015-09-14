
require 'image'

local trans = torch.class('dqn.TransitionTable')


function trans:__init(args)
    if vector_function == convert_text_to_ordered_list2 then
        self.stateDim = args.stateDim * 2 -- @karthik: double state-dim for desc and quest separately
    else
        self.stateDim  = args.stateDim -- State dimensionality.
    end
    self.numActions = args.numActions
    self.numObjects = args.numObjects
    self.histLen = args.histLen
    self.maxSize = args.maxSize or 1024^2
    self.bufferSize = args.bufferSize or 1024
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.zeroFrames = args.zeroFrames or 1
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0
    self.insertIndex = 0

    self.histIndices = {}
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.o = torch.LongTensor(self.maxSize):fill(0) --objects
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.available_objects = torch.zeros(self.maxSize, self.numObjects)
    self.action_encodings = torch.eye(self.numActions)
    self.object_encodings = torch.eye(self.numObjects)

    --structure for storing priority state indices (of the state itself, without hist size adjustments)
    self.priority_indices = {}

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_o = {}
    self.recent_t = {}

    local s_size = self.stateDim*histLen
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_o      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_available_objects = torch.zeros(self.bufferSize, self.numObjects)  

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end


function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
end


function trans:size()
    return self.numEntries
end


function trans:empty()
    return self.numEntries == 0
end


function trans:fill_buffer(priority_ratio)
    -- assert(self.numEntries >= self.bufferSize) --@karthik: for priority sweeping
    -- clear CPU buffers
    self.buf_ind = 1
    local ind, priority
    for buf_ind=1,self.bufferSize do
        if torch.rand(1)[1] < priority_ratio then
            priority = true
        else
            priority = false
        end
        local s, a, o, r, s2, term, available_objects = self:sample_one(priority)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_o[buf_ind] = o
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
        self.buf_available_objects[buf_ind] = available_objects
    end
    self.buf_s  = self.buf_s:float()
    self.buf_s2 = self.buf_s2:float()
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
    end
end


function trans:sample_one(priority)
    assert(self.numEntries > 1)
    local index = nil
    local valid = false
    while not valid do
        -- start at 2 because of previous action
        if priority and #self.priority_indices > 0 then
            while not index or index > self.numEntries-self.recentMemSize do
                index  = self.priority_indices[torch.random(1,#self.priority_indices)]
            end
            index = index - self.recentMemSize + 1 -- to account for histSize
            -- print("Choosing priority action", index, #self.priority_indices)
        else
            index = torch.random(2, self.numEntries-self.recentMemSize)
        end
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal states with probability (1-nonTermProb).
            -- Note that this is the terminal flag for s_{t+1}.
            valid = false
        end
        if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
            self.r[index+self.recentMemSize-1] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal or non-reward states with
            -- probability (1-nonTermProb).
            valid = false
        end
    end

    return self:get(index)
end


function trans:sample(batch_size, priority_ratio)
    priority_ratio = priority_ratio or 0.5
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_buffer(priority_ratio)
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_o, buf_r, buf_term, buf_available_objects = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_o, self.buf_r, self.buf_term, self.buf_available_objects
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_o[range], buf_r[range], buf_s2[range], buf_term[range], buf_available_objects[range]
end


function trans:concatFrames(index, use_recent)
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end

    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])
    end

    return fullstate
end

-- not used in this file
function trans:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    local obj_hist = torch.FloatTensor(self.histLen, self.numObjects)
    if use_recent then
        a, o, t = self.recent_a, self.recent_o, self.recent_t
    else
        a, o, t = self.a, self.o, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
            obj_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
        obj_hist[i]:copy(self.object_encodings[o[index+self.histIndices[i]-1]])
    end

    return act_hist, obj_hist
end


function trans:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames(1, true):float()
end


function trans:get(index)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1

    return s, self.a[ar_index], self.o[ar_index], self.r[ar_index], s2, self.t[ar_index+1], self.available_objects[ar_index]
end


function trans:add(s, a, o, r, term, available_objects)
    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(o, 'Object cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end

    --check if insertIndex is in priorityIndex, if so then remove it
    if self.insertIndex == self.priority_indices[1] then
        table.remove(self.priority_indices, 1)
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float()
    self.a[self.insertIndex] = a
    self.o[self.insertIndex] = o
    self.r[self.insertIndex] = r
    self.available_objects[self.insertIndex] = available_objects
    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end

    --add to priorityIndices if reward is positive
    if r > 0 then
        -- print("adding priority index", self.insertIndex, #self.priority_indices)
        table.insert(self.priority_indices, self.insertIndex)
    end
end


function trans:add_recent_state(s, term)
    local s = s:clone():float():byte()
    if #self.recent_s == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
        end
    end

    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans:add_recent_action(a, o)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
            table.insert(self.recent_o, 1)
        end
    end

    table.insert(self.recent_a, a)
    table.insert(self.recent_o, o)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
        table.remove(self.recent_o, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty transition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.numObjects,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:read(file)
    local stateDim, numActions, numObjects, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.numObjects = numObjects
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.o = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)
    self.object_encodings = torch.eye(self.numObjects)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_o = {}
    self.recent_t = {}

    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_o      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end
