--BOW
require 'nn'
require 'nnx'
require 'nngraph'

--require 'cunn'
-- IMP if args is not passed, it takes from global 'args'

return function(args)

    -- --override for debug
    -- function nn.Sequencer:updateGradInput(inputTable, gradOutputTable)
    --     -- for i=1, #gradOutputTable do
    --         -- print(gradOutputTable[i]:sum())
    --     -- end
    --    self.gradInput = {}
    --    if self.isRecurrent then
    --       assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
    --       assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
    --       for step, input in ipairs(inputTable) do
    --          self.module.step = step + 1
    --          self.module:updateGradInput(input, gradOutputTable[step])
    --       end
    --       -- back-propagate through time (BPTT)
    --       self.module:updateGradInputThroughTime()
    --       assert(self.module.gradInputs, "recurrent module did not fill gradInputs")
    --       for step=1,#inputTable do
    --          self.gradInput[step] = self.module.gradInputs[step]
    --       end
    --       assert(#self.gradInput == #inputTable, "missing gradInputs")
    --    else
    --       for step, input in ipairs(inputTable) do
    --          -- set the output/gradOutput states for this step
    --          local modules = self.module:listModules()
    --          local sequenceOutputs = self.sequenceOutputs[step]
    --          local sequenceGradInputs = self.sequenceGradInputs[step]
    --          if not sequenceGradInputs then
    --             sequenceGradInputs = {}
    --             self.sequenceGradInputs[step] = sequenceGradInputs
    --          end
    --          for i,modula in ipairs(modules) do
    --             local output, gradInput = modula.output, modula.gradInput
    --             local output_ = sequenceOutputs[i]
    --             assert(output_, "updateGradInputThroughTime should be preceded by updateOutput")
    --             modula.output = output_
    --             modula.gradInput = recursiveResizeAs(sequenceGradInputs[i], gradInput)
    --          end

    --          -- backward propagate this step
    --          self.gradInput[step] = self.module:updateGradInput(input, gradOutputTable[step])

    --          -- save the output/gradOutput states of this step
    --          for i,modula in ipairs(modules) do
    --             sequenceGradInputs[i] = modula.gradInput
    --          end
    --       end
    --    end
    --    return self.gradInput
    -- end


    function create_network(args)
        rho = args.state_dim --number of backprop steps
        r = nn.Recurrent(
           n_hid, EMBEDDING,
           nn.Linear(n_hid, n_hid), nn.Rectifier(), --check whether rect or sigmoid
           rho
        )

        rnn = nn.Sequential()

        rnn_seq = nn.Sequential()
        rnn_seq:add(nn.Sequencer(r))
        rnn_seq:add(nn.SelectTable(args.state_dim))
        rnn_seq:add(nn.Linear(n_hid, n_hid))

        -- alternative - considering outputs from all timepoints
        -- rnn_seq:add(nn.JoinTable(2))
        -- rnn_seq:add(nn.Linear(args.state_dim * n_hid, n_hid))

        rnn_seq:add(nn.Rectifier())
        rnn_seq:add(nn.Linear(n_hid, n_hid))
        rnn_seq:add(nn.Rectifier())

        parallel_flows = nn.ParallelTable()
        for f=1, args.hist_len * args.state_dim_multiplier do
            if f > 1 then
                parallel_flows:add(rnn_seq:clone("weight","bias", "gradWeight", "gradBias"))
            else
                parallel_flows:add(rnn_seq)
            end
        end


        local rnn_out = nn.ConcatTable()
        rnn_out:add(nn.Linear(args.hist_len  * args.state_dim_multiplier * n_hid, args.n_actions))
        rnn_out:add(nn.Linear(args.hist_len  * args.state_dim_multiplier * n_hid, args.n_objects))

        rnn:add(parallel_flows)
        rnn:add(nn.JoinTable(2))
        rnn:add(rnn_out)

        if args.gpu >=0 then
            rnn:cuda()
        end
        return rnn

    end

    return create_network(args)
end
