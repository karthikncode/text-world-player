require 'nn'

-- override for debug
-- function nn.LookupTable:accGradParameters(input, gradOutput, scale)
-- 	-- print(input, gradOutput, scale)
-- 	-- print("before", self.gradWeight:norm())
--    scale = scale or 1
--    if input:dim() == 1 then
--       self.nBackward = self.nBackward + 1
--       for i=1,input:size(1) do
--          local k = input[i]
--          self.inputs[k] = (self.inputs[k] or 0) + 1
--          self.gradWeight:select(1, k):add(scale, gradOutput:select(1, i))
--       end
--    elseif input:dim() == 2 then
--       self.nBackward = self.nBackward + input:size(1)
--       for i=1,input:size(1) do
--          local input = input:select(1, i)
--          local gradOutput = gradOutput:select(1, i)
--          for j=1,input:size(1) do
--             local k = input[j]
--             self.inputs[k] = (self.inputs[k] or 0) + 1
--             self.gradWeight:select(1, k):add(scale, gradOutput:select(1, j))
--          end
--       end
--    end
--    -- print("after", self.gradWeight:norm())   
-- end



n_hid = 100
nIndex = 128 -- vocab size 
EMBEDDING = nn.LookupTable(nIndex, n_hid)
-- local norm = EMBEDDING.weight:sum()/nIndex
-- EMBEDDING.weight:div(norm) -- zero out initial weights
-- print("init embedding sum",norm)

EMBEDDING.weight[#symbols+1] = torch.zeros(n_hid) -- NULL_INDEX