require 'nn'
require 'utils'
local _ = require 'underscore'


-- read in word vectors - one per line
function readWordVec(filename)
	local file = io.open(filename, "r");
	local data = {}
	local parts
	local wordVec = {} -- global
	for line in file:lines() do
		parts = split(line, " ")	
		wordVec[parts[1]] = _.rest(parts)
	end
	return wordVec
end


-- override (zero out NULL INDEX)
function nn.LookupTable:updateOutput(input)
   -- make sure input is a contiguous torch.LongTensor
	if (not input:isContiguous()) or torch.type(input) ~= 'torch.LongTensor' then
	  self._indices = self._indices or torch.LongTensor()
	  self._indices:resize(input:size()):copy(input)
	  input = self._indices
	end

	if input:dim() == 1 then
	  local nIndex = input:size(1)
	  self.size[1] = nIndex
	  self.output:index(self.weight, 1, input)
	elseif input:dim() == 2 then
	  local nExample = input:size(1)
	  local nIndex = input:size(2)
	  self.batchSize[1] = nExample
	  self.batchSize[2] = nIndex
	  
	  self._inputView = self._inputView or torch.LongTensor()
	  self._inputView:view(input, -1)
	  self.output:index(self.weight, 1, self._inputView)
	  self.output = self.output:view(nExample, nIndex, self.size[2])
	end

   --zero out NULL_INDEX
	local output = self.output:clone()	
  for i=1, input:size(1) do
		if input[i] == #symbols+1 then
			output[i]:mul(0)
		end	
	end  	

   return output
end


n_hid = 100
nIndex = 1500 -- vocab size 
EMBEDDING = nn.LookupTable(nIndex, n_hid)
-- local norm = EMBEDDING.weight:sum()/nIndex
-- EMBEDDING.weight:div(norm) -- zero out initial weights
-- print("init embedding sum",norm)

-- init with word vec

-- local wordVec = readWordVec(WORDVEC_FILE)
-- for i=1, #symbols do
-- 	print(wordVec[symbols[i]])
-- 	EMBEDDING.weight[i] = torch.Tensor(wordVec[symbols[i]])
-- 	assert(EMBEDDING.weight[i]:size(1) == n_hid)
-- end