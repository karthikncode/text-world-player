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
		parts = line:split(" ")
		wordVec[parts[1]] = _.rest(parts)
	end
	return wordVec
end

-- override (zero out NULL INDEX)
function nn.LookupTable:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end

	 --zero out NULL_INDEX
	local output = self.output:clone()
  for i=1, input:size(1) do
		if input[i] == #symbols+1 then
			output[i]:mul(0)
		end
	end

	self.output = output

   return self.output
end


n_hid = 20
nIndex = 2000 -- vocab size
EMBEDDING = nn.LookupTable(nIndex, n_hid)
