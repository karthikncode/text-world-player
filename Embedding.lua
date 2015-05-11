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


n_hid = 20
nIndex = 100 -- vocab size 
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