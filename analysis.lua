--- file to perform some analysis
manifold = require 'manifold'
stats = torch.load(arg[1])

vec = stats.embeddings

--normalize
for i, val in pairs(vec) do
	local norm = vec[i]:norm()
	if norm > 0 then
		vec[i]:div(norm)
	end
end

function dot(a, b)
	return torch.dot(vec[a], vec[b])
end

function nearest_neighbors()
	for i, v in pairs(vec) do
		local maxDot = -10
		local NN = i
		for j, w in pairs(vec) do
			if j ~= i then
				if torch.dot(v,w) > maxDot then
					maxDot = torch.dot(v,w)
					NN = j
				end
			end
		end
		print(i, NN ,maxDot)
	end
end

function find_len(table)
	local cnt = 0
	for k, v in pairs(table) do
		cnt = cnt+1
	end
	return cnt
end

function plot_tsne(vec)
	local n = find_len(vec)
	local m = torch.zeros(n-1, vec['you']:size(1))
	local i = 1
	local symbols = {}
	for k, val in pairs(vec) do
		if k~='NULL' then
			m[i] = vec[k]
			symbols[i] = k
			i = i+1
		end
	end
  opts = {ndims = 2, perplexity = 50, pca = 50, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(m)
  return mapped_x1, symbols
end

tsne, symbols = plot_tsne(vec)
--write
local file = io.open('tsne.txt', "w");
for i=1, #symbols do
	file:write(symbols[i] .. ' ' .. tsne[i][1]  .. ' ' .. tsne[i][2] .. '\n')
end





