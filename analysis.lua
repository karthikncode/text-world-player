--- file to perform some analysis

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

-- function NN(a)
-- 	for i, v in pairs(vec) do
-- 		if i ~= a then

-- end