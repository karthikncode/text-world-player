-- Layer to create quests and act as middle-man between Evennia and Agent

local DEFAULT_REWARD = -1

quests = {'You are hungry.', 'You are sleepy.', 'You are bored.', 'You are getting fat.'}
quest_actions = {'eat', 'sleep', 'watch' ,'exercise'}
quest_index = torch.random(1, #quests)


--Function to parse the output of the game (to extract rewards, etc. )
function parse_game_output(text)
	-- extract REWARD if it exists
	-- text is a list of sentences
	local reward = nil
	local text_to_agent = {quests[quest_index]}
	for i=1, #text do
		if string.match(text[i], "REWARD") then
			print(text[i])
			if string.match(text[i], quest_actions[quest_index]) then
				reward = string.match(text[i], "%d+")
			end
		else
			table.insert(text_to_agent, text[i])
		end

	end
	if not reward then
		reward = DEFAULT_REWARD
	end
	return text_to_agent, reward	
end

-- build game command to send to the game
function build_command(action, object)
	return action .. ' ' ..object
end

